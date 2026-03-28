"""Sphere-tracing renderer using techniques from the Google Research JAX raycast article.

References:
    https://google-research.github.io/self-organising-systems/2022/jax-raycast/
    https://iquilezles.org/articles/rmshadows/
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array


def _normalize(v: Array) -> Array:
    return v / jnp.linalg.norm(v)


def _camera_rays(
    camera_pos: Array,
    look_at: Array,
    resolution: tuple[int, int],
    fov: float,
) -> Array:
    """Build a flat array of unit ray directions for a perspective camera.

    Constructs an orthonormal camera frame (right, down, forward) aligned to
    world-up = [0, 1, 0].  Image rows run top-to-bottom in screen space.

    Args:
        camera_pos: World-space camera position, shape (3,).
        look_at: World-space point the camera looks toward, shape (3,).
        resolution: (height, width) in pixels.
        fov: Half-width field-of-view in world units at unit depth.

    Returns:
        Unit ray directions, shape (H*W, 3).
    """
    h, w = resolution
    forward = _normalize(look_at - camera_pos)
    world_up = jnp.array([0.0, 1.0, 0.0])
    right = _normalize(jnp.cross(forward, world_up))
    down = _normalize(jnp.cross(right, forward))

    fx = fov
    fy = fx / w * h
    # linspace works with abstract fov inside jax.jit (mgrid requires concrete bounds)
    xs = jnp.linspace(-fx, fx, w)
    ys = jnp.linspace(fy, -fy, h)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    y = yy.reshape(-1)
    x = xx.reshape(-1)
    dirs = jnp.stack([x, y, jnp.ones_like(x)], axis=1)
    R = jnp.stack([right, down, forward])
    dirs = dirs @ R
    return dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)


def _cast_shadow(
    sdf: Callable[[Array], Array],
    pos: Array,
    normal: Array,
    light_dir: Array,
    steps: int,
    hardness: float,
) -> Array:
    """Soft shadow via secondary sphere tracing toward the light.

    Uses the Inigo Quilez penumbra technique: the minimum ratio of SDF value
    to ray distance approximates how much light is blocked.

    The shadow ray origin is offset along the surface normal to avoid
    self-intersection with the surface the ray was cast from.

    Args:
        sdf: Signed distance function, callable (3,) → scalar.
        pos: Surface hit point.
        normal: Unit surface normal at pos.
        light_dir: Unit vector toward the light source.
        steps: Number of shadow ray march iterations.
        hardness: Controls shadow sharpness; higher = harder edges.

    Returns:
        Shadow factor in [0, 1]: 0 = fully shadowed, 1 = fully lit.
    """
    # Offset along the normal so the shadow ray starts clearly outside the surface
    origin = pos + normal * 2e-3

    def f(carry, _):
        t, shadow = carry
        h = sdf(origin + light_dir * t)
        # Clamp step to positive to prevent the ray going backwards inside geometry
        return (t + jnp.maximum(h, 1e-5), jnp.clip(hardness * h / t, 0.0, shadow)), None

    (_, shadow), _ = jax.lax.scan(f, (jnp.array(1e-2), jnp.array(1.0)), None, length=steps)
    return shadow


def _sphere_trace(
    sdf: Callable, origin: Array, direction: Array, steps: int
) -> tuple[Array, Array]:
    """Sphere trace from origin along direction for `steps` iterations.

    Returns:
        (t_hit, d_min): t at closest approach and the minimum SDF distance seen.
    """

    def march(carry, _):
        t, t_hit, d_min = carry
        d = sdf(origin + t * direction)
        # Guard against inf/NaN from smooth_min(inf, inf) when rays miss all geometry
        d_safe = jnp.where(jnp.isfinite(d), d, jnp.array(1e9))
        t_next = t + jnp.maximum(d_safe, 1e-5)
        closer = d_safe < d_min
        return (t_next, jnp.where(closer, t, t_hit), jnp.minimum(d_safe, d_min)), None

    (_, t_hit, d_min), _ = jax.lax.scan(
        march, (jnp.array(0.0), jnp.array(0.0), jnp.array(1e9)), None, length=steps
    )
    return t_hit, d_min


def _shade_one_light(
    ldir: Array,
    lcolor: Array,
    sdf: Callable,
    pos: Array,
    normal: Array,
    ray_dir: Array,
    ao: Array,
    base_color: Array,
    specular_color: Array,
    shininess: Array,
    shadow_steps: int,
    shadow_hardness: float,
) -> Array:
    shadow = _cast_shadow(sdf, pos, normal, ldir, shadow_steps, shadow_hardness)
    diffuse = jnp.clip(jnp.dot(normal, ldir), 0.0, 1.0) * shadow
    halfway = _normalize(ldir - ray_dir)
    specular = jnp.clip(jnp.dot(halfway, normal), 0.0, 1.0) ** shininess * shadow
    return lcolor * (base_color * (0.2 * ao + 0.7 * diffuse) + specular_color * 0.3 * specular)


def _shade_surface(
    sdf: Callable,
    mat: dict,
    pos: Array,
    normal: Array,
    ray_dir: Array,
    ao: Array,
    light_dirs: Array,
    light_colors: Array,
    shadow_steps: int,
    shadow_hardness: float,
    ambient: float,
) -> Array:
    """Blinn-Phong shading for a surface hit point.

    Returns:
        RGB color, unweighted by opacity or Fresnel.
    """
    base_color = mat["color"]
    roughness = mat["roughness"]
    metallic = mat["metallic"]
    shininess = jnp.maximum(2.0 / (roughness**2 + 1e-4) - 2.0, 1.0)
    specular_color = jnp.ones(3) * (1.0 - metallic) + base_color * metallic
    per_light = jax.vmap(
        lambda ld, lc: _shade_one_light(
            ld,
            lc,
            sdf,
            pos,
            normal,
            ray_dir,
            ao,
            base_color,
            specular_color,
            shininess,
            shadow_steps,
            shadow_hardness,
        )
    )(light_dirs, light_colors)
    return per_light.sum(0) + base_color * ambient


def _refract(d: Array, n: Array, eta: Array) -> Array:
    """Snell's-law refraction; falls back to reflection on total internal reflection.

    Args:
        d: Incident ray direction (unit vector, pointing toward surface).
        n: Surface normal (unit vector, pointing against incident ray).
        eta: Ratio of incident IOR to transmitted IOR (n1/n2).

    Returns:
        Refracted (or reflected on TIR) ray direction.
    """
    cos_i = -jnp.dot(d, n)
    sin2_t = eta**2 * (1.0 - cos_i**2)
    cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))
    refracted = eta * d + (eta * cos_i - cos_t) * n
    reflected = d - 2.0 * jnp.dot(d, n) * n  # TIR fallback
    return jnp.where(sin2_t >= 1.0, reflected, refracted)


def _fresnel_schlick(cos_theta: Array, ior: Array) -> Array:
    """Schlick approximation for dielectric reflectance.

    Args:
        cos_theta: Cosine of the angle between the ray and the surface normal.
        ior: Index of refraction of the medium being entered.

    Returns:
        Fresnel reflectance in [0, 1].
    """
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - jnp.maximum(cos_theta, 0.0)) ** 5


def _render_pixel(
    sdf: Callable[[Array], Array],
    material_fn: Callable[[Array], dict],
    ray_origin: Array,
    ray_dir: Array,
    light_dirs: Array,
    light_colors: Array,
    max_steps: int,
    max_dist: float,
    shadow_steps: int,
    shadow_hardness: float,
    ambient: float,
    edge_width: float,
    background_color: Array,
    refract_steps: int,
    use_grad_ao: bool = True,
) -> Array:
    """Trace one ray and return its shaded RGB color.

    Pipeline:
      1. Sphere trace to find surface hit (closest-approach tracking).
      2. Compute surface normal via autodiff gradient.
      3. Query material at hit point.
      4. For each light: cast soft shadow, compute Blinn-Phong diffuse + specular,
         weight by light color and material properties, accumulate into RGB.
      5. If refract_steps > 0: bend ray through material using Snell's law,
         march interior, exit, shade background, blend via Fresnel + opacity.
      6. Apply smooth edge coverage blending with background_color.
    """

    # Primary sphere trace
    t_hit, d_min = _sphere_trace(sdf, ray_origin, ray_dir, max_steps)
    pos = ray_origin + t_hit * ray_dir

    # Normal from gradient
    raw_normal = jax.grad(sdf)(pos)
    raw_mag = jnp.linalg.norm(raw_normal)
    normal = raw_normal / jnp.where(raw_mag > 1e-6, raw_mag, 1.0)
    ao = jnp.clip(raw_mag, 0.0, 1.0) if use_grad_ao else jnp.array(1.0)

    # Material at hit point
    mat = material_fn(pos)

    # Surface shading (opacity/Fresnel not yet applied)
    rgb_surface = _shade_surface(
        sdf,
        mat,
        pos,
        normal,
        ray_dir,
        ao,
        light_dirs,
        light_colors,
        shadow_steps,
        shadow_hardness,
        ambient,
    )

    if refract_steps > 0:
        opacity = mat["opacity"]
        ior = mat["ior"]
        cos_i = jnp.maximum(0.0, jnp.dot(-ray_dir, normal))
        fresnel = _fresnel_schlick(cos_i, ior)

        # 1. Bend ray into material (air → glass)
        dir_in = _refract(ray_dir, normal, 1.0 / ior)
        # 2. March interior with -sdf to find back face
        t_exit, _ = _sphere_trace(lambda p: -sdf(p), pos + 1e-3 * dir_in, dir_in, refract_steps)
        exit_pos = pos + 1e-3 * dir_in + t_exit * dir_in
        # 3. Exit normal — flip outward gradient so it opposes dir_in (for Snell's law)
        raw_exit_n = jax.grad(sdf)(exit_pos)
        exit_norm = -raw_exit_n / jnp.linalg.norm(raw_exit_n)
        # 4. Bend ray back into air (glass → air)
        dir_out = _refract(dir_in, exit_norm, ior)
        # 5. March scene from exit point to find what's behind the glass
        bg_origin = exit_pos + 1e-3 * dir_out
        t_bg, _ = _sphere_trace(sdf, bg_origin, dir_out, refract_steps)
        bg_pos = bg_origin + t_bg * dir_out
        bg_hit = t_bg < max_dist
        raw_bg_n = jax.grad(sdf)(bg_pos)
        bg_mag = jnp.linalg.norm(raw_bg_n)
        bg_norm = raw_bg_n / jnp.where(bg_mag > 1e-6, bg_mag, 1.0)
        bg_ao = jnp.clip(bg_mag, 0.0, 1.0)
        bg_mat = material_fn(bg_pos)
        rgb_behind = jnp.where(
            bg_hit,
            _shade_surface(
                sdf,
                bg_mat,
                bg_pos,
                bg_norm,
                dir_out,
                bg_ao,
                light_dirs,
                light_colors,
                shadow_steps,
                shadow_hardness,
                ambient,
            ),
            background_color,
        )
        # Tint transmitted light by glass colour
        rgb_transmitted = rgb_behind * mat["color"]

        # Blend:
        #   opacity=1          → fully opaque (surface only)
        #   opacity=0, head-on → mostly transmitted, small Fresnel highlight
        #   opacity=0, grazing → mostly Fresnel (TIR / strong reflection)
        rgb = (opacity + (1.0 - opacity) * fresnel) * rgb_surface + (1.0 - fresnel) * (
            1.0 - opacity
        ) * rgb_transmitted
    else:
        # Legacy behaviour: opacity fades to black
        rgb = rgb_surface * mat["opacity"]

    # Smooth edge: anti-alias contour by fading to background_color
    coverage = jnp.clip(1.0 - d_min / edge_width, 0.0, 1.0)
    hit_rgb = rgb * coverage + background_color * (1.0 - coverage)
    return jnp.where(t_hit < max_dist, hit_rgb, background_color)


def raymarch(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dirs: Array = jnp.array([0.5, 1.0, 0.3]),
    light_colors: Array | None = None,
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 64,
    max_dist: float = 20.0,
    shadow_steps: int = 24,
    shadow_hardness: float = 8.0,
    gamma: float = 2.2,
    ambient: float = 0.0,
    aa_samples: int = 1,
    background_color: Array = jnp.array([0.0, 0.0, 0.0]),
    refract_steps: int = 0,
) -> np.ndarray:
    """Render an SDF via sphere tracing and return an RGB image array.

    Uses ``jax.vmap`` over pixels and ``jax.lax.scan`` for the inner march
    loop.  Supports multiple colored lights, transparency, and refraction.

    Args:
        sdf: Signed distance function, callable ``(point: Array[3]) → Array[]``.
        camera_pos: Camera position in world space.
        look_at: Point the camera looks toward.
        light_dirs: Light direction(s) toward the light source(s).  Shape
            ``(3,)`` for a single light or ``(N, 3)`` for N lights.
        light_colors: RGB color(s) of the light(s), shape ``(3,)`` or
            ``(N, 3)``.  Defaults to white ``[1, 1, 1]`` for every light.
        resolution: Output image size as (height, width).
        fov: Half-width field-of-view parameter.
        max_steps: Sphere tracing iterations per ray.
        max_dist: Rays beyond this distance are treated as misses.
        shadow_steps: Iterations for the soft shadow ray.
        shadow_hardness: Shadow sharpness (higher = harder edges).
        gamma: Gamma correction exponent (2.2 = sRGB standard).
        ambient: Constant ambient light added to all hit pixels (0 = fully
            black shadows, higher = lifted shadows).
        aa_samples: Super-sampling factor for antialiasing.  1 = no AA;
            2 = 2×2 SSAA (4× rays, box-filtered); 3 = 3×3, etc.
        background_color: RGB color returned for rays that miss all geometry
            and used as the edge fade target.  Default is black.
        refract_steps: Number of interior march steps for glass refraction.
            0 disables refraction (legacy behaviour).  Try 32–64 for glass.

    Returns:
        Float32 numpy array of shape (H, W, 3) with values in [0, 1].
    """
    h, w = resolution

    # Normalise light directions — accept (3,) for a single light
    light_dirs = jnp.atleast_2d(jnp.asarray(light_dirs, dtype=jnp.float32))  # (N, 3)
    light_dirs = light_dirs / jnp.linalg.norm(light_dirs, axis=1, keepdims=True)

    if light_colors is None:
        light_colors = jnp.ones_like(light_dirs)  # white for every light
    else:
        light_colors = jnp.atleast_2d(jnp.asarray(light_colors, dtype=jnp.float32))

    background_color = jnp.asarray(background_color, dtype=jnp.float32)

    # Super-sample: render at N× resolution, box-filter down after shading
    render_res = (h * aa_samples, w * aa_samples)
    rays = _camera_rays(camera_pos, look_at, render_res, fov)  # (H*s*W*s, 3)

    # Edge width scaled to the super-sampled pixel footprint
    scene_dist = float(jnp.linalg.norm(camera_pos - look_at))
    edge_width = 2.0 * fov / min(h * aa_samples, w * aa_samples) * scene_dist

    if hasattr(sdf, "material_at"):
        material_fn = sdf.material_at
    else:
        from jaxcad.render.material import Material

        def material_fn(_p):
            return Material().as_dict()

    # For approximate SDFs (is_exact=False, e.g. Twist) the gradient magnitude
    # is not a reliable AO proxy — use ao=1 (unoccluded) instead.
    use_grad_ao = getattr(sdf, "is_exact", True)

    render_fn = jax.jit(
        jax.vmap(
            lambda ray_dir: _render_pixel(
                sdf,
                material_fn,
                camera_pos,
                ray_dir,
                light_dirs,
                light_colors,
                max_steps,
                max_dist,
                shadow_steps,
                shadow_hardness,
                ambient,
                edge_width,
                background_color,
                refract_steps,
                use_grad_ao,
            )
        )
    )

    rh, rw = render_res
    image = render_fn(rays).reshape(rh, rw, 3)  # (H*s, W*s, 3)

    # Box-filter downsample in linear space before gamma correction
    if aa_samples > 1:
        image = image.reshape(h, aa_samples, w, aa_samples, 3).mean(axis=(1, 3))

    image = jnp.clip(image ** (1.0 / gamma), 0.0, 1.0)
    return np.array(image)


def render_raymarched(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dirs: Array = jnp.array([0.5, 1.0, 0.3]),
    light_colors: Array | None = None,
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 48,
    max_dist: float = 20.0,
    shadow_steps: int = 32,
    shadow_hardness: float = 8.0,
    gamma: float = 1.0,
    ambient: float = 0.0,
    aa_samples: int = 1,
    background_color: Array = jnp.array([0.0, 0.0, 0.0]),
    refract_steps: int = 0,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Render an SDF via sphere tracing and display with matplotlib.

    Wraps :func:`raymarch` and shows the result on a ``plt.Axes``.

    Args:
        sdf: Signed distance function.
        camera_pos: Camera position in world space.
        look_at: Point the camera looks toward.
        light_dirs: Light direction(s), shape ``(3,)`` or ``(N, 3)``.
        light_colors: Light color(s), shape ``(3,)`` or ``(N, 3)``.
            Defaults to white for every light.
        resolution: Output image size as (height, width).
        fov: Half-width field-of-view parameter.
        max_steps: Sphere tracing iterations per ray.
        max_dist: Miss threshold distance.
        shadow_steps: Soft shadow ray iterations.
        shadow_hardness: Shadow edge sharpness.
        gamma: Gamma correction exponent.
        ambient: Constant ambient term (0 = fully black shadows).
        aa_samples: Super-sampling anti-aliasing factor.
        background_color: RGB color for rays that miss all geometry.
        refract_steps: Interior march steps for glass refraction (0 = disabled).
        ax: Existing matplotlib axes; creates new figure if None.
        title: Axes title.

    Returns:
        The matplotlib axes with the rendered image.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    image = raymarch(
        sdf,
        camera_pos=camera_pos,
        look_at=look_at,
        light_dirs=light_dirs,
        light_colors=light_colors,
        resolution=resolution,
        fov=fov,
        max_steps=max_steps,
        max_dist=max_dist,
        shadow_steps=shadow_steps,
        shadow_hardness=shadow_hardness,
        gamma=gamma,
        ambient=ambient,
        aa_samples=aa_samples,
        background_color=background_color,
        refract_steps=refract_steps,
    )

    ax.imshow(image, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title or "Raymarched Render", fontsize=12)
    return ax
