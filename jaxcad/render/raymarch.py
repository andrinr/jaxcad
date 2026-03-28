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
    # mgrid produces rows top-to-bottom (y: +fy → -fy) so image is not upside-down
    y, x = jnp.mgrid[fy : -fy : h * 1j, -fx : fx : w * 1j].reshape(2, -1)
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

    def f(_, carry):
        t, shadow = carry
        h = sdf(origin + light_dir * t)
        # Clamp step to positive to prevent the ray going backwards inside geometry
        return t + jnp.maximum(h, 1e-5), jnp.clip(hardness * h / t, 0.0, shadow)

    _, shadow = jax.lax.fori_loop(0, steps, f, (jnp.array(1e-2), jnp.array(1.0)))
    return shadow


def _render_pixel(
    sdf: Callable[[Array], Array],
    ray_origin: Array,
    ray_dir: Array,
    light_dir: Array,
    max_steps: int,
    max_dist: float,
    eps: float,
    shadow_steps: int,
    shadow_hardness: float,
) -> Array:
    """Trace one ray and return its shaded intensity.

    Pipeline:
      1. Sphere trace to find surface hit.
      2. Compute surface normal via autodiff gradient.
      3. Use gradient norm as an ambient occlusion proxy (concave areas
         have norm < 1, receiving less ambient light).
      4. Cast soft shadow ray toward light (offset along normal to avoid
         self-intersection).
      5. Blend ambient + diffuse + specular (Blinn-Phong).
    """

    # Sphere tracing — clamp step to positive so rays never go backwards
    # when they overshoot into geometry with a negative SDF value
    def march(_, t):
        return t + jnp.maximum(sdf(ray_origin + t * ray_dir), 1e-5)

    t = jax.lax.fori_loop(0, max_steps, march, jnp.array(0.0))

    pos = ray_origin + t * ray_dir
    hit = (sdf(pos) < eps) & (t < max_dist)

    # Normal and AO proxy from gradient
    raw_normal = jax.grad(sdf)(pos)
    ao = jnp.linalg.norm(raw_normal)
    normal = raw_normal / jnp.where(ao > 1e-6, ao, 1.0)

    # Soft shadow — pass normal so the shadow origin is offset off the surface
    shadow = _cast_shadow(sdf, pos, normal, light_dir, shadow_steps, shadow_hardness)

    # Blinn-Phong shading
    diffuse = jnp.clip(jnp.dot(normal, light_dir), 0.0, 1.0) * shadow
    halfway = _normalize(light_dir - ray_dir)
    specular = jnp.clip(jnp.dot(halfway, normal), 0.0, 1.0) ** 200.0 * shadow

    color = 0.2 * ao + 0.7 * diffuse + 0.3 * specular
    return jnp.where(hit, color, 0.0)


def raymarch(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dir: Array = jnp.array([0.5, 1.0, 0.3]),
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 64,
    max_dist: float = 20.0,
    eps: float = 1e-3,
    shadow_steps: int = 32,
    shadow_hardness: float = 8.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Render an SDF via sphere tracing and return a float32 image array.

    Uses ``jax.vmap`` over pixels and ``jax.lax.fori_loop`` for the inner
    march loop (avoids JIT unrolling, ~10× faster compilation than a Python
    for-loop).

    Args:
        sdf: Signed distance function, callable ``(point: Array[3]) → Array[]``.
        camera_pos: Camera position in world space.
        look_at: Point the camera looks toward.
        light_dir: Direction *toward* the light (will be normalized).
        resolution: Output image size as (height, width).
        fov: Half-width field-of-view parameter.
        max_steps: Sphere tracing iterations per ray.
        max_dist: Rays beyond this distance are treated as misses.
        eps: Surface threshold — ``sdf(p) < eps`` counts as a hit.
        shadow_steps: Iterations for the soft shadow ray.
        shadow_hardness: Shadow sharpness (higher = harder edges).
        gamma: Gamma correction exponent (2.2 = sRGB standard).

    Returns:
        Float32 numpy array of shape (H, W) with values in [0, 1].
    """
    h, w = resolution
    light_dir = _normalize(light_dir)
    rays = _camera_rays(camera_pos, look_at, resolution, fov)  # (H*W, 3)

    render_fn = jax.jit(
        jax.vmap(
            lambda ray_dir: _render_pixel(
                sdf,
                camera_pos,
                ray_dir,
                light_dir,
                max_steps,
                max_dist,
                eps,
                shadow_steps,
                shadow_hardness,
            )
        )
    )

    image = render_fn(rays).reshape(h, w)
    image = jnp.clip(image ** (1.0 / gamma), 0.0, 1.0)
    return np.array(image)


def render_raymarched(
    sdf: Callable[[Array], Array],
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    light_dir: Array = jnp.array([0.5, 1.0, 0.3]),
    resolution: tuple[int, int] = (200, 200),
    fov: float = 0.6,
    max_steps: int = 64,
    max_dist: float = 20.0,
    eps: float = 1e-3,
    shadow_steps: int = 32,
    shadow_hardness: float = 8.0,
    gamma: float = 2.2,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Render an SDF via sphere tracing and display with matplotlib.

    Wraps :func:`raymarch` and shows the result on a ``plt.Axes``.

    Args:
        sdf: Signed distance function.
        camera_pos: Camera position in world space.
        look_at: Point the camera looks toward.
        light_dir: Direction toward the light (will be normalized).
        resolution: Output image size as (height, width).
        fov: Half-width field-of-view parameter.
        max_steps: Sphere tracing iterations per ray.
        max_dist: Miss threshold distance.
        eps: Surface hit threshold.
        shadow_steps: Soft shadow ray iterations.
        shadow_hardness: Shadow edge sharpness.
        gamma: Gamma correction exponent.
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
        light_dir=light_dir,
        resolution=resolution,
        fov=fov,
        max_steps=max_steps,
        max_dist=max_dist,
        eps=eps,
        shadow_steps=shadow_steps,
        shadow_hardness=shadow_hardness,
        gamma=gamma,
    )

    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title or "Raymarched Render", fontsize=12)
    return ax
