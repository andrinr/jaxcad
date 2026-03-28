"""Compile a Scene to a differentiable (free_params, resolution) -> image function."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from jaxcad.render.scene import Scene

# ── helpers ───────────────────────────────────────────────────────────────────


def _collect(node_id: str, params_snapshot: dict, free: dict, fixed: dict) -> dict:
    """Resolve a node's params from free/fixed dicts into plain arrays."""
    result = {}
    for attr, param in params_snapshot.items():
        if param.free:
            result[attr] = free[param.name]
        else:
            path = f"{node_id}.{attr}"
            if path in fixed:
                result[attr] = fixed[path]
    return result


def _resolve(param, free_params: dict):
    """Return the current value of a Parameter (free → from dict, fixed → stored)."""
    return free_params[param.name] if param.free else param.value


# ── functionalize_scene ───────────────────────────────────────────────────────


def functionalize_scene(geometry) -> Callable:
    """Compile a geometry SDF tree to pure (sdf, material_fn) closures.

    Returns a curried function::

        scene_fn = functionalize_scene(geometry)
        sdf, material_fn = scene_fn(free_params, fixed_params)
        distance = sdf(point)
        mat_dict = material_fn(point)

    Both ``sdf`` and ``material_fn`` are fully differentiable w.r.t.
    ``free_params`` via ``jax.grad``.

    The node counter matches ``extract_parameters`` exactly (same DFS order,
    same counter increments for every Fluent node), so path-keyed fixed params
    align correctly.

    Args:
        geometry: Root SDF node of the geometry tree.

    Returns:
        ``(free_params, fixed_params) -> (sdf_fn, material_fn)``
    """
    from jaxcad.render.material import Material
    from jaxcad.sdf.boolean.base import BooleanOp
    from jaxcad.sdf.boolean.smooth import smooth_min
    from jaxcad.sdf.primitives.base import Primitive

    node_counter = {"count": 0}

    def build(obj):
        """DFS builder — returns (sdf_eval | None, mat_eval).

        For SDF nodes:   sdf_eval(p, free, fixed) -> distance
                         mat_eval(p, free, fixed) -> material dict
        For non-SDF nodes (Material): sdf_eval is None,
                         mat_eval(free, fixed) -> raw param dict  ← no p
        """
        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1
        ps = obj.params  # params snapshot

        # ── non-SDF Fluent (Material) ─────────────────────────────────────────
        if not hasattr(obj.__class__, "sdf"):
            for c in obj.children():
                build(c)

            def mat_params(free, fixed, _nid=node_id, _ps=ps):
                return _collect(_nid, _ps, free, fixed)

            return None, mat_params

        # ── SDF node ─────────────────────────────────────────────────────────
        pure_sdf = obj.__class__.sdf
        child_res = [build(c) for c in obj.children()]

        # SDF children: have sdf_eval (s is not None)
        sdf_ch = [(s, m) for s, m in child_res if s is not None]
        # Non-SDF children: material param collectors (s is None)
        mat_pfs = [m for s, m in child_res if s is None and m is not None]

        # ── Primitive ─────────────────────────────────────────────────────────
        if isinstance(obj, Primitive):
            mat_pf = mat_pfs[0] if mat_pfs else None

            def sdf_eval(p, free, fixed, _nid=node_id, _ps=ps, _fn=pure_sdf):
                return _fn(p, **_collect(_nid, _ps, free, fixed))

            def mat_eval(_p, free, fixed, _mpf=mat_pf):
                mp = _mpf(free, fixed) if _mpf is not None else {}
                return {
                    "color": mp.get("color", jnp.ones(3)),
                    "roughness": mp.get("roughness", jnp.array(0.5)),
                    "metallic": mp.get("metallic", jnp.array(0.0)),
                    "opacity": mp.get("opacity", jnp.array(1.0)),
                    "ior": mp.get("ior", jnp.array(1.0)),
                }

            return sdf_eval, mat_eval

        # ── BooleanOp (Union, Intersection, …) ───────────────────────────────
        if isinstance(obj, BooleanOp):

            def sdf_eval(p, free, fixed, _nid=node_id, _ps=ps, _fn=pure_sdf, _ch=sdf_ch):
                pv = _collect(_nid, _ps, free, fixed)
                evals = tuple(lambda p_, s=s: s(p_, free, fixed) for s, _ in _ch)
                return _fn(evals, p, **pv)

            def mat_eval(p, free, fixed, _nid=node_id, _ps=ps, _ch=sdf_ch):
                pv = _collect(_nid, _ps, free, fixed)
                smoothness = pv.get("smoothness", jnp.array(0.1))
                k = jnp.maximum(smoothness * 4.0, 1e-10)

                d0 = _ch[0][0](p, free, fixed)
                mat0 = _ch[0][1](p, free, fixed)
                for s_fn, m_fn in _ch[1:]:
                    d = s_fn(p, free, fixed)
                    m = m_fn(p, free, fixed)
                    t = jnp.clip(0.5 + 0.5 * (d - d0) / k, 0.0, 1.0)
                    mat0 = Material.blend(mat0, m, t)
                    d0 = smooth_min(d0, d, smoothness)
                return mat0

            return sdf_eval, mat_eval

        # ── Transform (Translate, Rotate, Scale, …) ──────────────────────────
        assert len(sdf_ch) == 1, f"Transform {obj.__class__.__name__} must have 1 SDF child"
        child_sdf_fn, child_mat_fn = sdf_ch[0]

        def sdf_eval(p, free, fixed, _nid=node_id, _ps=ps, _fn=pure_sdf, _csdf=child_sdf_fn):
            pv = _collect(_nid, _ps, free, fixed)
            return _fn(lambda p_: _csdf(p_, free, fixed), p, **pv)

        def mat_eval(p, free, fixed, _nid=node_id, _ps=ps, _cmat=child_mat_fn, _cls=obj.__class__):
            pv = _collect(_nid, _ps, free, fixed)
            tp = _cls._transform_point(p, **pv)
            return _cmat(tp, free, fixed)

        return sdf_eval, mat_eval

    inner_sdf, inner_mat = build(geometry)

    def scene_fn(free_params: dict, fixed_params: dict):
        return (
            lambda p: inner_sdf(p, free_params, fixed_params),
            lambda p: inner_mat(p, free_params, fixed_params),
        )

    return scene_fn


# ── functionalize_render ──────────────────────────────────────────────────────


def functionalize_render(
    scene: Scene,
    max_steps: int = 32,
    max_dist: float = 15.0,
    shadow_steps: int = 12,
    shadow_hardness: float = 6.0,
    gamma: float = 2.2,
) -> Callable:
    """Compile a ``Scene`` to a differentiable render function.

    Fixed geometry params are extracted once at call time and baked in, so the
    returned function only needs ``free_params`` — the dict that changes each
    optimisation step::

        render_fn = functionalize_render(scene)
        image     = render_fn(free_params, resolution=(64, 96))

    ``image`` is a JAX ``(H, W, 3)`` float32 array fully differentiable
    w.r.t. ``free_params`` via ``jax.grad``.

    Args:
        scene: ``Scene`` with geometry and ``RenderConfig``.
        max_steps: Sphere-tracing iterations per primary ray.
        max_dist: Miss threshold distance.
        shadow_steps: Soft shadow ray iterations.
        shadow_hardness: Shadow edge sharpness.
        gamma: Gamma correction exponent applied to the final image.

    Returns:
        ``(free_params, resolution=(H, W)) -> JAX image (H, W, 3)``
    """
    from jaxcad.extraction import extract_parameters
    from jaxcad.render.raymarch import _camera_rays, _render_pixel

    # Pre-extract geometry fixed params with the correct counter offset (0-based,
    # starting from geometry root) so paths match functionalize_scene's counter.
    _, geo_fixed, _ = extract_parameters(scene.geometry)
    scene_fn = functionalize_scene(scene.geometry)
    rc = scene.render_config

    def render_fn(free_params: dict, resolution: tuple[int, int] = (64, 96)):
        sdf, material_fn = scene_fn(free_params, geo_fixed)

        camera_pos = _resolve(rc.params["camera_pos"], free_params)
        look_at = _resolve(rc.params["look_at"], free_params)
        fov = _resolve(rc.params["fov"], free_params)
        bg = jax.nn.sigmoid(_resolve(rc.params["bg_color"], free_params))

        scene_dist = jnp.linalg.norm(camera_pos - look_at)
        h, w = resolution
        edge_width = 2.0 * fov / min(h, w) * scene_dist
        rays = _camera_rays(camera_pos, look_at, resolution, fov)

        def render_pixel(ray_dir):
            return _render_pixel(
                sdf,
                material_fn,
                camera_pos,
                ray_dir,
                rc.light_dirs,
                rc.light_colors,
                max_steps=max_steps,
                max_dist=max_dist,
                shadow_steps=shadow_steps,
                shadow_hardness=shadow_hardness,
                ambient=0.0,  # ambient handled via bg fallback
                edge_width=edge_width,
                background_color=bg,
                refract_steps=0,
            )

        image = jax.vmap(render_pixel)(rays).reshape(h, w, 3)
        return jnp.clip(jnp.maximum(image, 0.0) ** (1.0 / gamma), 0.0, 1.0)

    return render_fn
