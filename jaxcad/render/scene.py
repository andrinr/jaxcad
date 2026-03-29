"""Scene and RenderConfig: top-level Fluent nodes for inverse rendering."""

from __future__ import annotations

import jax.numpy as jnp

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Scalar, Vector


class RenderConfig(Fluent):
    """Camera and lighting configuration stored as Fluent parameters.

    Camera and background params are ``Scalar`` / ``Vector`` objects and can be
    marked ``free=True`` to include them in gradient-based optimisation.

    Lights are fixed plain arrays by default.  To make them differentiable,
    pass ``light_dirs`` / ``light_colors`` as lists of ``Vector`` params::

        light_dirs = [
            Vector([0.45, 0.85, 0.25], free=True, name="light_dir_0"),
            Vector([-0.3, 0.5, -0.2],  free=True, name="light_dir_1"),
        ]
        light_colors = [
            Vector([0.95, 0.88, 0.75], free=True, name="light_color_0"),
            Vector([0.3,  0.38, 0.55], free=True, name="light_color_1"),
        ]

    Args:
        camera_pos: World-space camera position.
        look_at: Point the camera looks toward.
        fov: Half-width field-of-view (world units at unit depth).
        bg_color: Background / sky colour in logit space (sigmoid applied at
            render time so values are unconstrained).
        light_dirs: ``(N, 3)`` array of light directions (normalised internally),
            or a list of ``Vector`` params for differentiable lights.
        light_colors: ``(N, 3)`` array of light RGB colours,
            or a list of ``Vector`` params for differentiable lights.
    """

    def __init__(
        self,
        camera_pos: Vector,
        look_at: Vector,
        fov: Scalar,
        bg_color: Vector,
        light_dirs=None,
        light_colors=None,
    ):
        self.params = {
            "camera_pos": camera_pos,
            "look_at": look_at,
            "fov": fov,
            "bg_color": bg_color,
        }

        # Detect free Vector params vs fixed plain arrays
        if (
            isinstance(light_dirs, (list, tuple))
            and light_dirs
            and isinstance(light_dirs[0], Vector)
        ):
            n = len(light_dirs)
            if light_colors is None:
                light_colors = [Vector([1.0, 1.0, 1.0]) for _ in range(n)]
            for i, ld in enumerate(light_dirs):
                self.params[f"light_dir_{i}"] = ld
            for i, lc in enumerate(light_colors):
                self.params[f"light_color_{i}"] = lc
            self.n_lights = n
            self.free_lights = True
        else:
            if light_dirs is None:
                light_dirs = jnp.array([[0.45, 0.85, 0.25]])
            ld = jnp.atleast_2d(jnp.asarray(light_dirs, dtype=jnp.float32))
            if light_colors is None:
                light_colors = jnp.ones_like(ld)
            self.light_dirs = ld / jnp.linalg.norm(ld, axis=1, keepdims=True)
            self.light_colors = jnp.atleast_2d(jnp.asarray(light_colors, dtype=jnp.float32))
            self.n_lights = ld.shape[0]
            self.free_lights = False

    def children(self) -> list:
        return []


class Scene(Fluent):
    """Root Fluent node: geometry tree + render configuration.

    A single ``extract_parameters(scene)`` call yields every free parameter
    across geometry, materials, and render config.

    Args:
        geometry: SDF tree (``Union``, ``Translate``, primitives, …).
        render_config: ``RenderConfig`` with camera and lighting.
    """

    def __init__(self, geometry, render_config: RenderConfig):
        self.params = {}
        self.geometry = geometry
        self.render_config = render_config

    def children(self) -> list:
        return [self.geometry, self.render_config]
