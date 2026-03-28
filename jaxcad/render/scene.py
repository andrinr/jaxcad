"""Scene and RenderConfig: top-level Fluent nodes for inverse rendering."""

from __future__ import annotations

import jax.numpy as jnp

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Scalar, Vector


class RenderConfig(Fluent):
    """Camera and lighting configuration stored as Fluent parameters.

    Camera and background params are ``Scalar`` / ``Vector`` objects and can be
    marked ``free=True`` to include them in gradient-based optimisation.
    Lights are plain arrays (not Parameters) — fixed by convention.

    Args:
        camera_pos: World-space camera position.
        look_at: Point the camera looks toward.
        fov: Half-width field-of-view (world units at unit depth).
        bg_color: Background / sky colour in logit space (sigmoid applied at
            render time so values are unconstrained).
        light_dirs: ``(N, 3)`` array of light directions (normalised internally).
            Defaults to a single warm key light.
        light_colors: ``(N, 3)`` array of light RGB colours.
            Defaults to white for every light.
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

        if light_dirs is None:
            light_dirs = jnp.array([[0.45, 0.85, 0.25]])
        ld = jnp.atleast_2d(jnp.asarray(light_dirs, dtype=jnp.float32))
        if light_colors is None:
            light_colors = jnp.ones_like(ld)
        self.light_dirs = ld / jnp.linalg.norm(ld, axis=1, keepdims=True)
        self.light_colors = jnp.atleast_2d(jnp.asarray(light_colors, dtype=jnp.float32))

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
