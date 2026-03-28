"""Rendering utilities for jaxcad SDFs.

Two backends:
- :func:`render_raymarched` / :func:`raymarch` — sphere tracing with soft
  shadows, AO proxy, Blinn-Phong shading and gamma correction.
- :func:`render_marching_cubes` — mesh extraction via marching cubes (requires
  scikit-image).
"""

from jaxcad.render.functionalize import functionalize_render, functionalize_scene
from jaxcad.render.marching_cubes import render_marching_cubes
from jaxcad.render.material import Material
from jaxcad.render.raymarch import raymarch, render_raymarched
from jaxcad.render.scene import RenderConfig, Scene

__all__ = [
    "raymarch",
    "render_raymarched",
    "render_marching_cubes",
    "Material",
    "RenderConfig",
    "Scene",
    "functionalize_scene",
    "functionalize_render",
]
