"""Box primitive."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives.base import Primitive


class Box(Primitive):
    """Axis-aligned box centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Vector parameter

    Example:
        box = Box(size=Vector([1, 2, 3], free=True, name='size'))
    """

    def __init__(self, size: Vector, material=None):
        from jaxcad.render.material import Material

        self.material = material if material is not None else Material()
        self.params = {"size": size}

    def material_at(self, _p):
        return self.material.as_dict()

    @staticmethod
    def sdf(p: Array, size: Array) -> Array:
        """Pure SDF function for box.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            size: Half-extents [x, y, z]

        Returns:
            Signed distance to box
        """
        q = jnp.abs(p) - size
        d = jnp.maximum(q, 0.0)
        return jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-20) + jnp.minimum(jnp.max(q, axis=-1), 0.0)

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Box.sdf(p, self.params["size"].xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Box.sdf
