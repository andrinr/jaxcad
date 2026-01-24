"""Box primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Vector
from jaxcad.primitives.base import Primitive


class Box(Primitive):
    """Axis-aligned box centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Array or Vector parameter

    Example:
        box = Box(size=[1, 2, 3])
        box = Box(size=Vector([1, 2, 3], free=True, name='size'))
    """

    def __init__(self, size: Union[Array, Vector]):
        if isinstance(size, Vector):
            self.size_param = size
        else:
            self.size_param = Vector(value=jnp.asarray(size), free=False)

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
        return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) +
                jnp.minimum(jnp.max(q, axis=-1), 0.0))

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Box.sdf(p, self.size_param.xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Box.sdf
