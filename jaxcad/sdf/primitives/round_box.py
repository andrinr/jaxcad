"""Round box primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar, Vector
from jaxcad.sdf.primitives.base import Primitive


class RoundBox(Primitive):
    """Box with rounded edges centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Array or Vector parameter
        radius: Rounding radius (float or Scalar parameter)
    """

    def __init__(self, size: Union[Array, Vector], radius: Union[float, Scalar]):
        self.params = {'size': size, 'radius': radius}

    @staticmethod
    def sdf(p: Array, size: Array, radius: float) -> Array:
        """Pure SDF function for rounded box.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            size: Half-extents [x, y, z]
            radius: Rounding radius

        Returns:
            Signed distance to rounded box
        """
        q = jnp.abs(p) - size
        return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) +
                jnp.minimum(jnp.max(q, axis=-1), 0.0) - radius)

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return RoundBox.sdf(p, self.params['size'].xyz, self.params['radius'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return RoundBox.sdf
