"""Cylinder primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.sdf.primitives.base import Primitive


class Cylinder(Primitive):
    """Cylinder along Z-axis centered at origin.

    Args:
        radius: Cylinder radius (float or Scalar parameter)
        height: Cylinder half-height (total height = 2 * height)
    """

    def __init__(self, radius: Union[float, Scalar], height: Union[float, Scalar]):
        self.params = {'radius': radius, 'height': height}

    @staticmethod
    def sdf(p: Array, radius: float, height: float) -> Array:
        """Pure SDF function for cylinder.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Cylinder radius
            height: Cylinder half-height

        Returns:
            Signed distance to cylinder
        """
        d_xy = jnp.linalg.norm(p[..., :2], axis=-1) - radius
        d_z = jnp.abs(p[..., 2]) - height
        return jnp.sqrt(jnp.maximum(d_xy, 0.0)**2 + jnp.maximum(d_z, 0.0)**2) + \
               jnp.minimum(jnp.maximum(d_xy, d_z), 0.0)

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Cylinder.sdf(p, self.params['radius'].value, self.params['height'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Cylinder.sdf
