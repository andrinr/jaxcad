"""Cylinder primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.primitives.base import Primitive


class Cylinder(Primitive):
    """Cylinder along Z-axis centered at origin.

    Args:
        radius: Cylinder radius (float or Scalar parameter)
        height: Cylinder half-height (total height = 2 * height)
    """

    def __init__(self, radius: Union[float, Scalar], height: Union[float, Scalar]):
        self.radius_param = radius if isinstance(radius, Scalar) else Scalar(value=radius, free=False)
        self.height_param = height if isinstance(height, Scalar) else Scalar(value=height, free=False)

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
        return Cylinder.sdf(p, self.radius_param.value, self.height_param.value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Cylinder.sdf
