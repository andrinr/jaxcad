"""Capsule primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.primitives.base import Primitive


class Capsule(Primitive):
    """Capsule (cylinder with spherical caps) along Z-axis.

    Args:
        radius: Capsule radius (float or Scalar)
        height: Half-height of cylindrical section (float or Scalar)
    """

    def __init__(self, radius: Union[float, Scalar], height: Union[float, Scalar]):
        self.radius_param = radius if isinstance(radius, Scalar) else Scalar(value=radius, free=False)
        self.height_param = height if isinstance(height, Scalar) else Scalar(value=height, free=False)

    @staticmethod
    def sdf(p: Array, radius: float, height: float) -> Array:
        """Pure SDF function for capsule.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Capsule radius
            height: Half-height of cylindrical section

        Returns:
            Signed distance to capsule surface
        """
        # Clamp z coordinate to cylindrical section
        z_clamped = jnp.clip(p[..., 2], -height, height)
        # Distance to line segment along z-axis
        q = jnp.stack([p[..., 0], p[..., 1], p[..., 2] - z_clamped], axis=-1)
        return jnp.linalg.norm(q, axis=-1) - radius

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Capsule.sdf(p, self.radius_param.value, self.height_param.value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Capsule.sdf
