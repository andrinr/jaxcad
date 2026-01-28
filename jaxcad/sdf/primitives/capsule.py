"""Capsule primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.sdf.primitives.base import Primitive


class Capsule(Primitive):
    """Capsule (cylinder with spherical caps) along Z-axis.

    Args:
        radius: Capsule radius (float or Scalar)
        height: Half-height of cylindrical section (float or Scalar)
    """

    def __init__(self, radius: Union[float, Scalar], height: Union[float, Scalar]):
        self.params = {'radius': radius, 'height': height}

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
        return Capsule.sdf(p, self.params['radius'].value, self.params['height'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Capsule.sdf
