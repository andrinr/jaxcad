"""Cylinder primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Cylinder(SDF):
    """Cylinder along Z-axis centered at origin.

    Args:
        radius: Cylinder radius
        height: Cylinder half-height (total height = 2 * height)
    """

    def __init__(self, radius: float, height: float):
        self.radius = radius
        self.height = height

    def __call__(self, p: Array) -> Array:
        """SDF for infinite cylinder capped at height"""
        # Distance to infinite cylinder in XY plane
        d_xy = jnp.linalg.norm(p[..., :2], axis=-1) - self.radius
        # Distance to top/bottom caps
        d_z = jnp.abs(p[..., 2]) - self.height

        # Combine: inside if both negative, outside uses positive components
        return jnp.sqrt(jnp.maximum(d_xy, 0.0)**2 + jnp.maximum(d_z, 0.0)**2) + \
               jnp.minimum(jnp.maximum(d_xy, d_z), 0.0)
