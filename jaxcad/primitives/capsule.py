"""Capsule primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Capsule(SDF):
    """Capsule (cylinder with spherical caps) along Z-axis.

    Args:
        radius: Capsule radius
        height: Half-height of cylindrical section
    """

    def __init__(self, radius: float, height: float):
        self.radius = radius
        self.height = height

    def __call__(self, p: Array) -> Array:
        """SDF for capsule"""
        # Clamp z coordinate to cylindrical section
        z_clamped = jnp.clip(p[..., 2], -self.height, self.height)
        # Distance to line segment along z-axis
        q = jnp.stack([p[..., 0], p[..., 1], p[..., 2] - z_clamped], axis=-1)
        return jnp.linalg.norm(q, axis=-1) - self.radius
