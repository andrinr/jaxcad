"""Cone primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Cone(SDF):
    """Cone along Z-axis with apex at origin.

    Args:
        radius: Base radius (at z = -height)
        height: Cone height
    """

    def __init__(self, radius: float, height: float):
        self.radius = radius
        self.height = height
        # Precompute cone angle factors
        self.c = jnp.array([radius / height, -1.0])
        self.c = self.c / jnp.linalg.norm(self.c)

    def __call__(self, p: Array) -> Array:
        """SDF for cone"""
        # Distance in XY plane
        q = jnp.stack([jnp.linalg.norm(p[..., :2], axis=-1), p[..., 2]], axis=-1)

        # Project onto cone surface
        d = jnp.sum(q * self.c, axis=-1)

        # Distance to cone
        return jnp.maximum(d, -self.height - p[..., 2])
