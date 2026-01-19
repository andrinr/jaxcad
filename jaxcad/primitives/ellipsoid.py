"""Ellipsoid primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Ellipsoid(SDF):
    """Ellipsoid centered at origin.

    Args:
        radii: Semi-axes lengths in each dimension (rx, ry, rz)
    """

    def __init__(self, radii: Array):
        self.radii = jnp.asarray(radii)

    def __call__(self, p: Array) -> Array:
        """Approximate SDF for ellipsoid"""
        # Normalize by radii
        p_normalized = p / self.radii
        # Distance in normalized space
        d = jnp.linalg.norm(p_normalized, axis=-1) - 1.0
        # Scale back (approximation)
        k = jnp.min(self.radii)
        return d * k
