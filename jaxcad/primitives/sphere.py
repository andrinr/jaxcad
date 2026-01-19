"""Sphere primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Sphere(SDF):
    """Sphere centered at origin with given radius.

    Args:
        radius: Sphere radius
    """

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, p: Array) -> Array:
        """SDF: ||p|| - radius"""
        return jnp.linalg.norm(p, axis=-1) - self.radius
