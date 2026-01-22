"""Plane primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Plane(SDF):
    """Infinite plane with given normal and distance from origin.

    Args:
        normal: Plane normal vector (will be normalized)
        distance: Signed distance from origin to plane
    """

    def __init__(self, normal: Array, distance: float = 0.0):
        self.normal = jnp.asarray(normal)
        self.normal = self.normal / jnp.linalg.norm(self.normal)
        self.distance = distance

    def __call__(self, p: Array) -> Array:
        """SDF for plane: dot(p, n) - d"""
        return jnp.dot(p, self.normal) - self.distance
