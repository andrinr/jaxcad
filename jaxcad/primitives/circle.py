"""Circle primitive (2D)."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Circle(SDF):
    """Circle in XY plane (infinite cylinder along Z).

    Args:
        radius: Circle radius
    """

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, p: Array) -> Array:
        """SDF: distance in XY plane - radius"""
        return jnp.linalg.norm(p[..., :2], axis=-1) - self.radius
