"""Torus primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Torus(SDF):
    """Torus in XY plane centered at origin.

    Args:
        major_radius: Distance from origin to tube center
        minor_radius: Tube radius
    """

    def __init__(self, major_radius: float, minor_radius: float):
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def __call__(self, p: Array) -> Array:
        """SDF for torus"""
        q_xy = jnp.linalg.norm(p[..., :2], axis=-1) - self.major_radius
        q = jnp.stack([q_xy, p[..., 2]], axis=-1)
        return jnp.linalg.norm(q, axis=-1) - self.minor_radius
