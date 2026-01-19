"""Box primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Box(SDF):
    """Axis-aligned box centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z)
    """

    def __init__(self, size: Array):
        self.size = jnp.asarray(size)

    def __call__(self, p: Array) -> Array:
        """SDF for box using max distance to faces"""
        q = jnp.abs(p) - self.size
        return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) +
                jnp.minimum(jnp.max(q, axis=-1), 0.0))
