"""Rectangle primitive (2D)."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Rectangle(SDF):
    """Rectangle in XY plane (infinite box along Z).

    Args:
        size: Half-extents in X and Y dimensions
    """

    def __init__(self, size: Array):
        self.size = jnp.asarray(size)

    def __call__(self, p: Array) -> Array:
        """SDF for rectangle in XY plane"""
        q = jnp.abs(p[..., :2]) - self.size
        return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) +
                jnp.minimum(jnp.max(q, axis=-1), 0.0))
