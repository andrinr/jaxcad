"""Box primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Point
from jaxcad.sdf import SDF


class Box(SDF):
    """Axis-aligned box centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Array or Point constraint
    """

    def __init__(self, size: Union[Array, Point]):
        # Accept both raw values and constraints
        if isinstance(size, Point):
            self.size_param = size
        else:
            # Wrap raw value in a fixed Point constraint
            self.size_param = Point(value=jnp.asarray(size), free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for box using max distance to faces"""
        # Always use .value from the parameter for JAX traceability
        q = jnp.abs(p) - self.size_param.value
        return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) +
                jnp.minimum(jnp.max(q, axis=-1), 0.0))
