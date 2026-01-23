"""Round box primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance, Point
from jaxcad.sdf import SDF


class RoundBox(SDF):
    """Box with rounded edges centered at origin.

    Args:
        size: Half-extents in each dimension (x, y, z) - Array or Point constraint
        radius: Rounding radius (float or Distance constraint)
    """

    def __init__(self, size: Union[Array, Point], radius: Union[float, Distance]):
        # Accept both raw values and constraints
        if isinstance(size, Point):
            self.size_param = size
        else:
            self.size_param = Point(value=jnp.asarray(size), free=False)

        if isinstance(radius, Distance):
            self.radius_param = radius
        else:
            self.radius_param = Distance(value=radius, free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for rounded box"""
        q = jnp.abs(p) - self.size_param.value
        return (jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) +
                jnp.minimum(jnp.max(q, axis=-1), 0.0) - self.radius_param.value)
