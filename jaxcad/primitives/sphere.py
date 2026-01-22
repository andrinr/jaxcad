"""Sphere primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance
from jaxcad.sdf import SDF


class Sphere(SDF):
    """Sphere centered at origin with given radius.

    Args:
        radius: Sphere radius (float or Distance constraint)
    """

    def __init__(self, radius: Union[float, Distance]):
        # Accept both raw values and constraints
        if isinstance(radius, Distance):
            self.radius_param = radius
            self.radius = radius.value
        else:
            # Wrap raw value in a fixed Distance constraint
            # Don't use float() - keep as JAX array for traceability
            self.radius_param = Distance(value=radius, free=False)
            self.radius = self.radius_param.value

    def __call__(self, p: Array) -> Array:
        """SDF: ||p|| - radius"""
        # Always use .value from the parameter for JAX traceability
        return jnp.linalg.norm(p, axis=-1) - self.radius_param.value
