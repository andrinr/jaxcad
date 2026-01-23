"""Rounded cylinder primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance
from jaxcad.sdf import SDF


class RoundedCylinder(SDF):
    """Cylinder with rounded caps along Y axis.

    Args:
        radius: Cylinder radius (float or Distance constraint)
        height: Half-height of cylinder (float or Distance constraint)
        rounding: Rounding radius for caps (float or Distance constraint)
    """

    def __init__(
        self,
        radius: Union[float, Distance],
        height: Union[float, Distance],
        rounding: Union[float, Distance],
    ):
        if isinstance(radius, Distance):
            self.radius_param = radius
        else:
            self.radius_param = Distance(value=radius, free=False)

        if isinstance(height, Distance):
            self.height_param = height
        else:
            self.height_param = Distance(value=height, free=False)

        if isinstance(rounding, Distance):
            self.rounding_param = rounding
        else:
            self.rounding_param = Distance(value=rounding, free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for rounded cylinder"""
        r = self.radius_param.value
        h = self.height_param.value
        ra = self.rounding_param.value

        # Distance to cylinder axis (XZ plane)
        d_radial = jnp.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2) - r + ra
        d_axial = jnp.abs(p[..., 1]) - h + ra

        return (jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0) +
                jnp.linalg.norm(jnp.stack([
                    jnp.maximum(d_radial, 0.0),
                    jnp.maximum(d_axial, 0.0)
                ], axis=-1), axis=-1) - ra)
