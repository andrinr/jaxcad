"""Triangular prism primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Point
from jaxcad.sdf import SDF


class TriangularPrism(SDF):
    """Triangular prism centered at origin.

    Args:
        h: Half-extents (base width/2, height, depth/2) - Array or Point constraint
    """

    def __init__(self, h: Union[Array, Point]):
        if isinstance(h, Point):
            self.h_param = h
        else:
            self.h_param = Point(value=jnp.asarray(h), free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for triangular prism (bound)"""
        h = self.h_param.value

        # Compute distance components
        q_x = jnp.abs(p[..., 0])
        q_y = p[..., 1]
        q_z = jnp.abs(p[..., 2])

        # Distance to triangular face in XY plane
        d_tri = jnp.maximum(q_x * 0.866025 + q_y * 0.5, -q_y) - h[..., 1] * 0.5

        # Distance to rectangular faces
        d_x = q_x - h[..., 0]
        d_z = q_z - h[..., 2]

        # Combine distances
        return jnp.maximum(d_tri, jnp.maximum(d_x, d_z))
