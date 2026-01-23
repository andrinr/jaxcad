"""Octahedron primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance
from jaxcad.sdf import SDF


class Octahedron(SDF):
    """Octahedron centered at origin.

    Args:
        size: Size parameter (float or Distance constraint)
    """

    def __init__(self, size: Union[float, Distance]):
        if isinstance(size, Distance):
            self.size_param = size
        else:
            self.size_param = Distance(value=size, free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for octahedron (exact)"""
        p_abs = jnp.abs(p)
        s = self.size_param.value
        m = jnp.sum(p_abs, axis=-1) - s

        # Calculate q for each component
        q_x = jnp.where(3.0 * p_abs[..., 0] < m, p_abs[..., 0], m * 0.57735027)
        q_y = jnp.where(3.0 * p_abs[..., 1] < m, p_abs[..., 1], m * 0.57735027)
        q_z = jnp.where(3.0 * p_abs[..., 2] < m, p_abs[..., 2], m * 0.57735027)

        q = jnp.stack([q_x, q_y, q_z], axis=-1)
        k = jnp.clip(0.5 * (q[..., 2] - q[..., 1] + s), 0.0, s)

        diff_x = q[..., 0]
        diff_y = q[..., 1] - s + k
        diff_z = q[..., 2] - k

        return jnp.linalg.norm(jnp.stack([diff_x, diff_y, diff_z], axis=-1), axis=-1)
