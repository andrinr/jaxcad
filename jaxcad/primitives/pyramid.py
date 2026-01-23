"""Pyramid primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance
from jaxcad.sdf import SDF


class Pyramid(SDF):
    """Pyramid centered at origin with base on XZ plane.

    Args:
        height: Height of pyramid (float or Distance constraint)
    """

    def __init__(self, height: Union[float, Distance]):
        if isinstance(height, Distance):
            self.height_param = height
        else:
            self.height_param = Distance(value=height, free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for pyramid"""
        h = self.height_param.value
        m2 = h * h + 0.25  # (h^2 + 0.5^2)

        # Symmetry in xz plane
        p_xz = jnp.abs(jnp.stack([p[..., 0], p[..., 2]], axis=-1))
        p_xz = jnp.where(
            (p_xz[..., 1:2] > p_xz[..., 0:1]),
            jnp.flip(p_xz, axis=-1),
            p_xz
        )
        p_xz = p_xz - jnp.array([0.5, 0.0])

        # Project into face plane
        q = jnp.stack([
            p_xz[..., 0] - 0.5 * jnp.clip(p_xz[..., 0] / m2, -h, h),
            h * p_xz[..., 0] - 0.5 * jnp.clip(h * p_xz[..., 0] / m2, -h, h) - m2,
            p_xz[..., 1]
        ], axis=-1)

        s = jnp.maximum(-q[..., 0], 0.0)
        t = jnp.clip((q[..., 1] - 0.5 * p[..., 1]) / (m2 + 0.25), 0.0, 1.0)

        a = m2 * (q[..., 0] + s) ** 2 + q[..., 1] ** 2
        b = m2 * (q[..., 0] + 0.5 * t) ** 2 + (q[..., 1] - m2 * t) ** 2

        d2 = jnp.where(
            jnp.minimum(q[..., 1], -q[..., 0] * m2 - q[..., 1] * 0.5) > 0.0,
            0.0,
            jnp.minimum(a, b)
        )

        # Return signed distance
        return jnp.sqrt((d2 + q[..., 2] ** 2) / m2) * jnp.sign(
            jnp.maximum(q[..., 2], -p[..., 1])
        )
