"""Capped cone primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance
from jaxcad.sdf import SDF


class CappedCone(SDF):
    """Cone with flat caps along Y axis.

    Args:
        height: Height of cone (float or Distance constraint)
        r1: Radius at bottom (float or Distance constraint)
        r2: Radius at top (float or Distance constraint)
    """

    def __init__(
        self,
        height: Union[float, Distance],
        r1: Union[float, Distance],
        r2: Union[float, Distance],
    ):
        if isinstance(height, Distance):
            self.height_param = height
        else:
            self.height_param = Distance(value=height, free=False)

        if isinstance(r1, Distance):
            self.r1_param = r1
        else:
            self.r1_param = Distance(value=r1, free=False)

        if isinstance(r2, Distance):
            self.r2_param = r2
        else:
            self.r2_param = Distance(value=r2, free=False)

    def __call__(self, p: Array) -> Array:
        """SDF for capped cone"""
        h = self.height_param.value
        r1 = self.r1_param.value
        r2 = self.r2_param.value

        # Distance in XZ plane
        q_xz = jnp.sqrt(p[..., 0] ** 2 + p[..., 2] ** 2)

        # Map to 2D problem
        q = jnp.stack([q_xz, p[..., 1]], axis=-1)

        k1 = jnp.stack([r2, h], axis=-1)
        k2 = jnp.stack([r2 - r1, 2.0 * h], axis=-1)

        ca = jnp.stack([
            jnp.clip(q[..., 0] - jnp.minimum(q[..., 1], 0.0) * (r2 - r1) / h, 0.0, r1),
            jnp.abs(q[..., 1]) - h
        ], axis=-1)

        cb = q - k1 + k2 * jnp.clip(
            jnp.sum((k1 - q) * k2, axis=-1) / jnp.sum(k2 * k2, axis=-1),
            0.0,
            1.0
        )[..., None]

        s = jnp.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)

        return s * jnp.sqrt(jnp.minimum(
            jnp.sum(ca * ca, axis=-1),
            jnp.sum(cb * cb, axis=-1)
        ))
