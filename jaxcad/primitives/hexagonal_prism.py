"""Hexagonal prism primitive."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class HexagonalPrism(SDF):
    """Hexagonal prism centered at origin, aligned along Y axis.

    Args:
        h: Half-extents (radius in XZ plane, half-height in Y) - 2D Array [radius, height]
    """

    def __init__(self, h: Union[Array, list, tuple]):
        # Store as simple array - 2D parameter [radius, height]
        self.h_param = jnp.asarray(h)
        if self.h_param.shape != (2,):
            raise ValueError(f"HexagonalPrism h must be 2D [radius, height], got shape {self.h_param.shape}")

    def __call__(self, p: Array) -> Array:
        """SDF for hexagonal prism"""
        # Constants for hexagon
        k = jnp.array([-0.8660254, 0.5, 0.57735])  # [-sqrt(3)/2, 1/2, 1/sqrt(3)]
        h = self.h_param

        p_abs = jnp.abs(p)

        # Project onto hexagonal cross-section
        p_xz = jnp.stack([p_abs[..., 0], p_abs[..., 2]], axis=-1)
        p_xz = p_xz - 2.0 * jnp.minimum(jnp.dot(k[:2], p_xz.T).T, 0.0)[:, None] * k[:2]

        d_xz = jnp.linalg.norm(
            p_xz - jnp.stack([
                jnp.clip(p_xz[..., 0], -k[2] * h[0], k[2] * h[0]),
                jnp.full_like(p_xz[..., 1], h[0])
            ], axis=-1),
            axis=-1
        )

        d_xz = d_xz * jnp.sign(p_xz[..., 1] - h[0])
        d_y = p_abs[..., 1] - h[1]

        # Combine distances
        return jnp.where(
            jnp.maximum(d_xz, d_y) < 0.0,
            jnp.maximum(d_xz, d_y),
            jnp.linalg.norm(jnp.stack([
                jnp.maximum(d_xz, 0.0),
                jnp.maximum(d_y, 0.0)
            ], axis=-1), axis=-1)
        )
