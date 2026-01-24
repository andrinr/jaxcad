"""Cone primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.primitives.base import Primitive


class Cone(Primitive):
    """Cone along Z-axis centered at origin.

    Args:
        radius: Base radius (float or Scalar)
        height: Cone height (apex at +height/2, base at -height/2) (float or Scalar)
    """

    def __init__(self, radius: Union[float, Scalar], height: Union[float, Scalar]):
        self.radius_param = radius if isinstance(radius, Scalar) else Scalar(value=radius, free=False)
        self.height_param = height if isinstance(height, Scalar) else Scalar(value=height, free=False)

    @staticmethod
    def sdf(p: Array, radius: float, height: float) -> Array:
        """Pure SDF function for cone.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Base radius
            height: Cone height

        Returns:
            Signed distance to cone surface
        """
        # Cone angle: sin and cos
        h = jnp.sqrt(radius * radius + height * height)
        sin_angle = radius / h
        cos_angle = height / h

        # Shift z so apex is at top, base at bottom
        z = p[..., 2] + height / 2.0

        # Distance in XY plane
        r = jnp.linalg.norm(p[..., :2], axis=-1)

        # Create 2D point (r, z)
        q = jnp.stack([r, z], axis=-1)

        # Cone direction vector (pointing from apex to base edge)
        c = jnp.array([sin_angle, -cos_angle])

        # Project q onto cone direction
        dot_qc = q[..., 0] * c[0] + q[..., 1] * c[1]

        # Clamped projection (only valid between apex and base)
        t = jnp.clip(dot_qc, 0.0, height)

        # Closest point on cone spine
        closest = jnp.stack([c[0] * t, c[1] * t], axis=-1)

        # Distance to cone surface
        d = jnp.linalg.norm(q - closest, axis=-1)

        # Inside/outside sign
        inside = (q[..., 0] * c[1] - q[..., 1] * c[0]) < 0

        # Cap the bottom
        d_base = jnp.where(z < 0, jnp.maximum(d, -z), d)

        return jnp.where(inside, -d_base, d_base)

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Cone.sdf(p, self.radius_param.value, self.height_param.value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Cone.sdf
