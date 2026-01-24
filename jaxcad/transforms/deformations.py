"""Deformation transformations for SDFs."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.transforms.base import Transform


class Twist(Transform):
    """Twist deformation around Z-axis.

    Rotates points progressively based on their Z coordinate.
    Note: This produces an approximation, not an exact SDF.

    Args:
        sdf: The SDF to twist
        strength: Twist amount (radians per unit height)
    """

    def __init__(self, sdf: SDF, strength: Union[float, Scalar]):
        self.sdf = sdf
        # Accept both raw values and Scalar parameters
        if isinstance(strength, Scalar):
            self.strength_param = strength
        else:
            self.strength_param = Scalar(value=float(strength), free=False)

    @staticmethod
    def sdf(child_sdf, p: Array, strength: float) -> Array:
        """Pure function for twist deformation.

        Args:
            child_sdf: SDF function to twist
            p: Query point(s)
            strength: Twist strength in radians per unit height

        Returns:
            Twisted SDF value (approximate)
        """
        # Calculate rotation angle based on Z coordinate
        angle = strength * p[..., 2] if p.ndim > 1 else strength * p[2]
        c, s = jnp.cos(angle), jnp.sin(angle)

        # Apply rotation to XY coordinates
        if p.ndim == 1:
            x = p[0] * c - p[1] * s
            y = p[0] * s + p[1] * c
            z = p[2]
            p_twisted = jnp.array([x, y, z])
        else:
            x = p[..., 0] * c - p[..., 1] * s
            y = p[..., 0] * s + p[..., 1] * c
            z = p[..., 2]
            p_twisted = jnp.stack([x, y, z], axis=-1)

        return child_sdf(p_twisted)

    def __call__(self, p: Array) -> Array:
        """Evaluate twisted SDF."""
        return Twist.sdf(self.sdf, p, self.strength_param.value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Twist.sdf
