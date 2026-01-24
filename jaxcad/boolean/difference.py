"""Difference boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.boolean.base import BooleanOp
from jaxcad.boolean.smooth import smooth_max
from jaxcad.sdf import SDF


class Difference(BooleanOp):
    """Difference of two SDFs (subtract second from first).

    Uses smooth maximum for differentiable blending.

    Args:
        sdf1: Base SDF
        sdf2: SDF to subtract
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, sdf1: SDF, sdf2: SDF, smoothness: float = 0.1):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.smoothness = smoothness

    @staticmethod
    def sdf(child_sdf1, child_sdf2, p: Array, smoothness: float) -> Array:
        """Pure function for difference operation.

        Args:
            child_sdf1: Base SDF function
            child_sdf2: SDF function to subtract
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Difference SDF value
        """
        d1 = child_sdf1(p)
        d2 = child_sdf2(p)
        if smoothness > 0:
            return smooth_max(d1, -d2, smoothness)
        return jnp.maximum(d1, -d2)

    def __call__(self, p: Array) -> Array:
        """Difference: max(d1, -d2) with smooth blending"""
        return Difference.sdf(self.sdf1, self.sdf2, p, self.smoothness)

    def to_functional(self):
        """Return pure function for compilation."""
        return Difference.sdf
