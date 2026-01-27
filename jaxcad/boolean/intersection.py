"""Intersection boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.boolean.base import BooleanOp
from jaxcad.boolean.smooth import smooth_max
from jaxcad.parameters import Scalar
from jaxcad.sdf import SDF


class Intersection(BooleanOp):
    """Intersection of two SDFs (only overlapping region).

    Uses smooth maximum for differentiable blending.

    Args:
        sdf1: First SDF
        sdf2: Second SDF
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, sdf1: SDF, sdf2: SDF, smoothness: float = 0.1):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.params = {'smoothness': smoothness}

    @staticmethod
    def sdf(child_sdf1, child_sdf2, p: Array, smoothness: float) -> Array:
        """Pure function for intersection operation.

        Args:
            child_sdf1: First SDF function
            child_sdf2: Second SDF function
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Intersection SDF value
        """
        d1 = child_sdf1(p)
        d2 = child_sdf2(p)
        # Use jnp.where for JAX-compatible branching
        return jnp.where(smoothness > 0, smooth_max(d1, d2, smoothness), jnp.maximum(d1, d2))

    def __call__(self, p: Array) -> Array:
        """Intersection: max(d1, d2) with smooth blending"""
        return Intersection.sdf(self.sdf1, self.sdf2, p, self.params['smoothness'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Intersection.sdf
