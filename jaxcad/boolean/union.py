"""Union boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.boolean.base import BooleanOp
from jaxcad.boolean.smooth import smooth_min
from jaxcad.parameters import Scalar
from jaxcad.sdf import SDF


class Union(BooleanOp):
    """Union of two SDFs (combines both shapes).

    Uses smooth minimum for differentiable blending at the intersection.

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
        """Pure function for union operation.

        Args:
            child_sdf1: First SDF function
            child_sdf2: Second SDF function
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Union SDF value
        """
        d1 = child_sdf1(p)
        d2 = child_sdf2(p)
        # Use jnp.where for JAX-compatible branching
        return jnp.where(smoothness > 0, smooth_min(d1, d2, smoothness), jnp.minimum(d1, d2))

    def __call__(self, p: Array) -> Array:
        """Union: min(d1, d2) with smooth blending"""
        return Union.sdf(self.sdf1, self.sdf2, p, self.params['smoothness'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Union.sdf
