"""Union boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf.boolean.smooth import smooth_min
from jaxcad.sdf import SDF


class Union(BooleanOp):
    """Union of two or more SDFs (combines all shapes).

    Uses smooth minimum for differentiable blending at the intersection.

    Args:
        sdfs: Tuple of SDFs to union
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, sdfs: tuple[SDF, ...], smoothness: float = 0.1):
        self.sdfs = sdfs
        self.params = {'smoothness': smoothness}

    @staticmethod
    def sdf(child_sdfs, p: Array, smoothness: float) -> Array:
        """Pure function for union operation.

        Args:
            child_sdfs: Tuple of SDF functions
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Union SDF value
        """
        result = child_sdfs[0](p)
        for child in child_sdfs[1:]:
            d = child(p)
            result = jnp.where(smoothness > 0, smooth_min(result, d, smoothness), jnp.minimum(result, d))
        return result

    def __call__(self, p: Array) -> Array:
        """Union: min over all children with smooth blending"""
        return Union.sdf(self.sdfs, p, self.params['smoothness'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Union.sdf
