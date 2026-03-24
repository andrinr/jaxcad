"""Difference boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf.boolean.smooth import smooth_max
from jaxcad.sdf import SDF


class Difference(BooleanOp):
    """Difference of SDFs (subtract all subsequent from first).

    Uses smooth maximum for differentiable blending.

    Args:
        sdfs: Tuple of SDFs; first is the base, rest are subtracted
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, *sdfs, smoothness: float = 0.1):
        if len(sdfs) == 1 and isinstance(sdfs[0], (tuple, list)):
            sdfs = tuple(sdfs[0])
        self.sdfs = sdfs
        self.params = {'smoothness': smoothness}

    @staticmethod
    def sdf(child_sdfs, p: Array, smoothness: float) -> Array:
        """Pure function for difference operation.

        Args:
            child_sdfs: Tuple of SDF functions; first is base, rest are subtracted
            p: Query point(s)
            smoothness: Blend radius

        Returns:
            Difference SDF value
        """
        result = child_sdfs[0](p)
        for child in child_sdfs[1:]:
            d = child(p)
            result = jnp.where(smoothness > 0, smooth_max(result, -d, smoothness), jnp.maximum(result, -d))
        return result

    def __call__(self, p: Array) -> Array:
        """Difference: subtract all subsequent SDFs from first"""
        return Difference.sdf(self.sdfs, p, self.params['smoothness'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Difference.sdf
