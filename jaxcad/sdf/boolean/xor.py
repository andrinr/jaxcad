"""XOR boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf import SDF


class Xor(BooleanOp):
    """XOR of two SDFs (symmetric difference).

    Args:
        sdfs: 2-tuple of SDFs
    """

    def __init__(self, sdfs: tuple[SDF, SDF]):
        self.sdfs = sdfs
        self.params = {}

    @staticmethod
    def sdf(child_sdfs, p: Array) -> Array:
        """Pure function for XOR operation.

        Args:
            child_sdfs: 2-tuple of SDF functions
            p: Query point(s)

        Returns:
            XOR SDF value
        """
        d1 = child_sdfs[0](p)
        d2 = child_sdfs[1](p)
        return jnp.maximum(jnp.minimum(d1, d2), -jnp.maximum(d1, d2))

    def __call__(self, p: Array) -> Array:
        """XOR: max(min(d1, d2), -max(d1, d2))"""
        return Xor.sdf(self.sdfs, p)

    def to_functional(self):
        """Return pure function for compilation."""
        return Xor.sdf
