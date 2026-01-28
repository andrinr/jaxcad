"""XOR boolean operation."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf import SDF


class Xor(BooleanOp):
    """XOR of two SDFs (symmetric difference).

    Args:
        sdf1: First SDF
        sdf2: Second SDF
    """

    def __init__(self, sdf1: SDF, sdf2: SDF):
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    @staticmethod
    def sdf(child_sdf1, child_sdf2, p: Array) -> Array:
        """Pure function for XOR operation.

        Args:
            child_sdf1: First SDF function
            child_sdf2: Second SDF function
            p: Query point(s)

        Returns:
            XOR SDF value
        """
        d1 = child_sdf1(p)
        d2 = child_sdf2(p)
        return jnp.maximum(jnp.minimum(d1, d2), -jnp.maximum(d1, d2))

    def __call__(self, p: Array) -> Array:
        """XOR: max(min(d1, d2), -max(d1, d2))"""
        return Xor.sdf(self.sdf1, self.sdf2, p)

    def to_functional(self):
        """Return pure function for compilation."""
        return Xor.sdf
