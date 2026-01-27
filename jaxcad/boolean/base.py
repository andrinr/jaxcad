"""Base class for boolean operation SDFs."""

from __future__ import annotations

from jaxcad.sdf import SDF


class BooleanOp(SDF):
    """Base class for boolean operation SDFs (binary operations in the SDF tree).

    Boolean operations combine two SDFs - union, intersection, difference, etc.
    They have two children.

    Subclasses must implement:
    - @staticmethod def sdf(child_sdf1, child_sdf2, p: Array, smoothness: float) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable

    Subclasses should store:
    - self.sdf1: The first child SDF
    - self.sdf2: The second child SDF
    - self.params: Dictionary of Parameter objects
    """
