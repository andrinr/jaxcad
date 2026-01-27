"""Base class for primitive SDFs."""

from __future__ import annotations

from jaxcad.sdf import SDF


class Primitive(SDF):
    """Base class for primitive SDFs (leaf nodes in the SDF tree).

    Primitives are the basic building blocks - spheres, boxes, cylinders, etc.
    They have no children.

    Subclasses must implement:
    - @staticmethod def sdf(p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable
    """
