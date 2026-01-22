"""Base SDF (Signed Distance Function) class."""

from abc import ABC, abstractmethod

from jax import Array


class SDF(ABC):
    """Abstract base class for Signed Distance Functions.

    An SDF represents geometry implicitly as a function f(p) -> distance,
    where:
    - f(p) < 0: point p is inside the shape
    - f(p) = 0: point p is on the surface
    - f(p) > 0: point p is outside the shape

    All primitives and CSG operations inherit from this class.
    """

    @abstractmethod
    def __call__(self, p: Array) -> Array:
        """Evaluate the signed distance at point(s) p.

        Args:
            p: Point(s) to evaluate, shape (..., 3) for 3D or (..., 2) for 2D

        Returns:
            Signed distance value(s), shape (...)
        """
        pass

    def __or__(self, other: "SDF") -> "SDF":
        """Union operator: self | other"""
        from jaxcad.boolean import Union
        return Union(self, other)

    def __and__(self, other: "SDF") -> "SDF":
        """Intersection operator: self & other"""
        from jaxcad.boolean import Intersection
        return Intersection(self, other)

    def __sub__(self, other: "SDF") -> "SDF":
        """Difference operator: self - other"""
        from jaxcad.boolean import Difference
        return Difference(self, other)

    @classmethod
    def register_transform(cls, name: str, transform_class):
        """Register a transform as a fluent API method.

        Args:
            name: Method name to add to SDF class
            transform_class: Transform class to instantiate

        Example:
            SDF.register_transform('translate', Translate)
            # Now you can do: sphere.translate([1, 0, 0])
        """
        def method(self, *args, **kwargs):
            return transform_class(self, *args, **kwargs)

        method.__name__ = name
        method.__doc__ = transform_class.__doc__
        setattr(cls, name, method)
