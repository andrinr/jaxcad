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

    def translate(self, offset: Array) -> "SDF":
        """Translate this SDF by offset vector."""
        from jaxcad.transforms import Translate
        return Translate(self, offset)

    def scale(self, scale: float | Array) -> "SDF":
        """Scale this SDF uniformly or non-uniformly."""
        from jaxcad.transforms import Scale
        return Scale(self, scale)

    def rotate(self, axis: str | Array, angle: float) -> "SDF":
        """Rotate this SDF around an axis."""
        from jaxcad.transforms import Rotate
        return Rotate(self, axis, angle)

    def twist(self, axis: str = 'z', strength: float = 1.0) -> "SDF":
        """Twist this SDF around an axis."""
        from jaxcad.transforms import Twist
        return Twist(self, axis, strength)

    def bend(self, axis: str = 'z', strength: float = 1.0) -> "SDF":
        """Bend this SDF along an axis."""
        from jaxcad.transforms import Bend
        return Bend(self, axis, strength)

    def taper(self, axis: str = 'z', strength: float = 0.5) -> "SDF":
        """Taper this SDF along an axis."""
        from jaxcad.transforms import Taper
        return Taper(self, axis, strength)

    def repeat_infinite(self, spacing: Array) -> "SDF":
        """Infinitely repeat this SDF in space."""
        from jaxcad.transforms import RepeatInfinite
        return RepeatInfinite(self, spacing)

    def repeat_finite(self, spacing: Array, count: Array) -> "SDF":
        """Repeat this SDF a finite number of times."""
        from jaxcad.transforms import RepeatFinite
        return RepeatFinite(self, spacing, count)

    def mirror(self, axis: str = 'x', offset: float = 0.0) -> "SDF":
        """Mirror this SDF across a plane."""
        from jaxcad.transforms import Mirror
        return Mirror(self, axis, offset)
