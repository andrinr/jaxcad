"""Transformation operations for SDFs."""

from jaxcad.sdf import SDF
from jaxcad.transforms.affine import Translate, Rotate, Scale
from jaxcad.transforms.deformations import (
    Twist, Bend, Taper, RepeatInfinite, RepeatFinite, Mirror
)
from jaxcad.transforms.domain import (
    Symmetry, InfiniteRepetition, FiniteRepetition,
    Elongation, Rounding, Onion
)

__all__ = [
    "Translate", "Rotate", "Scale",
    "Twist", "Bend", "Taper",
    "RepeatInfinite", "RepeatFinite", "Mirror",
    "Symmetry", "InfiniteRepetition", "FiniteRepetition",
    "Elongation", "Rounding", "Onion"
]

# Register all transforms as fluent API methods on SDF class
SDF.register_transform('translate', Translate)
SDF.register_transform('rotate', Rotate)
SDF.register_transform('scale', Scale)
SDF.register_transform('twist', Twist)
SDF.register_transform('bend', Bend)
SDF.register_transform('taper', Taper)
SDF.register_transform('repeat_infinite', RepeatInfinite)
SDF.register_transform('repeat_finite', RepeatFinite)
SDF.register_transform('mirror', Mirror)
SDF.register_transform('symmetry', Symmetry)
SDF.register_transform('elongate', Elongation)
SDF.register_transform('round', Rounding)
SDF.register_transform('onion', Onion)
