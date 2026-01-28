"""SDF (Signed Distance Function) module.

This module contains all SDF-related functionality:
- Base SDF class
- Primitives (Sphere, Box, Cylinder, etc.)
- Boolean operations (Union, Intersection, Difference)
- Transforms (Translate, Rotate, Scale, Twist)
"""

from jaxcad.sdf.base import SDF
from jaxcad.sdf.primitives import (
    Primitive,
    Box,
    Capsule,
    Cone,
    Cylinder,
    RoundBox,
    Sphere,
    Torus,
)
from jaxcad.sdf.boolean import (
    BooleanOp,
    Union,
    Intersection,
    Difference,
    Xor,
    smooth_min,
    smooth_max,
    union,
    intersection,
    difference,
    xor,
)
from jaxcad.sdf.transforms import (
    Translate,
    Rotate,
    Scale,
    Twist,
)

__all__ = [
    # Base
    "SDF",
    # Primitives
    "Primitive",
    "Box",
    "Capsule",
    "Cone",
    "Cylinder",
    "RoundBox",
    "Sphere",
    "Torus",
    # Boolean operations
    "BooleanOp",
    "Union",
    "Intersection",
    "Difference",
    "Xor",
    "smooth_min",
    "smooth_max",
    "union",
    "intersection",
    "difference",
    "xor",
    # Transforms
    "Translate",
    "Rotate",
    "Scale",
    "Twist",
]
