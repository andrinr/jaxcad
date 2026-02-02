"""Construction layer: Bridge from geometry primitives to SDF primitives.

This module provides functions to construct SDF primitives from geometric entities.
It acts as the bridge between the parametric geometry layer and the SDF layer.

Construction functions preserve parameter references, so constraints on geometry
automatically affect the resulting SDFs.

Functions:
- extrude(rectangle, depth) → Box
- from_line(line, radius) → Capsule
- from_circle(circle, height) → Cylinder
- from_point(point, radius) → Sphere
"""

from __future__ import annotations

from jaxcad.construction.extrude import extrude
from jaxcad.construction.from_line import from_line
from jaxcad.construction.from_circle import from_circle
from jaxcad.construction.from_point import from_point

__all__ = [
    'extrude',
    'from_line',
    'from_circle',
    'from_point',
]
