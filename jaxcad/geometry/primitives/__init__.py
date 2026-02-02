"""Geometric primitives for parametric construction.

This module contains 2D and 3D geometric entities that can be used to:
- Define parametric relationships with free/fixed parameters
- Apply geometric constraints
- Construct SDF primitives via the construction layer

Entities:
- Line: Parametric line segment in 3D
- Rectangle: Parametric rectangle in 3D
- Circle: Parametric circle in 3D
"""

from jaxcad.geometry.primitives.line import Line
from jaxcad.geometry.primitives.rectangle import Rectangle
from jaxcad.geometry.primitives.circle import Circle

__all__ = [
    'Line',
    'Rectangle',
    'Circle',
]
