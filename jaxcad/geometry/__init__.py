"""Geometric entities and parameters for parametric CAD.

This module provides:
1. Parameter types (Vector, Scalar) for optimization
2. Geometric primitives (Line, Rectangle, Circle) for construction
3. Re-exports for convenience

The geometry layer is independent of SDFs and constraints.
"""

# Parameter types
from jaxcad.geometry.parameters import (
    Parameter,
    Scalar,
    Vector,
    as_parameter,
)

# Geometric primitives
from jaxcad.geometry.primitives import (
    Line,
    Rectangle,
    Circle,
)

__all__ = [
    # Parameters
    'Parameter',
    'Scalar',
    'Vector',
    'as_parameter',
    # Primitives
    'Line',
    'Rectangle',
    'Circle',
]
