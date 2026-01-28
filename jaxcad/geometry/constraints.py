"""Constraint system for geometric relationships (FUTURE).

This module is reserved for the future constraint system that will allow
expressing geometric relationships like:
- Distance constraints between points
- Angle constraints between lines
- Parallel/perpendicular relationships
- Tangency constraints
- etc.

The constraint system will automatically reduce degrees of freedom and
build a valid parameter space for optimization.

Example (future):
    from jaxcad.constraints import Point, Line, Distance, Parallel

    # Define parametric constraints
    A = Point([0, 0], free=True)  # 2 DOF
    B = Point([1, 0], free=True)  # 2 DOF
    L1 = Line([0, 1], [1, 1])     # Fixed, no DOF

    # Constraints automatically reduce DOF
    Distance(A, B, 1.0)           # Reduces total DOF by 1
    L2 = Line(start=A, end=B)
    Parallel(L1, L2)              # Reduces total DOF by 1
    # Total: 2 + 2 - 1 - 1 = 2 DOF

For now, this module re-exports parameter types from parameters.py for
backwards compatibility.
"""

# Re-export parameter types for backwards compatibility
from jaxcad.geometry.parameters import (
    Parameter,
    Scalar,
    Vector,
    Distance,
    Angle,
    Point,
)

__all__ = [
    'Parameter',
    'Scalar',
    'Vector',
    'Distance',
    'Angle',
    'Point',
]
