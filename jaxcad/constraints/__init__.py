"""Constraint system for geometric relationships.

This module provides a constraint system that allows expressing geometric
relationships and automatically reduces degrees of freedom (DOF) during optimization.

Architecture:
- Constraints define relationships between parameters (distance, angle, etc.)
- ConstraintGraph collects constraints and computes reduced DOF space
- Null space projection maps reduced parameters to full parameter space
- Integration with extract_parameters() for optimization

Example:
    from jaxcad.geometry import Vector, Scalar
    from jaxcad.constraints import DistanceConstraint, ConstraintGraph

    # Two free points (6 DOF total)
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    # Distance constraint (reduces DOF by 1)
    constraint = DistanceConstraint(p1, p2, Scalar(0.2))

    # Build constraint graph
    graph = ConstraintGraph()
    graph.add_constraint(constraint)

    # Extract reduced DOF (5 DOF instead of 6)
    reduced_params = graph.extract_free_dof([p1, p2])
"""

from __future__ import annotations

# Import all constraint types
from jaxcad.constraints.base import Constraint
from jaxcad.constraints.distance import DistanceConstraint
from jaxcad.constraints.angle import AngleConstraint
from jaxcad.constraints.parallel import ParallelConstraint
from jaxcad.constraints.perpendicular import PerpendicularConstraint
from jaxcad.constraints.graph import ConstraintGraph
from jaxcad.constraints.solve import solve_constraints, newton_raphson
from jaxcad.extraction import extract_parameters_with_constraints

# Convenience aliases (for backward compatibility with planned API)
Distance = DistanceConstraint
Angle = AngleConstraint
Parallel = ParallelConstraint
Perpendicular = PerpendicularConstraint

# Re-export parameter types for convenience
from jaxcad.geometry.parameters import Parameter, Scalar, Vector

# Type alias for Point (just a Vector)
Point = Vector

__all__ = [
    # Base class
    'Constraint',
    # Constraint types
    'DistanceConstraint',
    'AngleConstraint',
    'ParallelConstraint',
    'PerpendicularConstraint',
    # Graph
    'ConstraintGraph',
    # Solver
    'solve_constraints',
    'newton_raphson',
    'extract_parameters_with_constraints',
    # Aliases
    'Distance',
    'Angle',
    'Parallel',
    'Perpendicular',
    # Re-exports
    'Parameter',
    'Scalar',
    'Vector',
    'Point',
]
