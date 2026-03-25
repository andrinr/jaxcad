"""Constraint system for geometric relationships.

This module provides a constraint system that allows expressing geometric
relationships and automatically reduces degrees of freedom (DOF) during optimization.

Architecture:
- Constraints define relationships between parameters (distance, angle, etc.)
- Free functions in dof.py compute reduced DOF space via null-space projection
- Integration with extract_parameters() for optimization

Example:
    from jaxcad.geometry import Vector, Scalar
    from jaxcad.constraints import DistanceConstraint, extract_free_dof

    # Two free points (6 DOF total)
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    # Distance constraint (reduces DOF by 1)
    constraint = DistanceConstraint(p1, p2, Scalar(0.2))

    # Extract reduced DOF (5 DOF instead of 6)
    reduced_params, null_space = extract_free_dof([constraint], [p1, p2])
"""

from __future__ import annotations

# Import all constraint types
from jaxcad.constraints.base import Constraint
from jaxcad.constraints.dof import (
    all_parameters,
    extract_free_dof,
    in_null_space,
    linearize_at,
    null_space_update,
    project_gradient,
    project_to_full,
    project_to_manifold,
    project_to_reduced,
    projected_update,
    total_dof_reduction,
)
from jaxcad.constraints.solve import solve_constraints
from jaxcad.constraints.types.angle import AngleConstraint
from jaxcad.constraints.types.distance import DistanceConstraint
from jaxcad.constraints.types.parallel import ParallelConstraint
from jaxcad.constraints.types.perpendicular import PerpendicularConstraint
from jaxcad.extraction import extract_parameters_with_constraints

# Re-export parameter types for convenience
from jaxcad.geometry.parameters import Parameter, Scalar, Vector

# Convenience aliases (for backward compatibility with planned API)
Distance = DistanceConstraint
Angle = AngleConstraint
Parallel = ParallelConstraint
Perpendicular = PerpendicularConstraint

# Type alias for Point (just a Vector)
Point = Vector

__all__ = [
    # Base class
    "Constraint",
    # Constraint types
    "DistanceConstraint",
    "AngleConstraint",
    "ParallelConstraint",
    "PerpendicularConstraint",
    # DOF free functions
    "extract_free_dof",
    "linearize_at",
    "in_null_space",
    "null_space_update",
    "project_gradient",
    "project_to_full",
    "project_to_manifold",
    "project_to_reduced",
    "projected_update",
    "total_dof_reduction",
    "all_parameters",
    # Solver
    "solve_constraints",
    "extract_parameters_with_constraints",
    # Aliases
    "Distance",
    "Angle",
    "Parallel",
    "Perpendicular",
    # Re-exports
    "Parameter",
    "Scalar",
    "Vector",
    "Point",
]
