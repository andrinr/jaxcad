"""Tests for ConstraintGraph."""

import pytest
import jax
import jax.numpy as jnp

from jaxcad.geometry.parameters import Vector
from jaxcad.constraints import (
    ConstraintGraph,
    DistanceConstraint,
    PerpendicularConstraint,
)


def test_empty_graph():
    """Test empty constraint graph."""
    graph = ConstraintGraph()

    assert len(graph.constraints) == 0
    assert graph.get_total_dof_reduction() == 0


def test_add_single_constraint():
    """Test adding a single constraint to graph."""
    graph = ConstraintGraph()

    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    constraint = DistanceConstraint(p1, p2, 1.0)
    graph.add_constraint(constraint)

    assert len(graph.constraints) == 1
    assert graph.get_total_dof_reduction() == 1


def test_add_multiple_constraints():
    """Test multiple constraints in graph."""
    graph = ConstraintGraph()

    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')
    v1 = Vector([1, 0, 0], free=True, name='v1')
    v2 = Vector([0, 1, 0], free=True, name='v2')

    graph.add_constraint(DistanceConstraint(p1, p2, 1.0))
    graph.add_constraint(PerpendicularConstraint(v1, v2))

    assert len(graph.constraints) == 2
    # Distance (1) + Perpendicular (1) = 2 DOF reduction
    assert graph.get_total_dof_reduction() == 2


def test_extract_free_dof_no_constraints():
    """Test DOF extraction with no constraints."""
    graph = ConstraintGraph()

    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    reduced, null_space = graph.extract_free_dof([p1, p2])

    # With no constraints, reduced DOF should equal full DOF (6)
    assert reduced.shape[0] == 6  # 2 points × 3 coordinates
    assert null_space.shape == (6, 6)  # Identity-like mapping


def test_extract_free_dof_with_distance_constraint():
    """Test DOF extraction with distance constraint."""
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    graph = ConstraintGraph()
    graph.add_constraint(DistanceConstraint(p1, p2, 1.0))

    reduced, null_space = graph.extract_free_dof([p1, p2])

    # With 1 distance constraint: 6 DOF - 1 = 5 DOF
    assert reduced.shape[0] == 5
    assert null_space.shape == (6, 5)  # Maps 5D → 6D


def test_project_to_full():
    """Test projecting reduced params back to full space."""
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    graph = ConstraintGraph()
    graph.add_constraint(DistanceConstraint(p1, p2, 1.0))

    reduced, null_space = graph.extract_free_dof([p1, p2])

    # Project back
    base_point = jnp.concatenate([p1.xyz, p2.xyz])
    full = graph.project_to_full(reduced, null_space, base_point)

    assert full.shape == (6,)


def test_null_space_satisfies_constraints():
    """Test that movements in null space approximately preserve constraints."""
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([1, 0, 0], free=True, name='p2')

    graph = ConstraintGraph()
    constraint = DistanceConstraint(p1, p2, 1.0)
    graph.add_constraint(constraint)

    # Get null space
    reduced, null_space = graph.extract_free_dof([p1, p2])

    # Base point (current configuration)
    base_point = jnp.concatenate([p1.xyz, p2.xyz])

    # Move in null space
    delta_reduced = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    new_reduced = reduced + delta_reduced

    # Project to full space
    new_full = graph.project_to_full(new_reduced, null_space, base_point)

    # Check that constraint is still approximately satisfied
    new_p1 = new_full[:3]
    new_p2 = new_full[3:]

    param_values = {'p1': new_p1, 'p2': new_p2}
    residual = constraint.compute_residual(param_values)

    # Residual should be small (approximate due to linearization)
    assert jnp.abs(residual) < 0.1


@pytest.mark.parametrize("n_points,n_constraints,expected_reduced_dof", [
    (2, 1, 5),   # 6 DOF - 1 constraint
    (3, 2, 7),   # 9 DOF - 2 constraints
    (4, 3, 9),   # 12 DOF - 3 constraints
])
def test_dof_counting(n_points, n_constraints, expected_reduced_dof):
    """Test DOF counting with multiple constraints."""
    # Create n_points points
    points = [Vector([i, 0, 0], free=True, name=f'p{i}') for i in range(n_points)]

    graph = ConstraintGraph()

    # Add distance constraints between consecutive points
    for i in range(n_constraints):
        graph.add_constraint(DistanceConstraint(points[i], points[i+1], 1.0))

    assert graph.get_total_dof_reduction() == n_constraints

    reduced, null_space = graph.extract_free_dof(points)

    assert reduced.shape[0] == expected_reduced_dof
    assert null_space.shape == (n_points * 3, expected_reduced_dof)
