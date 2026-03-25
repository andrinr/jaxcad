"""Tests for DOF free functions (dof.py)."""

import jax.numpy as jnp
import pytest

from jaxcad.constraints import (
    DistanceConstraint,
    PerpendicularConstraint,
    all_parameters,
    extract_free_dof,
    linearize_at,
    project_to_full,
    project_to_reduced,
    total_dof_reduction,
)
from jaxcad.constraints.dof import (
    _build_residual_fn,
    _build_x0,
    _fixed_params,
)
from jaxcad.geometry.parameters import Vector


def test_empty_constraints():
    """Test with no constraints."""
    constraints = []
    assert total_dof_reduction(constraints) == 0


def test_single_constraint_dof_reduction():
    """Test DOF reduction from a single constraint."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    assert total_dof_reduction(constraints) == 1


def test_multiple_constraints_dof_reduction():
    """Test DOF reduction with multiple constraints."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    v1 = Vector([1, 0, 0], free=True, name="v1")
    v2 = Vector([0, 1, 0], free=True, name="v2")

    constraints = [DistanceConstraint(p1, p2, 1.0), PerpendicularConstraint(v1, v2)]
    # Distance (1) + Perpendicular (1) = 2 DOF reduction
    assert total_dof_reduction(constraints) == 2


def test_all_parameters():
    """Test that all_parameters returns deduplicated parameter list."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    params = all_parameters(constraints)

    assert len(params) == 2
    names = {p.name for p in params}
    assert names == {"p1", "p2"}


def test_extract_free_dof_no_constraints():
    """Test DOF extraction with no constraints."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    reduced, null_space = extract_free_dof([], [p1, p2])

    # With no constraints, reduced DOF should equal full DOF (6)
    assert reduced.shape[0] == 6  # 2 points × 3 coordinates
    assert null_space.shape == (6, 6)  # Identity-like mapping


def test_extract_free_dof_with_distance_constraint():
    """Test DOF extraction with distance constraint."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    reduced, null_space = extract_free_dof(constraints, [p1, p2])

    # With 1 distance constraint: 6 DOF - 1 = 5 DOF
    assert reduced.shape[0] == 5
    assert null_space.shape == (6, 5)  # Maps 5D → 6D


def test_project_to_full():
    """Test projecting reduced params back to full space."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    reduced, null_space = extract_free_dof(constraints, [p1, p2])

    base_point = jnp.concatenate([p1.xyz, p2.xyz])
    full = project_to_full(reduced, null_space, base_point)

    assert full.shape == (6,)


def test_project_round_trip():
    """Test that project_to_reduced inverts project_to_full."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    reduced, null_space = extract_free_dof(constraints, [p1, p2])
    base_point = jnp.concatenate([p1.xyz, p2.xyz])

    full = project_to_full(reduced, null_space, base_point)
    recovered = project_to_reduced(full, null_space, base_point)

    assert jnp.allclose(recovered, reduced, atol=1e-6)


def test_linearize_at_same_as_extract_at_initial_point():
    """linearize_at at the initial x0 should match extract_free_dof."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    param_list = [p1, p2]

    reduced0, ns0 = extract_free_dof(constraints, param_list)
    x0 = _build_x0(param_list)
    reduced1, ns1 = linearize_at(x0, constraints, param_list)

    assert jnp.allclose(ns0, ns1, atol=1e-6)
    assert jnp.allclose(reduced0, reduced1, atol=1e-6)


def test_linearize_at_refreshes_null_space():
    """After a move in reduced space, linearize_at should return a valid null space."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraints = [DistanceConstraint(p1, p2, 1.0)]
    param_list = [p1, p2]

    # Move to a different full-space point (a rigid translation preserves the constraint)
    new_full = jnp.array([0.5, 0.5, 0.0, 1.5, 0.5, 0.0])
    reduced, null_space = linearize_at(new_full, constraints, param_list)

    # Null space should still remove 1 DOF: shape (6, 5)
    assert null_space.shape == (6, 5)
    assert reduced.shape == (5,)


def test_weight_on_constraint_scales_residuals():
    """weight attribute on the constraint should scale its residuals."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([2, 0, 0], free=True, name="p2")

    c_unit = DistanceConstraint(p1, p2, 1.0)  # weight=1.0 (default)
    c_half = DistanceConstraint(p1, p2, 1.0, weight=0.5)

    param_list = [p1, p2]
    x0 = _build_x0(param_list)

    fn_unit = _build_residual_fn([c_unit], param_list, _fixed_params([c_unit]))
    fn_half = _build_residual_fn([c_half], param_list, _fixed_params([c_half]))

    assert jnp.allclose(fn_half(x0), 0.5 * fn_unit(x0), atol=1e-6)


def test_weight_default_is_one():
    """Default weight should be 1.0."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    c = DistanceConstraint(p1, p2, 1.0)
    assert c.weight == 1.0


def test_null_space_satisfies_constraints():
    """Test that movements in null space approximately preserve constraints."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    constraint = DistanceConstraint(p1, p2, 1.0)
    constraints = [constraint]
    reduced, null_space = extract_free_dof(constraints, [p1, p2])

    base_point = jnp.concatenate([p1.xyz, p2.xyz])

    delta_reduced = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    new_reduced = reduced + delta_reduced

    new_full = project_to_full(new_reduced, null_space, base_point)

    new_p1 = new_full[:3]
    new_p2 = new_full[3:]

    param_values = {"p1": new_p1, "p2": new_p2}
    residual = constraint.compute_residual(param_values)

    # Residual should be small (approximate due to linearization)
    assert jnp.abs(residual) < 0.1


@pytest.mark.parametrize(
    "n_points,n_constraints,expected_reduced_dof",
    [
        (2, 1, 5),  # 6 DOF - 1 constraint
        (3, 2, 7),  # 9 DOF - 2 constraints
        (4, 3, 9),  # 12 DOF - 3 constraints
    ],
)
def test_dof_counting(n_points, n_constraints, expected_reduced_dof):
    """Test DOF counting with multiple constraints."""
    points = [Vector([i, 0, 0], free=True, name=f"p{i}") for i in range(n_points)]

    constraints = [DistanceConstraint(points[i], points[i + 1], 1.0) for i in range(n_constraints)]

    assert total_dof_reduction(constraints) == n_constraints

    reduced, null_space = extract_free_dof(constraints, points)

    assert reduced.shape[0] == expected_reduced_dof
    assert null_space.shape == (n_points * 3, expected_reduced_dof)
