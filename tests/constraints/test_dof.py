"""Tests for DOF free functions (dof.py)."""

import jax.numpy as jnp
import pytest

from jaxcad.constraints import (
    DistanceConstraint,
    build_residual_fn,
    null_space,
)
from jaxcad.constraints.dof import compute_param_vector
from jaxcad.geometry.parameters import Vector


def _free_and_meta(*params):
    return (
        {p.name: p.value for p in params},
        {p.name: p for p in params},
    )


def test_null_space_no_constraints():
    """With no constraints the null space should be the full identity."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    N = null_space(*_free_and_meta(p1, p2))

    assert N.shape == (6, 6)


def test_null_space_with_distance_constraint():
    """One distance constraint should remove 1 DOF."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    DistanceConstraint(p1, p2, 1.0)

    N = null_space(*_free_and_meta(p1, p2))

    assert N.shape == (6, 5)


def test_weight_default_is_one():
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    c = DistanceConstraint(p1, p2, 1.0)
    assert c.weight == 1.0


def test_weight_on_constraint_scales_residuals():
    """weight should scale the constraint's residuals."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([2, 0, 0], free=True, name="p2")

    c_unit = DistanceConstraint(p1, p2, 1.0)
    c_half = DistanceConstraint(p1, p2, 1.0, weight=0.5)

    params = {"p1": p1, "p2": p2}
    x0 = compute_param_vector(params)

    fn_unit = build_residual_fn([c_unit], params)
    fn_half = build_residual_fn([c_half], params)

    assert jnp.allclose(fn_half(x0), 0.5 * fn_unit(x0), atol=1e-6)


def test_null_space_satisfies_constraints():
    """Movements in the null space should approximately preserve constraints."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    constraint = DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)
    base_vec = N.pack(free)

    new_full = base_vec + N.matrix @ jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    residual = constraint.compute_residual({"p1": new_full[:3], "p2": new_full[3:]})

    assert jnp.abs(residual) < 0.1


@pytest.mark.parametrize(
    "n_points,n_constraints,expected_dof",
    [
        (2, 1, 5),
        (3, 2, 7),
        (4, 3, 9),
    ],
)
def test_dof_counting(n_points, n_constraints, expected_dof):
    points = [Vector([i, 0, 0], free=True, name=f"p{i}") for i in range(n_points)]
    for i in range(n_constraints):
        DistanceConstraint(points[i], points[i + 1], 1.0)

    free = {p.name: p.value for p in points}
    meta = {p.name: p for p in points}
    N = null_space(free, meta)

    assert N.shape == (n_points * 3, expected_dof)
