"""Tests for constraints/solve.py — solve_constraints."""

import jax.numpy as jnp
import pytest

from jaxcad.constraints import DistanceConstraint
from jaxcad.constraints.solve import solve_constraints
from jaxcad.geometry.parameters import Vector
from jaxcad.sdf.primitives.sphere import Sphere
from jaxcad.sdf.transforms.affine.translate import Translate


def _trilateration_scene():
    """Trilateration: three anchors, one unknown point p."""
    anchor_a = Vector(jnp.array([0.0, 0.0, 0.0]), free=False, name="anchor_a")
    anchor_b = Vector(jnp.array([4.0, 0.0, 0.0]), free=False, name="anchor_b")
    anchor_c = Vector(jnp.array([2.0, 3.0, 0.0]), free=False, name="anchor_c")
    true_p = jnp.array([2.0, 1.0, 0.0])

    p = Vector(jnp.array([0.5, 0.5, 0.0]), free=True, name="p")
    scene = Translate(Sphere(radius=0.5), offset=p)

    DistanceConstraint(p, anchor_a, float(jnp.linalg.norm(true_p - anchor_a.value)))
    DistanceConstraint(p, anchor_b, float(jnp.linalg.norm(true_p - anchor_b.value)))
    DistanceConstraint(p, anchor_c, float(jnp.linalg.norm(true_p - anchor_c.value)))

    return scene, p, true_p


def test_solve_constraints_trilateration():
    """Solver recovers true_p from three distance constraints."""
    scene, p, true_p = _trilateration_scene()

    solved = solve_constraints(scene)

    assert jnp.allclose(solved["p"], true_p, atol=1e-5)


def test_solve_constraints_returns_free_params_dict():
    """Return value has same keys as extract_parameters free dict."""
    from jaxcad.extraction import extract_parameters

    scene, _, _ = _trilateration_scene()
    free_params, _, _ = extract_parameters(scene)
    solved = solve_constraints(scene)

    assert set(solved.keys()) == set(free_params.keys())


def test_solve_constraints_under_constrained():
    """Raises ValueError when DOF > constraints."""
    anchor = Vector(jnp.array([0.0, 0.0, 0.0]), free=False, name="anchor_uc")
    p = Vector(jnp.array([1.0, 0.0, 0.0]), free=True, name="p_uc")
    scene = Translate(Sphere(radius=0.5), offset=p)

    DistanceConstraint(p, anchor, 1.0)  # only 1 of 3 needed DOF removed

    with pytest.raises(ValueError, match="Under-constrained"):
        solve_constraints(scene)


def test_solve_constraints_over_constrained():
    """Raises ValueError when constraints > DOF."""
    Vector(jnp.array([0.0, 0.0, 0.0]), free=False, name="anchor_oc")
    p = Vector(jnp.array([1.0, 0.0, 0.0]), free=True, name="p_oc")
    scene = Translate(Sphere(radius=0.5), offset=p)

    # 4 constraints on a 3-DOF parameter
    for i, dist in enumerate([1.0, 1.0, 1.0, 1.0]):
        a = Vector(jnp.array([float(i), 0.0, 0.0]), free=False, name=f"anc_oc_{i}")
        DistanceConstraint(p, a, dist)

    with pytest.raises(ValueError, match="Over-constrained"):
        solve_constraints(scene)


def test_solve_constraints_scalar_param():
    """Solver works when the free parameter is a Scalar (1-DOF)."""
    # One free scalar radius, constrained to equal 2.0 via a distance from origin
    # to a point at [2,0,0]: ||[r,0,0] - [0,0,0]|| = 2 → r = 2
    # We encode this as: p is free Vector along x, one distance constraint fixes it.
    anchor = Vector(jnp.array([0.0, 0.0, 0.0]), free=False, name="anchor_sc")
    p = Vector(jnp.array([1.0, 0.0, 0.0]), free=True, name="p_sc")
    scene = Translate(Sphere(radius=0.5), offset=p)

    DistanceConstraint(p, anchor, float(jnp.linalg.norm(jnp.array([2.0, 0.0, 0.0]))))
    DistanceConstraint(
        p,
        Vector(jnp.array([0.0, 1.0, 0.0]), free=False, name="anc2_sc"),
        float(jnp.linalg.norm(jnp.array([2.0, -1.0, 0.0]))),
    )
    DistanceConstraint(
        p,
        Vector(jnp.array([0.0, 0.0, 1.0]), free=False, name="anc3_sc"),
        float(jnp.linalg.norm(jnp.array([2.0, 0.0, -1.0]))),
    )

    solved = solve_constraints(scene)
    assert jnp.allclose(solved["p_sc"], jnp.array([2.0, 0.0, 0.0]), atol=1e-5)
