"""Constraint solving via optimistix."""

from typing import Any

import optimistix as optx
from jax import Array

from jaxcad.constraints.dof import (
    _build_residual_fn,
    _build_x0,
    _collect_constraints,
    _fixed_params,
    total_dof_reduction,
)
from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.sdf import SDF


def solve_constraints(
    sdf: SDF,
    *,
    tol: float = 1e-6,
    max_steps: int = 256,
) -> dict[str, Any]:
    """Solve geometric constraints attached to a scene's free parameters.

    Extracts free parameters from the SDF tree, discovers all constraints
    registered on them, checks that the system is exactly constrained, then
    runs Levenberg-Marquardt (via optimistix) to find parameter values that
    satisfy all constraints.

    Args:
        sdf: The SDF tree whose free parameters should be solved.
        tol: Convergence tolerance (used as both rtol and atol).
        max_steps: Maximum solver iterations.

    Returns:
        A free_params dict (same format as extract_parameters) with solved
        parameter values — ready to pass to functionalize.

    Raises:
        ValueError: If the scene is under- or over-constrained.
        RuntimeError: If the solver does not converge.

    Example:
        ```python
        solved_params = solve_constraints(scene)
        sdf_fn = functionalize(scene)(solved_params, fixed_params)
        ```
    """
    # Local import avoids a circular dependency: constraints.solve → jaxcad → constraints
    from jaxcad.extraction import extract_parameters

    free_params_dict, _ = extract_parameters(sdf)

    # Deduplicate free params by name to get an ordered list
    seen: set = set()
    param_list = []
    for param in free_params_dict.values():
        if param.name not in seen:
            seen.add(param.name)
            param_list.append(param)

    # Discover all constraints from the free parameters
    constraints = _collect_constraints(param_list)

    # DOF check
    total_dof = sum(p.value.size for p in param_list)
    constraint_dof = total_dof_reduction(constraints)
    remaining = total_dof - constraint_dof
    if remaining > 0:
        raise ValueError(
            f"Under-constrained: {remaining} DOF remaining. "
            f"({total_dof} parameter DOF, {constraint_dof} constraint equations)"
        )
    if remaining < 0:
        raise ValueError(f"Over-constrained by {-remaining} equations.")

    fixed = _fixed_params(constraints)
    flat_fn = _build_residual_fn(constraints, param_list, fixed)
    residual_fn = lambda x, _: flat_fn(x)  # noqa: E731
    x0 = _build_x0(param_list)

    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    sol = optx.least_squares(residual_fn, solver, x0, max_steps=max_steps)
    x = sol.value

    # Reconstruct the solved free_params dict
    name_to_solved: dict[str, Array] = {}
    offset = 0
    for p in param_list:
        size = p.value.size
        name_to_solved[p.name] = x[offset : offset + size]
        if isinstance(p, Scalar):
            name_to_solved[p.name] = name_to_solved[p.name][0]
        offset += size

    return {
        path: (
            Vector(
                value=name_to_solved[param.name], free=True, name=param.name, bounds=param.bounds
            )
            if isinstance(param, Vector)
            else Scalar(
                value=name_to_solved[param.name], free=True, name=param.name, bounds=param.bounds
            )
        )
        for path, param in free_params_dict.items()
    }
