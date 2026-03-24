"""Constraint solving: find unique parameter values satisfying all constraints."""

from typing import Dict, Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF
from jaxcad.compiler.extraction import extract_parameters
from jaxcad.geometry.parameters import Vector, Scalar


def solve_constraints(sdf: SDF, tol: float = 1e-6, max_iter: int = 100) -> Dict[str, Any]:
    """Solve constraints to find unique parameter values satisfying all geometric constraints.

    Errors loudly if the scene is under- or over-constrained. Otherwise, runs a
    Newton-Raphson solver to find the parameter values that satisfy all constraints
    exactly (to within `tol`).

    Args:
        sdf: The SDF tree whose free parameters should be solved.
        tol: Convergence tolerance for the residual norm.
        max_iter: Maximum Newton-Raphson iterations.

    Returns:
        A free_params dict (same format as extract_parameters) with solved Parameter
        values — ready to pass to functionalize as the free_params argument.

    Raises:
        ValueError: If the scene is under- or over-constrained.
        RuntimeError: If Newton-Raphson does not converge.

    Example:
        free_params, fixed_params = extract_parameters(scene)
        solved_params = solve_constraints(scene)
        sdf_fn = functionalize(scene)(solved_params, fixed_params)
        distance = sdf_fn(point)
    """
    # Step 1 — Extract parameters and discover constraints
    free_params_dict, fixed_params_dict = extract_parameters(sdf)

    # Deduplicate free params by name to get an ordered list
    param_set = set()
    param_list = []
    for param in free_params_dict.values():
        if param.name not in param_set:
            param_set.add(param.name)
            param_list.append(param)

    # Discover constraints from free params (use id() — constraints aren't hashable)
    constraint_ids = set()
    discovered_constraints = []
    for param in param_list:
        for constraint in param.get_constraints():
            if id(constraint) not in constraint_ids:
                constraint_ids.add(id(constraint))
                discovered_constraints.append(constraint)

    # Step 2 — DOF check
    total_dof = sum(p.value.size for p in param_list)
    constraint_dof = sum(c.dof_reduction() for c in discovered_constraints)
    remaining = total_dof - constraint_dof

    if remaining > 0:
        raise ValueError(
            f"Under-constrained: {remaining} DOF remaining. "
            f"({total_dof} parameter DOF, {constraint_dof} constraint equations)"
        )
    if remaining < 0:
        raise ValueError(f"Over-constrained by {-remaining} equations.")

    # Step 3 — Build residual function
    # Collect fixed params: from SDF tree + any referenced by constraints (e.g. anchors)
    fixed_by_name: Dict[str, Array] = {}
    for p in fixed_params_dict.values():
        if p.name is not None:
            fixed_by_name[p.name] = p.value
    free_names = {p.name for p in param_list}
    for constraint in discovered_constraints:
        for cp in constraint.get_parameters():
            if cp.name is not None and not cp.free and cp.name not in fixed_by_name:
                fixed_by_name[cp.name] = cp.value
            # Also handle free params referenced by constraint but not in param_list
            # (shouldn't happen in normal usage, but be safe)
            if cp.name is not None and cp.free and cp.name not in free_names:
                param_list.append(cp)
                free_names.add(cp.name)

    def residual_fn(x_flat: Array) -> Array:
        param_values = dict(fixed_by_name)
        offset = 0
        for p in param_list:
            size = p.value.size
            val = x_flat[offset:offset + size]
            param_values[p.name] = val[0] if isinstance(p, Scalar) else val
            offset += size
        residuals = [jnp.atleast_1d(c.compute_residual(param_values))
                     for c in discovered_constraints]
        return jnp.concatenate(residuals)

    # Step 4 — Newton-Raphson solver
    # Build initial flat vector from current parameter values
    x_parts = []
    for p in param_list:
        if isinstance(p, Vector):
            x_parts.append(p.xyz)
        else:
            x_parts.append(jnp.atleast_1d(p.value))
    x = jnp.concatenate(x_parts)

    for _ in range(max_iter):
        r = residual_fn(x)
        if jnp.linalg.norm(r) < tol:
            break
        J = jax.jacobian(residual_fn)(x)
        dx = jnp.linalg.lstsq(J, r, rcond=None)[0]
        x = x - dx
    else:
        final_norm = float(jnp.linalg.norm(residual_fn(x)))
        if final_norm >= tol:
            raise RuntimeError(
                f"solve_constraints did not converge after {max_iter} iterations. "
                f"Final residual: {final_norm:.2e}"
            )

    # Step 5 — Reconstruct free_params dict with solved values
    name_to_solved: Dict[str, Array] = {}
    offset = 0
    for p in param_list:
        size = p.value.size
        val = x[offset:offset + size]
        name_to_solved[p.name] = val[0] if isinstance(p, Scalar) else val
        offset += size

    solved_free_params: Dict[str, Any] = {}
    for path, param in free_params_dict.items():
        solved_val = name_to_solved[param.name]
        if isinstance(param, Vector):
            solved_free_params[path] = Vector(
                value=solved_val, free=True, name=param.name, bounds=param.bounds
            )
        else:
            solved_free_params[path] = Scalar(
                value=solved_val, free=True, name=param.name, bounds=param.bounds
            )

    return solved_free_params
