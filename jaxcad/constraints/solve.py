"""Constraint solving via Newton-Raphson."""

from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF
from jaxcad.constraints.graph import ConstraintGraph
from jaxcad.geometry.parameters import Vector, Scalar


def newton_raphson(
    residual_fn: Callable[[Array], Array],
    x0: Array,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Array:
    """Find x such that residual_fn(x) ≈ 0 using Newton-Raphson.

    At each step computes the full Jacobian via JAX autodiff and takes a
    least-squares step, so it works for both square and overdetermined systems.

    Args:
        residual_fn: Maps a flat parameter vector to a residual vector.
        x0: Initial guess.
        tol: Convergence tolerance on the residual norm.
        max_iter: Maximum number of iterations.

    Returns:
        Solution vector x.

    Raises:
        RuntimeError: If the solver does not converge within max_iter steps.
    """
    x = x0
    for _ in range(max_iter):
        r = residual_fn(x)
        if jnp.linalg.norm(r) < tol:
            return x
        J = jax.jacobian(residual_fn)(x)
        dx = jnp.linalg.lstsq(J, r, rcond=None)[0]
        x = x - dx

    final_norm = float(jnp.linalg.norm(residual_fn(x)))
    if final_norm >= tol:
        raise RuntimeError(
            f"newton_raphson did not converge after {max_iter} iterations. "
            f"Final residual norm: {final_norm:.2e}"
        )
    return x


def solve_constraints(
    sdf: SDF,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Dict[str, Any]:
    """Solve geometric constraints attached to a scene's free parameters.

    Extracts free parameters from the SDF tree, discovers all constraints
    registered on them, checks that the system is exactly constrained, then
    runs Newton-Raphson to find parameter values that satisfy all constraints.

    Args:
        sdf: The SDF tree whose free parameters should be solved.
        tol: Convergence tolerance for the residual norm.
        max_iter: Maximum Newton-Raphson iterations.

    Returns:
        A free_params dict (same format as extract_parameters) with solved
        parameter values — ready to pass to functionalize.

    Raises:
        ValueError: If the scene is under- or over-constrained.
        RuntimeError: If Newton-Raphson does not converge.

    Example:
        solved_params = solve_constraints(scene)
        sdf_fn = functionalize(scene)(solved_params, fixed_params)
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
    graph = ConstraintGraph.from_parameters(param_list)

    # DOF check
    total_dof = sum(p.value.size for p in param_list)
    constraint_dof = graph.get_total_dof_reduction()
    remaining = total_dof - constraint_dof
    if remaining > 0:
        raise ValueError(
            f"Under-constrained: {remaining} DOF remaining. "
            f"({total_dof} parameter DOF, {constraint_dof} constraint equations)"
        )
    if remaining < 0:
        raise ValueError(f"Over-constrained by {-remaining} equations.")

    # Collect fixed params referenced by constraints (e.g. anchors not in the SDF tree)
    fixed_by_name: Dict[str, Array] = {}
    for constraint in graph.constraints:
        for cp in constraint.get_parameters():
            if cp.name is not None and not cp.free and cp.name not in fixed_by_name:
                fixed_by_name[cp.name] = cp.value

    # Build a flat residual function over all free params
    def residual_fn(x_flat: Array) -> Array:
        param_values = dict(fixed_by_name)
        offset = 0
        for p in param_list:
            size = p.value.size
            val = x_flat[offset:offset + size]
            param_values[p.name] = val[0] if isinstance(p, Scalar) else val
            offset += size
        return jnp.concatenate([
            jnp.atleast_1d(c.compute_residual(param_values))
            for c in graph.constraints
        ])

    x0 = jnp.concatenate([
        p.xyz if isinstance(p, Vector) else jnp.atleast_1d(p.value)
        for p in param_list
    ])

    x = newton_raphson(residual_fn, x0, tol=tol, max_iter=max_iter)

    # Reconstruct the solved free_params dict
    name_to_solved: Dict[str, Array] = {}
    offset = 0
    for p in param_list:
        size = p.value.size
        name_to_solved[p.name] = x[offset:offset + size]
        if isinstance(p, Scalar):
            name_to_solved[p.name] = name_to_solved[p.name][0]
        offset += size

    return {
        path: (
            Vector(value=name_to_solved[param.name], free=True, name=param.name, bounds=param.bounds)
            if isinstance(param, Vector) else
            Scalar(value=name_to_solved[param.name], free=True, name=param.name, bounds=param.bounds)
        )
        for path, param in free_params_dict.items()
    }
