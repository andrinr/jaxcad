"""Free functions for constraint DOF analysis and null-space projection."""

from __future__ import annotations

import warnings
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.constraints.base import Constraint
from jaxcad.geometry.parameters import Parameter, Scalar

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _collect_constraints(param_list: list[Parameter]) -> list[Constraint]:
    """Discover all constraints attached to the given parameters (deduplicated)."""
    seen: set = set()
    constraints = []
    for param in param_list:
        for constraint in param.get_constraints():
            if id(constraint) not in seen:
                seen.add(id(constraint))
                constraints.append(constraint)
    return constraints


def _fixed_params(constraints: list[Constraint]) -> dict[str, Array]:
    """Collect values of non-free parameters referenced by constraints."""
    params = all_parameters(constraints)
    return {
        param.name: param.value for param in params if not param.free and param.name is not None
    }


def _build_x0(param_list: list[Parameter]) -> Array:
    """Build the initial flat parameter vector from an ordered list of free params."""
    return jnp.concatenate([jnp.atleast_1d(p.value) for p in param_list])


def _build_residual_fn(
    constraints: list[Constraint],
    param_list: list[Parameter],
    fixed_by_name: dict[str, Array],
) -> Callable[[Array], Array]:
    """Build a flat residual function over all constraints.

    Each constraint's residuals are scaled by its ``weight`` attribute before
    concatenation. Set ``constraint.weight`` at construction time to normalize
    across types or scales.
    """
    # Capture weights as a static Python tuple so they're never JAX-traced.
    weights: tuple[float, ...] = tuple(c.weight for c in constraints)

    def flat_fn(x_flat: Array) -> Array:
        param_values = dict(fixed_by_name)
        offset = 0
        for p in param_list:
            size = p.value.size
            val = x_flat[offset : offset + size]
            param_values[p.name] = val[0] if isinstance(p, Scalar) else val
            offset += size
        return jnp.concatenate(
            [
                w * jnp.atleast_1d(c.compute_residual(param_values))
                for w, c in zip(weights, constraints)
            ]
        )

    return flat_fn


def _compute_null_space(jacobian: Array, n_params: int, tolerance: float = 1e-10) -> Array:
    """Compute the null space of the constraint Jacobian via SVD."""
    if jacobian.size == 0 or jacobian.shape[0] == 0:
        return jnp.eye(n_params)

    _, s, Vt = jnp.linalg.svd(jacobian, full_matrices=True)
    rank = jnp.sum(s > tolerance)
    V = Vt.T
    return V[:, rank:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def all_parameters(constraints: list[Constraint]) -> list[Parameter]:
    """Return the unique list of all parameters involved in the given constraints."""
    params = []
    seen: set = set()
    for constraint in constraints:
        for param in constraint.get_parameters():
            if param.name not in seen:
                params.append(param)
                seen.add(param.name)
    return params


def total_dof_reduction(constraints: list[Constraint]) -> int:
    """Compute the total DOF reduction across all constraints."""
    return sum(c.dof_reduction() for c in constraints)


def extract_free_dof(
    constraints: list[Constraint], param_list: list[Parameter]
) -> tuple[Array, Array]:
    """Extract reduced degrees of freedom from parameters under constraints.

    Linearizes the constraint manifold at the current parameter values
    (from param_list) and returns coordinates in the resulting null space.

    Args:
        constraints: List of constraints to apply.
        param_list: Ordered list of free Parameter objects.

    Returns:
        reduced_params: Reduced DOF vector (size = original_dof - constraint_dof).
        null_space: Null-space matrix for projecting reduced → full parameters.
    """
    free_params = []
    for param in param_list:
        if not param.free:
            warnings.warn(
                f"Parameter {param.name} is not free, skipping in DOF extraction",
                stacklevel=2,
            )
            continue
        free_params.append(param)

    x0 = _build_x0(free_params)
    n_params = x0.shape[0]

    if not constraints:
        null_space = jnp.eye(n_params)
        return null_space.T @ x0, null_space

    fixed_by_name = _fixed_params(constraints)
    flat_fn = _build_residual_fn(constraints, free_params, fixed_by_name)
    jacobian = jax.jacobian(flat_fn)(x0)
    null_space = _compute_null_space(jacobian, n_params)
    return null_space.T @ x0, null_space


def linearize_at(
    full: Array,
    constraints: list[Constraint],
    param_list: list[Parameter],
) -> tuple[Array, Array]:
    """Recompute (reduced, null_space) by linearizing constraints at an arbitrary point.

    Use this to refresh the linearization at each optimization step when
    constraints are nonlinear (approach 1b). The cost is one Jacobian evaluation
    + SVD per call.

    The null space is recomputed at ``full`` rather than at the initial parameter
    values, so it stays accurate as the optimizer moves away from the starting point.

    Args:
        full: Current point in full parameter space (flat array, layout defined by
            param_list).
        constraints: List of constraints.
        param_list: Ordered list of free Parameter objects (defines flat array layout).

    Returns:
        reduced: Reduced coordinates of ``full`` under the new linearization.
        null_space: Updated null-space matrix (n_params × n_free_dof).

    Example:
        ```python
        # Outer optimization loop (approach 1b)
        full = _build_x0(param_list)
        for _ in range(n_steps):
            reduced, N = linearize_at(full, constraints, param_list)
            grad = jax.grad(lambda r: loss(base + N @ r))(reduced)
            reduced = reduced - lr * grad
            full = project_to_full(reduced, N)
        ```
    """
    n_params = full.shape[0]

    if not constraints:
        null_space = jnp.eye(n_params)
        return null_space.T @ full, null_space

    fixed_by_name = _fixed_params(constraints)
    flat_fn = _build_residual_fn(constraints, param_list, fixed_by_name)
    jacobian = jax.jacobian(flat_fn)(full)
    null_space = _compute_null_space(jacobian, n_params)
    return null_space.T @ full, null_space


def project_to_full(reduced: Array, null_space: Array, base_point: Array | None = None) -> Array:
    """Project reduced DOF parameters back to full parameter space.

    Computes: full = base_point + null_space @ reduced

    Args:
        reduced: Reduced DOF vector.
        null_space: Null-space matrix from extract_free_dof or linearize_at.
        base_point: Base point in full space (default: origin).

    Returns:
        Full parameter vector.
    """
    if base_point is None:
        base_point = jnp.zeros(null_space.shape[0])
    return base_point + null_space @ reduced


def project_to_reduced(full: Array, null_space: Array, base_point: Array | None = None) -> Array:
    """Project a full parameter vector back to reduced coordinates.

    Inverse of project_to_full. Since null_space has orthonormal columns
    (from SVD), its pseudo-inverse is null_space.T:

        reduced = null_space.T @ (full - base_point)

    Args:
        full: Full parameter vector (size = n_params).
        null_space: Null-space matrix from extract_free_dof or linearize_at.
        base_point: Base point in full space (default: origin).

    Returns:
        Reduced DOF vector.
    """
    if base_point is None:
        base_point = jnp.zeros(null_space.shape[0])
    return null_space.T @ (full - base_point)


def project_gradient(g: Array, null_space: Array) -> Array:
    """Project a gradient vector onto the constraint null space (tangent plane).

    Computes N Nᵀ g — the component of g that lies in the feasible
    directions. The orthogonal complement (Nᵀ)⊥ g is discarded; it points
    toward constraint violation and should not be followed.

    Args:
        g: Gradient vector in full parameter space (size = n_params).
        null_space: Null-space matrix from extract_free_dof or linearize_at.

    Returns:
        Projected gradient in full parameter space (same shape as g).
    """
    return null_space @ (null_space.T @ g)


def project_to_manifold(
    full: Array,
    constraints: list[Constraint],
    param_list: list[Parameter],
) -> Array:
    """One Newton step projecting a point back onto the constraint manifold.

    Computes: full ← full − Jᵀ(JJᵀ)⁻¹ r(full)

    This is the minimum-norm correction that satisfies the linearized
    constraints at `full`. For affine constraints it is exact; for nonlinear
    constraints it is first-order accurate and can be iterated.

    Use this inside a gradient descent loop after an unconstrained step to
    keep iterates on (or near) the constraint manifold. For solving the full
    constraint system from scratch, use `solve_constraints` instead.

    Args:
        full: Current point in full parameter space (size = n_params).
        constraints: List of active constraints.
        param_list: Ordered list of free Parameter objects.

    Returns:
        Corrected point in full parameter space.
    """
    if not constraints:
        return full
    fixed_by_name = _fixed_params(constraints)
    flat_fn = _build_residual_fn(constraints, param_list, fixed_by_name)
    r = flat_fn(full)
    J = jax.jacobian(flat_fn)(full)
    lam = jnp.linalg.solve(J @ J.T, r)
    return full - J.T @ lam


def in_null_space(objective_fn, null_space: Array, base_point: Array):
    """Reparametrize an objective to operate in the constraint null space.

    Returns a new callable that takes a reduced coordinate vector and maps it
    to full parameter space before evaluating:

        wrapped(reduced) = objective_fn(base_point + null_space @ reduced)

    The wrapped function can be passed directly to jax.value_and_grad and any
    standard optimizer — no explicit gradient projection is needed. Use
    base_point = x0 and null_space from extract_free_dof for a fixed null
    space, or update base_point and null_space each step (from linearize_at)
    for a relinearized variant.

    Args:
        objective_fn: Callable mapping a full-space array to a scalar.
        null_space: Null-space matrix (n_params × n_free).
        base_point: Origin of the reduced coordinate frame in full space.

    Returns:
        Wrapped objective taking a reduced vector of size n_free.
    """

    def wrapped(reduced: Array):
        return objective_fn(base_point + null_space @ reduced)

    return wrapped


def null_space_update(
    full: Array,
    g: Array,
    null_space: Array,
    optimizer,
    opt_state,
) -> tuple[Array, object]:
    """Apply an optimizer update in the reduced null-space coordinate frame.

    Projects the gradient to the null space, applies the optimizer update in
    that lower-dimensional space, and maps the result back to full parameter
    space. Compatible with any optax optimizer (or any object with a
    `.update(grads, state)` method that returns `(updates, new_state)` where
    updates are added to parameters).

    Use this for null-space GD / Adam steps (approaches 1a and 1b).

    Args:
        full: Current point in full parameter space.
        g: Gradient of the objective in full parameter space.
        null_space: Null-space matrix from extract_free_dof or linearize_at.
        optimizer: Optax optimizer instance.
        opt_state: Optimizer state (must have been initialized from a reduced
            vector of size null_space.shape[1]).

    Returns:
        Tuple of (updated full-space point, new optimizer state).
    """
    g_reduced = null_space.T @ g
    updates_reduced, new_state = optimizer.update(g_reduced, opt_state)
    return full + null_space @ updates_reduced, new_state


def projected_update(
    full: Array,
    g: Array,
    constraints: list[Constraint],
    param_list: list[Parameter],
    optimizer,
    opt_state,
) -> tuple[Array, object]:
    """Apply an optimizer update in full space then project back to the manifold.

    Takes an unconstrained optimizer step, then calls project_to_manifold to
    snap the result back onto the constraint manifold (one Newton correction).

    Use this for projected-GD / Adam steps (approach 2).

    Args:
        full: Current point in full parameter space (should be on the manifold).
        g: Gradient of the objective in full parameter space.
        constraints: List of active constraints.
        param_list: Ordered list of free Parameter objects.
        optimizer: Optax optimizer instance.
        opt_state: Optimizer state initialized from a full-space vector.

    Returns:
        Tuple of (updated full-space point on the manifold, new optimizer state).
    """
    updates, new_state = optimizer.update(g, opt_state)
    return project_to_manifold(full + updates, constraints, param_list), new_state
