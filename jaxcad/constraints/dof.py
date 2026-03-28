"""Free functions for constraint DOF analysis and null-space projection."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.constraints.types.base import Constraint
from jaxcad.geometry.parameters import (
    NamedParams,
    Parameter,
    Scalar,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _collect_constraints(params: NamedParams) -> list[Constraint]:
    """Discover all constraints attached to the given parameters (deduplicated)."""
    seen: set = set()
    constraints = []
    for param in params.values():
        for constraint in param.get_constraints():
            if id(constraint) not in seen:
                seen.add(id(constraint))
                constraints.append(constraint)
    return constraints


def _fixed_params(constraints: list[Constraint]) -> dict[str, Array]:
    """Collect values of non-free parameters referenced by constraints."""
    seen: set = set()
    result: dict[str, Array] = {}
    for constraint in constraints:
        for param in constraint.get_parameters():
            if param.name not in seen:
                seen.add(param.name)
                if not param.free:
                    result[param.name] = param.value
    return result


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


def compute_param_vector(params: NamedParams) -> Array:
    """Build the initial flat parameter vector from an ordered params dict."""
    return jnp.concatenate([jnp.atleast_1d(p.value) for p in params.values()])


def unpack_param_vector(x_flat: Array, params: NamedParams) -> dict[str, Array]:
    """Inverse of compute_param_vector: split a flat array back into named values.

    Args:
        x_flat: Flat parameter vector as returned by :func:`compute_param_vector`.
        params: The same ordered NamedParams used to build ``x_flat``.

    Returns:
        Dict mapping parameter name to its value (scalars are 0-d arrays).
    """
    result: dict[str, Array] = {}
    offset = 0
    for name, p in params.items():
        size = p.value.size
        val = x_flat[offset : offset + size]
        result[name] = val[0] if isinstance(p, Scalar) else val
        offset += size
    return result


def _as_vec(v: Parameter | Array) -> Array:
    return jnp.atleast_1d(v.value if isinstance(v, Parameter) else v)


class NullSpaceMap:
    """Null-space projection matrix that operates directly on parameter dicts.

    Wraps the null-space matrix N (shape n_params × n_free) together with the
    parameter structure so that dict ↔ vector conversions are automatic.

    Supports:
        ``N @ reduced_vec``  →  ``dict[str, Array]``  (expand reduced → full space)
        ``full_dict @ N``    →  ``Array``              (project full → reduced space)
        ``full_vec  @ N``    →  ``Array``              (same, from a flat vector)

    The raw matrix is accessible via ``N.matrix`` if needed.
    """

    def __init__(self, matrix: Array, params: NamedParams) -> None:
        self._matrix = matrix
        self._params = params

    @property
    def matrix(self) -> Array:
        return self._matrix

    @property
    def shape(self) -> tuple[int, ...]:
        return self._matrix.shape

    def __matmul__(self, reduced: Array) -> dict[str, Array]:
        """``N @ r``: expand a reduced-space vector to a full-space dict."""
        return unpack_param_vector(self._matrix @ reduced, self._params)

    def pack(self, full_dict: dict) -> Array:
        """Pack a full-space dict into a flat vector (consistent with params ordering)."""
        return jnp.concatenate(
            [_as_vec(full_dict[name]) for name in self._params if name in full_dict]
        )

    def unpack(self, flat: Array) -> dict[str, Array]:
        """Unpack a flat vector into a full-space dict (inverse of pack)."""
        return unpack_param_vector(flat, self._params)

    def __rmatmul__(self, other: dict | Array) -> Array:
        """``x @ N``: project a full-space dict or flat vector to reduced coords."""
        vec = self.pack(other) if isinstance(other, dict) else other
        return vec @ self._matrix

    def __repr__(self) -> str:
        n, k = self._matrix.shape
        return f"NullSpaceMap(n_params={n}, n_free={k})"


def build_residual_fn(
    constraints: list[Constraint],
    params: NamedParams,
) -> Callable[[Array], Array]:
    """Build a flat residual function over all constraints.

    Each constraint's residuals are scaled by its ``weight`` attribute before
    concatenation. Set ``constraint.weight`` at construction time to normalize
    across types or scales.
    """
    fixed_by_name = _fixed_params(constraints)
    weights: tuple[float, ...] = tuple(c.weight for c in constraints)
    param_list = list(params.values())

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


def null_space(
    free_params: dict[str, Array],
    metadata: dict[str, Parameter],
) -> NullSpaceMap:
    """Compute the null-space projection for parameters under constraints.

    Collects constraints from the metadata, linearizes the constraint manifold
    at the current free_params values, and returns a :class:`NullSpaceMap`.

    Args:
        free_params: Name-keyed dict mapping parameter names to plain Arrays
            (as returned by :func:`~jaxcad.extraction.extract_parameters`).
        metadata: Name-keyed dict mapping parameter names to Parameter objects
            (carries constraints, bounds, type info).

    Returns:
        A :class:`NullSpaceMap` ``N`` of shape ``(n_params, n_free)`` such that:
        ``N @ r`` expands reduced coords to a full-space dict, and
        ``full_dict @ N`` projects a full-space dict to reduced coords.
    """
    constraints = _collect_constraints(metadata)
    x0 = jnp.concatenate([jnp.atleast_1d(free_params[name]) for name in metadata])
    n_params = x0.shape[0]

    if not constraints:
        matrix = jnp.eye(n_params)
    else:
        flat_fn = build_residual_fn(constraints, metadata)
        jacobian = jax.jacobian(flat_fn)(x0)
        matrix = _compute_null_space(jacobian, n_params)

    return NullSpaceMap(matrix, metadata)
