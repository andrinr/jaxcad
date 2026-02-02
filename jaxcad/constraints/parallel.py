"""Parallel constraint between two vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector, Parameter
from jaxcad.constraints.base import Constraint


@dataclass
class ParallelConstraint(Constraint):
    """Constraint that forces two vectors to be parallel.

    This is equivalent to AngleConstraint with angle=0 or angle=π.

    Reduces DOF by 2 (two scalar equations from cross product = 0).

    Args:
        vector1: First direction vector (Vector parameter)
        vector2: Second direction vector (Vector parameter)

    Example:
        v1 = Vector([1, 0, 0], free=True, name='v1')
        v2 = Vector([2, 0, 0], free=True, name='v2')

        # Force vectors parallel
        constraint = ParallelConstraint(v1, v2)
    """

    vector1: Vector
    vector2: Vector

    def compute_residual(self, param_values: Dict[str, Array]) -> Array:
        """Compute parallel constraint residual: v1 × v2.

        Two vectors are parallel iff their cross product is zero.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Cross product (3D array, should be ~[0, 0, 0] when satisfied)
        """
        v1_name = self.vector1.name
        v2_name = self.vector2.name

        if v1_name not in param_values or v2_name not in param_values:
            raise ValueError(f"Parameter values must include '{v1_name}' and '{v2_name}'")

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Cross product (for 3D vectors only)
        return jnp.cross(v1_val, v2_val)

    def jacobian(self, param_values: Dict[str, Array]) -> Array:
        """Compute Jacobian of parallel constraint.

        For v1 × v2 = 0, we get 3 equations but only 2 are independent.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Jacobian array of shape (3, total_params)
        """
        def residual_fn(v1, v2):
            return jnp.cross(v1, v2)

        v1_name = self.vector1.name
        v2_name = self.vector2.name

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Jacobian of cross product
        jac = jax.jacobian(residual_fn, argnums=(0, 1))(v1_val, v2_val)

        # Concatenate Jacobians w.r.t. v1 and v2
        # jac[0] has shape (3, 3) for v1, jac[1] has shape (3, 3) for v2
        return jnp.concatenate([jac[0], jac[1]], axis=1)

    def dof_reduction(self) -> int:
        """Parallel constraint adds 2 independent equations (cross product in 3D)."""
        return 2

    def get_parameters(self) -> List[Parameter]:
        """Return both vectors involved in the parallel constraint."""
        return [self.vector1, self.vector2]

    def __repr__(self) -> str:
        return f"ParallelConstraint({self.vector1.name}, {self.vector2.name})"
