"""Perpendicular constraint between two vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector, Parameter
from jaxcad.constraints.base import Constraint


@dataclass
class PerpendicularConstraint(Constraint):
    """Constraint that forces two vectors to be perpendicular.

    This is equivalent to AngleConstraint with angle=π/2.

    Reduces DOF by 1 (one scalar equation: dot product = 0).

    Args:
        vector1: First direction vector (Vector parameter)
        vector2: Second direction vector (Vector parameter)

    Example:
        v1 = Vector([1, 0, 0], free=True, name='v1')
        v2 = Vector([0, 1, 0], free=True, name='v2')

        # Force vectors perpendicular
        constraint = PerpendicularConstraint(v1, v2)
    """

    vector1: Vector
    vector2: Vector

    def __post_init__(self):
        """Populate params dict."""
        self.params = {
            'vector1': self.vector1,
            'vector2': self.vector2,
        }

        # Register constraint on parameters
        self._register_constraint()

    def compute_residual(self, param_values: Dict[str, Array]) -> Array:
        """Compute perpendicular constraint residual: v1 · v2.

        Two vectors are perpendicular iff their dot product is zero.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (dot product, should be 0 when satisfied)
        """
        v1_name = self.vector1.name
        v2_name = self.vector2.name

        if v1_name not in param_values or v2_name not in param_values:
            raise ValueError(f"Parameter values must include '{v1_name}' and '{v2_name}'")

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Dot product
        return jnp.dot(v1_val, v2_val)

    def jacobian(self, param_values: Dict[str, Array]) -> Array:
        """Compute Jacobian of perpendicular constraint.

        For v1 · v2 = 0:
        ∂/∂v1 = v2
        ∂/∂v2 = v1

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Jacobian array of shape (1, total_params)
        """
        v1_name = self.vector1.name
        v2_name = self.vector2.name

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Gradient of dot product
        grad_v1 = v2_val
        grad_v2 = v1_val

        return jnp.concatenate([grad_v1, grad_v2])

    def dof_reduction(self) -> int:
        """Perpendicular constraint adds 1 scalar equation."""
        return 1

    def get_parameters(self) -> List[Parameter]:
        """Return both vectors involved in the perpendicular constraint."""
        return [self.vector1, self.vector2]

    def __repr__(self) -> str:
        return f"PerpendicularConstraint({self.vector1.name}, {self.vector2.name})"
