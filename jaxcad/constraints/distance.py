"""Distance constraint between two points/vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector, Scalar, Parameter
from jaxcad.constraints.base import Constraint


@dataclass
class DistanceConstraint(Constraint):
    """Constraint that fixes the distance between two points/vectors.

    This constraint enforces: ||p1 - p2|| = distance

    Reduces DOF by 1 (one scalar equation).

    Args:
        param1: First point/vector (Vector parameter)
        param2: Second point/vector (Vector parameter)
        distance: Target distance (Scalar parameter or float)

    Example:
        p1 = Vector([0, 0, 0], free=True, name='p1')
        p2 = Vector([1, 0, 0], free=True, name='p2')

        # Fix distance to 0.2
        constraint = DistanceConstraint(p1, p2, Scalar(0.2))

        # Or with raw float
        constraint = DistanceConstraint(p1, p2, 0.2)
    """

    param1: Vector
    param2: Vector
    distance: Scalar | float

    def __post_init__(self):
        """Convert distance to Scalar if needed."""
        from jaxcad.geometry.parameters import as_parameter
        if not isinstance(self.distance, Scalar):
            self.distance = as_parameter(self.distance)

    def compute_residual(self, param_values: Dict[str, Array]) -> Array:
        """Compute distance constraint residual: ||p1 - p2|| - d.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when constraint is satisfied)
        """
        # Get parameter values
        p1_name = self.param1.name
        p2_name = self.param2.name

        if p1_name not in param_values or p2_name not in param_values:
            raise ValueError(f"Parameter values must include '{p1_name}' and '{p2_name}'")

        p1_val = param_values[p1_name]
        p2_val = param_values[p2_name]

        # Compute distance
        current_dist = jnp.linalg.norm(p1_val - p2_val)
        target_dist = self.distance.value

        # Residual: ||p1 - p2|| - d
        return current_dist - target_dist

    def jacobian(self, param_values: Dict[str, Array]) -> Array:
        """Compute Jacobian of distance constraint.

        For ||p1 - p2|| - d = 0:
        ∂/∂p1 = (p1 - p2) / ||p1 - p2||
        ∂/∂p2 = -(p1 - p2) / ||p1 - p2||

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Jacobian array of shape (1, total_params)
        """
        # Use JAX automatic differentiation
        def residual_fn(p1, p2):
            diff = p1 - p2
            return jnp.linalg.norm(diff) - self.distance.value

        p1_name = self.param1.name
        p2_name = self.param2.name

        p1_val = param_values[p1_name]
        p2_val = param_values[p2_name]

        # Compute gradients
        grad_p1, grad_p2 = jax.grad(residual_fn, argnums=(0, 1))(p1_val, p2_val)

        return jnp.concatenate([grad_p1, grad_p2])

    def dof_reduction(self) -> int:
        """Distance constraint adds 1 scalar equation, reducing DOF by 1."""
        return 1

    def get_parameters(self) -> List[Parameter]:
        """Return both parameters involved in the distance constraint."""
        return [self.param1, self.param2]

    def __repr__(self) -> str:
        return f"DistanceConstraint({self.param1.name}, {self.param2.name}, d={self.distance.value})"
