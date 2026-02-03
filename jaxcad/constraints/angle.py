"""Angle constraint between two vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector, Scalar, Parameter
from jaxcad.constraints.base import Constraint


@dataclass
class AngleConstraint(Constraint):
    """Constraint that fixes the angle between two vectors.

    This constraint enforces: arccos(v1 · v2 / (||v1|| ||v2||)) = angle

    Reduces DOF by 1 (one scalar equation).

    Args:
        vector1: First direction vector (Vector parameter)
        vector2: Second direction vector (Vector parameter)
        angle: Target angle in radians (Scalar parameter or float)

    Example:
        v1 = Vector([1, 0, 0], free=True, name='v1')
        v2 = Vector([0, 1, 0], free=True, name='v2')

        # Fix angle to 90 degrees (π/2 radians)
        constraint = AngleConstraint(v1, v2, jnp.pi / 2)
    """

    vector1: Vector
    vector2: Vector
    angle: Scalar | float

    def __post_init__(self):
        """Convert angle to Scalar and populate params dict."""
        from jaxcad.geometry.parameters import as_parameter
        if not isinstance(self.angle, Scalar):
            self.angle = as_parameter(self.angle)

        # Store in params dict for Fluent pattern
        self.params = {
            'vector1': self.vector1,
            'vector2': self.vector2,
            'angle': self.angle,
        }

        # Register constraint on parameters
        self._register_constraint()

    def compute_residual(self, param_values: Dict[str, Array]) -> Array:
        """Compute angle constraint residual: arccos(v1·v2/(||v1||||v2||)) - θ.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Scalar residual (0 when constraint is satisfied)
        """
        v1_name = self.vector1.name
        v2_name = self.vector2.name

        if v1_name not in param_values or v2_name not in param_values:
            raise ValueError(f"Parameter values must include '{v1_name}' and '{v2_name}'")

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Normalize vectors
        v1_norm = v1_val / jnp.linalg.norm(v1_val)
        v2_norm = v2_val / jnp.linalg.norm(v2_val)

        # Compute angle via dot product
        cos_angle = jnp.clip(jnp.dot(v1_norm, v2_norm), -1.0, 1.0)
        current_angle = jnp.arccos(cos_angle)

        # Residual
        return current_angle - self.angle.value

    def jacobian(self, param_values: Dict[str, Array]) -> Array:
        """Compute Jacobian of angle constraint using JAX autodiff.

        Args:
            param_values: Dict with keys matching parameter names

        Returns:
            Jacobian array of shape (1, total_params)
        """
        def residual_fn(v1, v2):
            v1_norm = v1 / jnp.linalg.norm(v1)
            v2_norm = v2 / jnp.linalg.norm(v2)
            cos_angle = jnp.clip(jnp.dot(v1_norm, v2_norm), -1.0, 1.0)
            return jnp.arccos(cos_angle) - self.angle.value

        v1_name = self.vector1.name
        v2_name = self.vector2.name

        v1_val = param_values[v1_name]
        v2_val = param_values[v2_name]

        # Compute gradients
        grad_v1, grad_v2 = jax.grad(residual_fn, argnums=(0, 1))(v1_val, v2_val)

        return jnp.concatenate([grad_v1, grad_v2])

    def dof_reduction(self) -> int:
        """Angle constraint adds 1 scalar equation, reducing DOF by 1."""
        return 1

    def get_parameters(self) -> List[Parameter]:
        """Return both vectors involved in the angle constraint."""
        return [self.vector1, self.vector2]

    def __repr__(self) -> str:
        return f"AngleConstraint({self.vector1.name}, {self.vector2.name}, θ={self.angle.value})"
