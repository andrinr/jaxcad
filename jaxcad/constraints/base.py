"""Base class for geometric constraints.

A constraint defines a relationship between parameters that must be satisfied.
Each constraint reduces the degrees of freedom (DOF) by the number of
independent constraint equations it introduces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

from jax import Array

from jaxcad.geometry.parameters import Parameter


class Constraint(ABC):
    """Base class for geometric constraints.

    A constraint defines a relationship between parameters that must be satisfied.
    Each constraint reduces the degrees of freedom (DOF) by the number of
    independent constraint equations it introduces.

    Constraints are represented as residual equations c(x) = 0:
    - For optimization, we want c(x) ≈ 0
    - The Jacobian ∂c/∂x defines the constraint manifold
    - The null space of J gives the free directions (reduced DOF)

    Subclasses must implement:
    - compute_residual(): Returns constraint violation c(x)
    - jacobian(): Returns ∂c/∂x
    - dof_reduction(): Number of DOF this constraint removes
    - get_parameters(): List of parameters involved in this constraint
    """

    @abstractmethod
    def compute_residual(self, param_values: Dict[str, Array]) -> Array:
        """Compute constraint residual c(x).

        Args:
            param_values: Dictionary mapping parameter names to their current values

        Returns:
            Constraint residual (scalar or array). Zero means constraint is satisfied.
        """
        pass

    @abstractmethod
    def jacobian(self, param_values: Dict[str, Array]) -> Array:
        """Compute constraint Jacobian ∂c/∂x.

        Args:
            param_values: Dictionary mapping parameter names to their current values

        Returns:
            Jacobian matrix of shape (n_constraints, n_params)
        """
        pass

    @abstractmethod
    def dof_reduction(self) -> int:
        """Number of degrees of freedom this constraint removes.

        Returns:
            Number of independent constraint equations (typically 1)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[Parameter]:
        """Get list of parameters involved in this constraint.

        Returns:
            List of Parameter objects that this constraint depends on
        """
        pass
