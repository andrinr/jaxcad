"""Base class for geometric constraints.

A constraint defines a relationship between parameters that must be satisfied.
Each constraint reduces the degrees of freedom (DOF) by the number of
independent constraint equations it introduces.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Dict

from jax import Array

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Parameter


class Constraint(Fluent):
    """Base class for geometric constraints.

    A constraint defines a relationship between parameters that must be satisfied.
    Each constraint reduces the degrees of freedom (DOF) by the number of
    independent constraint equations it introduces.

    Attributes:
        params: Dictionary of Parameter objects referenced by this constraint

    Each Constraint instance should store its parameters in self.params dictionary:
        self.params = {
            'param1': Vector([0, 0, 0], free=True, name='p1'),
            'param2': Vector([1, 0, 0], free=True, name='p2'),
            'distance': Scalar(1.0, free=False, name='d'),
        }

    Constraints are automatically registered on their parameters when created,
    enabling automatic discovery during tree traversal.

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

    params: dict[str, 'Parameter']

    def _register_constraint(self) -> None:
        """Register this constraint on all parameters it references.

        This should be called at the end of __post_init__ in subclasses.
        Enables implicit constraint discovery without explicit ConstraintGraph.
        """
        # Auto-cast params
        if hasattr(self, 'params'):
            self._cast_params()
        # Auto-register on parameters
        for param in self.get_parameters():
            param.add_constraint(self)

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
