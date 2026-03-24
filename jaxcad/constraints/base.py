"""Base class for geometric constraints.

A constraint defines a relationship between parameters that must be satisfied.
Each constraint reduces the degrees of freedom (DOF) by the number of
independent constraint equations it introduces.
"""

from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp
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
    - dof_reduction(): Number of DOF this constraint removes
    - get_parameters(): List of parameters involved in this constraint

    Subclasses may optionally override:
    - jacobian(): Returns ∂c/∂x; defaults to AD on compute_residual
    """

    params: dict[str, Parameter]

    def _register_constraint(self) -> None:
        """Register this constraint on all parameters it references.

        This should be called at the end of __post_init__ in subclasses.
        Enables implicit constraint discovery without explicit ConstraintGraph.
        """
        # Auto-cast params
        if hasattr(self, "params"):
            self._cast_params()
        # Auto-register on parameters
        for param in self.get_parameters():
            param.add_constraint(self)

    @abstractmethod
    def compute_residual(self, param_values: dict[str, Array]) -> Array:
        """Compute constraint residual c(x).

        Args:
            param_values: Dictionary mapping parameter names to their current values

        Returns:
            Constraint residual (scalar or array). Zero means constraint is satisfied.
        """
        pass

    def jacobian(self, param_values: dict[str, Array]) -> Array:
        """Compute constraint Jacobian ∂c/∂x via automatic differentiation.

        Differentiates compute_residual with respect to each parameter returned
        by get_parameters(). Subclasses may override with a manually derived
        Jacobian for performance or numerical reasons.

        Args:
            param_values: Dictionary mapping parameter names to their current values

        Returns:
            Jacobian array of shape (n_residuals, n_params) for vector residuals,
            or (n_params,) for scalar residuals.
        """
        params = self.get_parameters()
        names = [p.name for p in params]
        vals = [param_values[n] for n in names]
        argnums = tuple(range(len(params)))

        def residual(*vals):
            pv = dict(param_values)
            for n, v in zip(names, vals):
                pv[n] = v
            return self.compute_residual(pv)

        r = residual(*vals)
        if jnp.ndim(r) == 0:
            grads = jax.grad(residual, argnums=argnums)(*vals)
            return jnp.concatenate([jnp.atleast_1d(g) for g in grads])
        else:
            jacs = jax.jacobian(residual, argnums=argnums)(*vals)
            return jnp.concatenate(list(jacs), axis=-1)

    @abstractmethod
    def dof_reduction(self) -> int:
        """Number of degrees of freedom this constraint removes.

        Returns:
            Number of independent constraint equations (typically 1)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> list[Parameter]:
        """Get list of parameters involved in this constraint.

        Returns:
            List of Parameter objects that this constraint depends on
        """
        pass
