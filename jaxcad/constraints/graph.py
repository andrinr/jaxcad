"""Constraint graph for managing multiple constraints and computing reduced DOF."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints.base import Constraint
from jaxcad.geometry.parameters import Parameter, Scalar, Vector


@dataclass
class ConstraintGraph:
    """Manages a collection of constraints and computes reduced DOF space.

    The constraint graph:
    1. Collects all constraints
    2. Identifies which parameters are involved
    3. Computes the constraint Jacobian matrix J
    4. Finds the null space N of J (free directions)
    5. Maps reduced parameters to full parameters via null space projection

    This enables optimization in a reduced DOF space where constraints
    are automatically satisfied.

    Example:
        # Two points with distance constraint
        p1 = Vector([0, 0, 0], free=True, name='p1')
        p2 = Vector([1, 0, 0], free=True, name='p2')

        graph = ConstraintGraph()
        graph.add_constraint(DistanceConstraint(p1, p2, 0.2))

        # Get reduced DOF
        reduced_params, projection = graph.extract_free_dof([p1, p2])
        # reduced_params has 5 DOF instead of 6
    """

    constraints: list[Constraint] = field(default_factory=list)

    @classmethod
    def from_parameters(cls, param_list: list[Parameter]) -> ConstraintGraph:
        """Build a ConstraintGraph by discovering all constraints attached to the given parameters.

        Constraints register themselves on their parameters at construction time, so this
        collects them without any explicit graph management by the caller.

        Args:
            param_list: Free parameters whose attached constraints should be collected.

        Returns:
            A ConstraintGraph containing all discovered constraints (deduplicated).
        """
        seen: set = set()
        constraints = []
        for param in param_list:
            for constraint in param.get_constraints():
                if id(constraint) not in seen:
                    seen.add(id(constraint))
                    constraints.append(constraint)
        graph = cls()
        graph.constraints = constraints
        return graph

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the graph.

        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)

    def get_total_dof_reduction(self) -> int:
        """Compute total DOF reduction from all constraints.

        Returns:
            Sum of DOF reductions from all constraints
        """
        return sum(c.dof_reduction() for c in self.constraints)

    def get_all_parameters(self) -> list[Parameter]:
        """Get unique list of all parameters involved in constraints.

        Returns:
            List of unique Parameter objects
        """
        params = []
        seen = set()
        for constraint in self.constraints:
            for param in constraint.get_parameters():
                if param.name not in seen:
                    params.append(param)
                    seen.add(param.name)
        return params

    def build_constraint_jacobian(
        self, param_values: dict[str, Array], all_param_names: list[str]
    ) -> Array:
        """Build the full constraint Jacobian matrix.

        Args:
            param_values: Current values of all parameters
            all_param_names: Ordered list of all parameter names

        Returns:
            Jacobian matrix J of shape (n_constraints, n_total_params)
        """
        if not self.constraints:
            return jnp.zeros((0, 0))

        # Calculate total parameter dimension
        total_dim = sum(param_values[name].size for name in all_param_names)

        # Collect Jacobians from all constraints
        jacobians = []
        for constraint in self.constraints:
            # Get parameter indices for this constraint
            constraint_params = [p.name for p in constraint.get_parameters()]

            # Compute Jacobian for just the constrained parameters
            jac_local = constraint.jacobian(param_values)
            if jac_local.ndim == 1:
                jac_local = jac_local.reshape(1, -1)

            # Expand to full parameter space
            jac_full = jnp.zeros((jac_local.shape[0], total_dim))

            # Fill in the Jacobian for the parameters involved in this constraint
            offset = 0
            for name in all_param_names:
                param_dim = param_values[name].size
                if name in constraint_params:
                    # This parameter is involved in the constraint
                    local_idx = constraint_params.index(name)
                    jac_full = jac_full.at[:, offset : offset + param_dim].set(
                        jac_local[:, local_idx * param_dim : (local_idx + 1) * param_dim]
                    )
                offset += param_dim

            jacobians.append(jac_full)

        return jnp.vstack(jacobians)

    def compute_null_space(self, jacobian: Array, n_params: int, tolerance: float = 1e-10) -> Array:
        """Compute null space of constraint Jacobian.

        The null space N contains all directions in parameter space that
        satisfy the constraints (i.e., J @ N = 0).

        Args:
            jacobian: Constraint Jacobian matrix
            n_params: Total number of parameters (dimension of parameter space)
            tolerance: Threshold for considering singular values as zero

        Returns:
            Null space matrix N of shape (n_params, n_free_dof)
        """
        if jacobian.size == 0 or jacobian.shape[0] == 0:
            # No constraints - return identity
            return jnp.eye(n_params)

        # Use SVD to find null space
        _, s, Vt = jnp.linalg.svd(jacobian, full_matrices=True)

        # Find rank (number of non-zero singular values)
        rank = jnp.sum(s > tolerance)

        # Null space is spanned by right singular vectors corresponding to zero singular values
        # V has shape (n_params, n_params), we want columns from rank onwards
        V = Vt.T
        null_space = V[:, rank:]

        return null_space

    def extract_free_dof(self, parameters: list[Parameter]) -> tuple[Array, Array]:
        """Extract reduced degrees of freedom from parameters under constraints.

        This computes:
        1. Current parameter values as a flat array
        2. Constraint Jacobian J
        3. Null space N of J
        4. Reduced coordinates in null space

        Args:
            parameters: List of Parameter objects (should be free=True)

        Returns:
            reduced_params (Array): Reduced DOF vector (size = original_dof - constraint_dof).
            null_space (Array): Null space matrix for projecting reduced → full params.

        Example:
            p1 = Vector([0, 0, 0], free=True, name='p1')
            p2 = Vector([1, 0, 0], free=True, name='p2')

            graph = ConstraintGraph()
            graph.add_constraint(DistanceConstraint(p1, p2, 0.5))

            reduced, null_space = graph.extract_free_dof([p1, p2])
            # reduced has shape (5,) instead of (6,)

            # To get full params from reduced:
            # full_params = base_point + null_space @ reduced
        """
        # Build parameter value dictionary
        param_values = {}
        full_params = []
        param_names = []

        for param in parameters:
            if not param.free:
                warnings.warn(
                    f"Parameter {param.name} is not free, skipping in DOF extraction", stacklevel=2
                )
                continue

            param_names.append(param.name)

            if isinstance(param, Vector):
                param_values[param.name] = param.xyz
                full_params.append(param.xyz)
            elif isinstance(param, Scalar):
                param_values[param.name] = param.value
                full_params.append(jnp.array([param.value]))
            else:
                warnings.warn(f"Unknown parameter type {type(param)}", stacklevel=2)

        # Also include fixed parameters referenced by constraints (needed for Jacobian evaluation)
        for param in self.get_all_parameters():
            if not param.free and param.name not in param_values:
                if isinstance(param, Vector):
                    param_values[param.name] = param.xyz
                elif isinstance(param, Scalar):
                    param_values[param.name] = param.value

        # Flatten to 1D array
        full_params_flat = jnp.concatenate(full_params)
        n_params = full_params_flat.shape[0]

        # Build constraint Jacobian
        jacobian = self.build_constraint_jacobian(param_values, param_names)

        # Compute null space
        null_space = self.compute_null_space(jacobian, n_params)

        # Project current parameters onto null space
        # We want to find reduced coords α such that: full = base + N @ α
        # If we use full as the base point initially, then α = 0
        # For optimization, we'll work with α directly

        # For now, return the null space projection of current params
        # This gives us the component of params that lies in the null space
        reduced_params = null_space.T @ full_params_flat

        return reduced_params, null_space

    def project_to_full(
        self, reduced_params: Array, null_space: Array, base_point: Array | None = None
    ) -> Array:
        """Project reduced DOF parameters back to full parameter space.

        Computes: full_params = base_point + null_space @ reduced_params

        Args:
            reduced_params: Reduced DOF vector
            null_space: Null space matrix from extract_free_dof
            base_point: Base point in full space (default: origin)

        Returns:
            Full parameter vector
        """
        if base_point is None:
            base_point = jnp.zeros(null_space.shape[0])

        return base_point + null_space @ reduced_params

    def __repr__(self) -> str:
        n_constraints = len(self.constraints)
        dof_reduction = self.get_total_dof_reduction()
        return f"ConstraintGraph({n_constraints} constraints, reduces DOF by {dof_reduction})"
