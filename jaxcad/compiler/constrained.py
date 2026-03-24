"""Constraint-aware parameter extraction."""

from typing import Optional
import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF
from jaxcad.compiler.extraction import extract_parameters
from jaxcad.constraints.graph import ConstraintGraph
from jaxcad.geometry.parameters import Vector, Scalar

def extract_parameters_with_constraints(
    sdf: SDF
) -> tuple[Array, Array, Array, list]:
    """Extract parameters from SDF and apply constraint reduction.

    This function combines SDF parameter extraction with constraint-based
    DOF reduction, providing a complete parameter space for optimization.

    Constraints are automatically discovered from the parameters in the SDF tree.
    When a constraint is created (e.g., DistanceConstraint(p1, p2, 1.0)), it
    automatically registers itself on the parameters it references. This function
    collects all constraints from the discovered parameters and builds a
    ConstraintGraph implicitly.

    Args:
        sdf: The SDF tree to extract parameters from

    Returns:
        Tuple of:
        - reduced_params: Reduced DOF parameter vector (size = total_dof - constraint_dof)
        - null_space: Null space matrix for projecting reduced → full params
        - base_point: Base point in full parameter space (current param values)
        - param_list: Ordered list of Parameter objects (for reference)

    Example:
        # Build geometry with constraints
        p1 = Vector([0, 0, 0], free=True, name='p1')
        p2 = Vector([1, 0, 0], free=True, name='p2')

        # Constraint automatically registers on p1 and p2
        DistanceConstraint(p1, p2, 1.0)

        line = Line(start=p1, end=p2)
        capsule = from_line(line, radius=0.5)

        # Extract constrained parameters (no graph needed!)
        reduced, null_space, base, params = extract_parameters_with_constraints(capsule)

        # Now optimize in reduced space
        def loss_fn(reduced_params):
            full = base + null_space @ reduced_params
            # ... use full params ...
    """
    # Extract free parameters from SDF tree
    free_params_dict, fixed_params_dict = extract_parameters(sdf)

    # Get list of unique free Parameter objects
    param_set = set()
    param_list = []

    for param in free_params_dict.values():
        if param.name not in param_set:
            param_set.add(param.name)
            param_list.append(param)

    # Discover constraints from parameters
    constraint_set = set()
    discovered_constraints = []

    for param in param_list:
        for constraint in param.get_constraints():
            if constraint not in constraint_set:
                constraint_set.add(constraint)
                discovered_constraints.append(constraint)


    constraint_graph = ConstraintGraph()
    for constraint in discovered_constraints:
        constraint_graph.add_constraint(constraint)

    # Extract reduced DOF using constraint graph
    reduced_params, null_space = constraint_graph.extract_free_dof(param_list)

    base_parts = []
    for param in param_list:
        if isinstance(param, Vector):
            base_parts.append(param.xyz)
        elif isinstance(param, Scalar):
            base_parts.append(jnp.array([param.value]))

    base_point = jnp.concatenate(base_parts) if base_parts else jnp.array([])

    return reduced_params, null_space, base_point, param_list
