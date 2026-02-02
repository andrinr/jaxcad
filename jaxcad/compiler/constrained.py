"""Constraint-aware parameter extraction."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF
from jaxcad.compiler.extraction import extract_parameters


def extract_parameters_with_constraints(
    sdf: SDF,
    constraint_graph: 'ConstraintGraph'
) -> tuple[Array, Array, Array, list]:
    """Extract parameters from SDF and apply constraint reduction.

    This function combines SDF parameter extraction with constraint-based
    DOF reduction, providing a complete parameter space for optimization.

    Args:
        sdf: The SDF tree to extract parameters from
        constraint_graph: ConstraintGraph containing geometric constraints

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

        graph = ConstraintGraph()
        graph.add_constraint(Distance(p1, p2, 1.0))

        line = Line(start=p1, end=p2)
        capsule = from_line(line, radius=0.5)

        # Extract constrained parameters
        reduced, null_space, base, params = extract_parameters_with_constraints(
            capsule,
            graph
        )

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

    # Extract reduced DOF using constraint graph
    reduced_params, null_space = constraint_graph.extract_free_dof(param_list)

    # Build base point (current parameter values)
    from jaxcad.geometry.parameters import Vector, Scalar
    base_parts = []
    for param in param_list:
        if isinstance(param, Vector):
            base_parts.append(param.xyz)
        elif isinstance(param, Scalar):
            base_parts.append(jnp.array([param.value]))

    base_point = jnp.concatenate(base_parts) if base_parts else jnp.array([])

    return reduced_params, null_space, base_point, param_list
