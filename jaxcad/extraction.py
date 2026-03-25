"""Parameter extraction from SDF trees."""

from typing import Any

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


def extract_parameters(sdf: SDF) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract free and fixed parameters from an SDF tree.

    Args:
        sdf: The SDF to extract parameters from

    Returns:
        free_params (dict): Dict mapping parameter paths to free Parameter objects.
            Paths are in format "node_id.param_name" (e.g., "sphere_0.radius").
        fixed_params (dict): Dict mapping parameter paths to fixed Parameter objects.
    """
    from jaxcad.sdf.boolean.base import BooleanOp
    from jaxcad.sdf.transforms.base import Transform

    free_params = {}
    fixed_params = {}
    node_counter = {"count": 0}

    def walk(obj: SDF) -> None:
        class_name = obj.__class__.__name__.lower()
        node_id = f"{class_name}_{node_counter['count']}"
        node_counter["count"] += 1

        if hasattr(obj, "params"):
            for param_name, param in obj.params.items():
                param_path = f"{node_id}.{param_name}"
                if param.free:
                    free_params[param_path] = param
                else:
                    fixed_params[param_path] = param

        if isinstance(obj, Transform):
            walk(obj.sdf)
        elif isinstance(obj, BooleanOp):
            for child in obj.sdfs:
                walk(child)

    walk(sdf)
    return free_params, fixed_params


def extract_parameters_with_constraints(
    sdf: SDF,
) -> tuple[Array, Array, Array, list]:
    """Extract parameters from SDF and apply constraint-based DOF reduction.

    Discovers all constraints registered on the free parameters and returns
    the reduced coordinate vector and null-space matrix needed to project back
    to full parameter space.

    Args:
        sdf: The SDF tree to extract parameters from.

    Returns:
        reduced_params (Array): Reduced DOF vector (size = total_dof - constraint_dof).
        null_space (Array): Matrix mapping reduced → full parameter space.
        base_point (Array): Current parameter values as a flat array.
        param_list (list): Ordered list of free Parameter objects.

    Example:
        ```python
        reduced, null_space, base, params = extract_parameters_with_constraints(scene)
        full_params = base + null_space @ reduced
        ```
    """
    from jaxcad.constraints.dof import _collect_constraints, extract_free_dof
    from jaxcad.geometry.parameters import Scalar, Vector

    free_params_dict, _ = extract_parameters(sdf)

    seen: set = set()
    param_list = []
    for param in free_params_dict.values():
        if param.name not in seen:
            seen.add(param.name)
            param_list.append(param)

    constraints = _collect_constraints(param_list)
    reduced_params, null_space = extract_free_dof(constraints, param_list)

    base_parts = []
    for param in param_list:
        if isinstance(param, Vector):
            base_parts.append(param.xyz)
        elif isinstance(param, Scalar):
            base_parts.append(jnp.array([param.value]))

    base_point = jnp.concatenate(base_parts) if base_parts else jnp.array([])

    return reduced_params, null_space, base_point, param_list
