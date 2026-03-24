"""Compilation of SDF trees to pure JAX functions."""

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


def functionalize(sdf: SDF) -> Callable:
    """Compile an SDF to a pure function with free and fixed parameters.

    Returns a curried function with signature:
        sdf_fn(free_params, fixed_params) -> (point -> distance)

    Args:
        sdf: The SDF to compile

    Returns:
        Callable: Curried function ``sdf_fn(free_params, fixed_params) -> (point -> distance)``
            mapping parameter dicts to a callable ``point: Array (3,) -> distance: Array ()``.

    Example:
        ```python
        radius = Scalar(value=1.0, free=True, name='radius')
        sphere = Sphere(radius=radius)
        sdf_fn = functionalize(sphere)
        distance = sdf_fn({'sphere_0.radius': 2.0}, {})(jnp.array([0., 0., 0.]))
        ```
    """
    from jaxcad.sdf.boolean.base import BooleanOp
    from jaxcad.sdf.primitives.base import Primitive
    from jaxcad.sdf.transforms.base import Transform

    node_counter = {"count": 0}

    def extract_value(val: Any) -> Any:
        """Extract raw numeric value from Parameter or pass through raw values."""
        from jaxcad.geometry.parameters import Parameter

        if isinstance(val, Parameter):
            return val.extract_value()
        return val

    def build_function(obj: SDF) -> Callable:
        """Recursively build evaluation function."""
        # Generate node ID for parameter lookup
        class_name = obj.__class__.__name__.lower()
        node_id = f"{class_name}_{node_counter['count']}"
        node_counter["count"] += 1

        # Get the pure SDF function for this class
        pure_sdf = obj.__class__.sdf

        if isinstance(obj, Primitive):
            # Primitives: sdf(p, **params)
            # Capture obj and node_id in closure
            params_snapshot = obj.params
            current_node_id = node_id

            def eval_primitive(p: Array, free_params: dict, fixed_params: dict) -> Array:
                # Collect parameter values from both dicts
                param_values = {}
                for param_name in params_snapshot:
                    param_path = f"{current_node_id}.{param_name}"
                    if param_path in free_params:
                        param_values[param_name] = extract_value(free_params[param_path])
                    elif param_path in fixed_params:
                        param_values[param_name] = extract_value(fixed_params[param_path])
                return pure_sdf(p, **param_values)

            return eval_primitive

        elif isinstance(obj, Transform):
            # Transforms: sdf(child_fn, p, **params)
            child_fn = build_function(obj.sdf)
            params_snapshot = obj.params
            current_node_id = node_id

            def eval_transform(p: Array, free_params: dict, fixed_params: dict) -> Array:
                # Collect parameter values
                param_values = {}
                for param_name in params_snapshot:
                    param_path = f"{current_node_id}.{param_name}"
                    if param_path in free_params:
                        param_values[param_name] = extract_value(free_params[param_path])
                    elif param_path in fixed_params:
                        param_values[param_name] = extract_value(fixed_params[param_path])

                # Call child function with parameters
                def child_eval(p_inner):
                    return child_fn(p_inner, free_params, fixed_params)

                return pure_sdf(child_eval, p, **param_values)

            return eval_transform

        elif isinstance(obj, BooleanOp):
            # Boolean ops: sdf(child_fns_tuple, p, **params)
            child_fns = [build_function(child) for child in obj.sdfs]
            params_snapshot = obj.params
            current_node_id = node_id

            def eval_boolean(
                p: Array,
                free_params: dict,
                fixed_params: dict,
                _child_fns=child_fns,
                _params=params_snapshot,
                _node_id=current_node_id,
            ) -> Array:
                # Collect parameter values
                param_values = {}
                for param_name in _params:
                    param_path = f"{_node_id}.{param_name}"
                    if param_path in free_params:
                        param_values[param_name] = extract_value(free_params[param_path])
                    elif param_path in fixed_params:
                        param_values[param_name] = extract_value(fixed_params[param_path])

                # Build child eval closures
                child_evals = tuple(
                    (lambda p_inner, fn=fn: fn(p_inner, free_params, fixed_params))
                    for fn in _child_fns
                )
                return pure_sdf(child_evals, p, **param_values)

            return eval_boolean

        else:
            # Fallback
            return lambda _p, _fp, _fp2: jnp.array(0.0)

    inner = build_function(sdf)
    return lambda free_params, fixed_params: lambda p: inner(p, free_params, fixed_params)
