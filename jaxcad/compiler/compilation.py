"""Compilation of SDF trees to pure JAX functions."""

from typing import Any, Callable, Dict

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


def to_function(sdf: SDF) -> Callable:
    """Compile an SDF to a pure function with free and fixed parameters.

    Returns a function with signature:
        sdf_fn(point, free_params, fixed_params) -> distance

    Args:
        sdf: The SDF to compile

    Returns:
        Pure function that takes:
        - point: Array (3,) - query point
        - free_params: Dict[str, Array] - free parameter values
        - fixed_params: Dict[str, Array] - fixed parameter values

    Example:
        >>> from jaxcad.sdf.primitives import Sphere
        >>> from jaxcad.geometry import Scalar
        >>>
        >>> radius = Scalar(value=1.0, free=True, name='radius')
        >>> sphere = Sphere(radius=radius)
        >>>
        >>> sdf_fn = to_function(sphere)
        >>>
        >>> # Query with specific parameter values
        >>> free_vals = {'sphere_0.radius': 2.0}
        >>> fixed_vals = {}
        >>> distance = sdf_fn(jnp.array([0., 0., 0.]), free_vals, fixed_vals)
    """
    from jaxcad.sdf.primitives.base import Primitive
    from jaxcad.sdf.transforms.base import Transform
    from jaxcad.sdf.boolean.base import BooleanOp

    node_counter = {'count': 0}

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
        node_counter['count'] += 1

        # Get the pure SDF function for this class
        pure_sdf = obj.__class__.sdf

        if isinstance(obj, Primitive):
            # Primitives: sdf(p, **params)
            # Capture obj and node_id in closure
            params_snapshot = obj.params
            current_node_id = node_id

            def eval_primitive(p: Array, free_params: Dict, fixed_params: Dict) -> Array:
                # Collect parameter values from both dicts
                param_values = {}
                for param_name in params_snapshot.keys():
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

            def eval_transform(p: Array, free_params: Dict, fixed_params: Dict) -> Array:
                # Collect parameter values
                param_values = {}
                for param_name in params_snapshot.keys():
                    param_path = f"{current_node_id}.{param_name}"
                    if param_path in free_params:
                        param_values[param_name] = extract_value(free_params[param_path])
                    elif param_path in fixed_params:
                        param_values[param_name] = extract_value(fixed_params[param_path])

                # Call child function with parameters
                child_eval = lambda p_inner: child_fn(p_inner, free_params, fixed_params)
                return pure_sdf(child_eval, p, **param_values)

            return eval_transform

        elif isinstance(obj, BooleanOp):
            # Boolean ops: sdf(child_fns_tuple, p, **params)
            child_fns = [build_function(child) for child in obj.sdfs]
            params_snapshot = obj.params
            current_node_id = node_id

            def eval_boolean(p: Array, free_params: Dict, fixed_params: Dict, _child_fns=child_fns, _params=params_snapshot, _node_id=current_node_id) -> Array:
                # Collect parameter values
                param_values = {}
                for param_name in _params.keys():
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

    return build_function(sdf)
