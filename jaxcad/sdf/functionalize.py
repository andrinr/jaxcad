"""Compilation of SDF trees to pure JAX functions."""

from typing import Callable

from jax import Array

from jaxcad.sdf.base import SDF


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

    node_counter = {"count": 0}

    def collect_params(node_id, params_snapshot, free_params, fixed_params):
        result = {}
        for attr_name, param in params_snapshot.items():
            if param.free:
                result[attr_name] = free_params[param.name]  # name-keyed, plain Array
            else:
                path = f"{node_id}.{attr_name}"
                if path in fixed_params:
                    result[attr_name] = fixed_params[path]  # path-keyed, plain Array
        return result

    def build_function(obj: SDF) -> Callable | None:
        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1

        if not hasattr(obj.__class__, "sdf"):
            # Non-SDF Fluent node (e.g. DiffMaterial) — keep counter in sync,
            # recurse into children, but produce no SDF callable.
            for c in obj.children():
                build_function(c)
            return None

        pure_sdf = obj.__class__.sdf
        params_snapshot = obj.params
        raw_child_fns = [build_function(c) for c in obj.children()]
        child_fns = [f for f in raw_child_fns if f is not None]

        def eval_fn(p: Array, free_params: dict, fixed_params: dict) -> Array:
            param_values = collect_params(node_id, params_snapshot, free_params, fixed_params)
            child_evals = [lambda p_, fn=fn: fn(p_, free_params, fixed_params) for fn in child_fns]
            if isinstance(obj, BooleanOp):
                return pure_sdf(tuple(child_evals), p, **param_values)
            elif child_evals:
                return pure_sdf(child_evals[0], p, **param_values)
            else:
                return pure_sdf(p, **param_values)

        return eval_fn

    inner = build_function(sdf)
    return lambda free_params, fixed_params: lambda p: inner(p, free_params, fixed_params)
