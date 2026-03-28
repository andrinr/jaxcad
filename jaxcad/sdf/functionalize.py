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

    def build_function(obj: SDF) -> Callable:
        node_id = f"{obj.__class__.__name__.lower()}_{node_counter['count']}"
        node_counter["count"] += 1
        pure_sdf = obj.__class__.sdf
        params_snapshot = obj.params
        child_fns = [build_function(c) for c in obj.children()]

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


# def reduce(sdf_fn: Callable, sdf: SDF) -> Callable:
#     """Transform a functionalized SDF to operate in constraint null-space coordinates.

#     Takes a function produced by :func:`functionalize` and returns a new function
#     where the free-parameter dict is replaced by a reduced coordinate vector
#     living in the constraint null space.

#     Args:
#         sdf_fn: Callable produced by :func:`functionalize` with signature
#             ``(free_params: dict, fixed_params: dict) -> (point -> distance)``.
#         sdf: The SDF tree whose constraints define the null space.

#     Returns:
#         Callable with signature
#         ``(reduced_free: Array, fixed_params: dict) -> (point -> distance)``.
#     """
#     from jaxcad.constraints.dof import free_params_to_reduced
#     from jaxcad.extraction import extract_parameters
#     from jaxcad.geometry.parameters import Scalar

#     free_params_dict, _ = extract_parameters(sdf)
#     _, null_space, base_point, params = free_params_to_reduced(free_params_dict)

#     sizes = [p.value.size for p in params.values()]
#     names = list(params.keys())
#     is_scalar = [isinstance(p, Scalar) for p in params.values()]

#     def reduced_sdf_fn(reduced_free: Array, fixed_params: dict) -> Callable:
#         full_flat = base_point + null_space @ reduced_free

#         name_to_value: dict = {}
#         offset = 0
#         for name, size, scalar in zip(names, sizes, is_scalar):
#             val = full_flat[offset : offset + size]
#             name_to_value[name] = val[0] if scalar else val
#             offset += size

#         reconstructed = {
#             path: name_to_value[param.name] for path, param in free_params_dict.items()
#         }
#         return sdf_fn(reconstructed, fixed_params)

#     return reduced_sdf_fn


# def functionalize_reduced(sdf: SDF) -> Callable:
#     """Compile an SDF to a function over constraint null-space coordinates.

#     Equivalent to ``reduce(functionalize(sdf), sdf)``. The free-parameter
#     dict is replaced by a reduced coordinate vector in the constraint null space.

#     Args:
#         sdf: The SDF tree to compile.

#     Returns:
#         Callable with signature
#         ``(reduced_free: Array, fixed_params: dict) -> (point -> distance)``.

#     Example:
#         ```python
#         reduced_fn = functionalize_reduced(scene)
#         distance = reduced_fn(reduced_params, {})(jnp.array([0., 0., 0.]))
#         ```
#     """
#     return reduce(functionalize(sdf), sdf)
