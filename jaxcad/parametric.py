"""Decorator-based parametric compilation with PyTree parameters.

This module provides a clean decorator API for compiling SDFs to
fully differentiable functional form using JAX pytrees.
"""

from functools import wraps
from typing import Callable

import jax.numpy as jnp
from jax import Array

from jaxcad.compiler.graph import extract_graph, GraphNode
from jaxcad.sdf import SDF


def parametric(sdf_or_builder):
    """Decorator to compile SDF to parametric functional form.

    Automatically extracts all parameters as a PyTree and creates a
    fully differentiable function.

    Usage:
        @parametric
        def my_sdf():
            sphere = Sphere(radius=1.0)
            return sphere.translate([2.0, 0.0, 0.0])

        # my_sdf is now a function: (params, point) -> sdf_value
        params = my_sdf.init_params()
        value = my_sdf(params, point)

        # Optimize with JAX
        grad = jax.grad(lambda p: my_sdf(p, point) ** 2)

    Or use directly on an SDF:
        sphere = Sphere(radius=1.0)
        child_sdf = parametric(sphere)
        params = child_sdf.init_params()
        value = child_sdf(params, point)
    """
    # Handle both @parametric and @parametric() syntax
    if callable(sdf_or_builder):
        # Direct decoration: @parametric
        return _compile_parametric(sdf_or_builder)
    else:
        # Called as @parametric() - return decorator
        def decorator(fn):
            return _compile_parametric(fn)
        return decorator


def _compile_parametric(sdf_or_builder):
    """Internal: Compile SDF or builder to parametric form."""

    # Determine if it's an SDF or a builder function
    if isinstance(sdf_or_builder, SDF):
        # Direct SDF
        sdf = sdf_or_builder
        builder = None
    else:
        # Builder function
        builder = sdf_or_builder
        sdf = builder()  # Call to get SDF

    # Extract computation graph
    graph = extract_graph(sdf)

    # Extract parameters as PyTree
    params_tree, param_map = _extract_params_pytree(graph)

    # Build evaluation function
    eval_fn = _build_eval_fn(graph, param_map)

    # Wrap with nice API
    @wraps(sdf_or_builder)
    def parametric_sdf(params, query_point: Array) -> Array:
        """Evaluate SDF with given parameters and query point.

        Args:
            params: PyTree of parameter values
            query_point: Point to evaluate SDF at

        Returns:
            SDF value at query point
        """
        return eval_fn(params, query_point)

    # Attach metadata
    parametric_sdf.init_params = lambda: params_tree
    parametric_sdf.param_map = param_map
    parametric_sdf.graph = graph

    return parametric_sdf


def _extract_params_pytree(graph):
    """Extract parameters as a PyTree structure using JAX tree utilities.

    Automatically finds all Parameter instances (Distance, Vector, Angle)
    in the graph by traversing the SDF object tree.

    Returns:
        (params_tree, param_map) where:
        - params_tree: Dict with parameter values (JAX pytree)
        - param_map: Mapping from node_id to parameter keys
    """
    from jaxcad.parameters import Parameter, Vector

    params = {}
    param_map = {}

    def extract_from_node(node: GraphNode):
        """Extract parameters from a node using pytree traversal."""
        for child in node.children:
            extract_from_node(child)

        # Extract from primitives
        if node.op_type == OpType.PRIMITIVE and node.child_sdf:
            prim = node.child_sdf
            prim_name = prim.__class__.__name__.lower()
            node_key = f"{prim_name}_{node.node_id}"

            # Use JAX tree utilities to find all Parameter instances
            prim_params = {}
            for attr_name, attr_value in prim.__dict__.items():
                if isinstance(attr_value, Parameter):
                    # Extract just the attribute name (e.g., 'radius_param' -> 'radius')
                    param_name = attr_name.replace('_param', '')
                    # For Vector parameters, extract only xyz (3D) not the full 4D homogeneous coords
                    if isinstance(attr_value, Vector):
                        prim_params[param_name] = attr_value.xyz
                    else:
                        prim_params[param_name] = attr_value.value
                    param_map[(node.node_id, param_name)] = (node_key, param_name)

            if prim_params:
                params[node_key] = prim_params

        # Extract from transforms using a mapping
        transform_params = {
            OpType.TRANSLATE: 'offset',
            OpType.ROTATE: 'angle',
            OpType.SCALE: 'scale',
            OpType.TWIST: 'strength',
        }

        if node.op_type in transform_params:
            param_name = transform_params[node.op_type]
            param_value = node.params.get(param_name)

            if param_value is not None:
                # Special handling for scale (only extract if uniform)
                if node.op_type == OpType.SCALE:
                    param_array = jnp.asarray(param_value)
                    if param_array.ndim != 0 and not isinstance(param_value, (int, float)):
                        # Non-uniform scale - skip
                        return

                # Create parameter entry
                op_name = node.op_type.value
                key = f"{op_name}_{node.node_id}"
                params[key] = {param_name: jnp.asarray(param_value)}
                param_map[node.node_id] = (key, param_name)

    if graph.nodes:
        extract_from_node(graph.nodes[-1])

    return params, param_map


def _build_eval_fn(graph, param_map):
    """Build evaluation function that takes params pytree."""

    # Import all transforms once
    from jaxcad.transforms.affine import Translate, Rotate, Scale
    from jaxcad.transforms.deformations import Twist
    from jaxcad.boolean import smooth_min, smooth_max

    # Mapping from OpType to (transform_class, param_name, pure_function)
    TRANSFORM_DISPATCH = {
        OpType.TRANSLATE: (Translate, 'offset', Translate.sdf),
        OpType.ROTATE: (Rotate, 'angle', Rotate.sdf),
        OpType.SCALE: (Scale, 'scale', Scale.sdf),
        OpType.TWIST: (Twist, 'strength', Twist.sdf),
    }

    # Boolean operation dispatch
    BOOLEAN_DISPATCH = {
        OpType.UNION: lambda l, r, s: smooth_min(l, r, s),
        OpType.INTERSECTION: lambda l, r, s: smooth_max(l, r, s),
        OpType.DIFFERENCE: lambda l, r, s: smooth_max(l, -r, s),
    }

    def eval_fn(params, query_point: Array) -> Array:
        """Evaluate with params pytree."""

        def rebuild_sdf(node: GraphNode) -> Callable:
            """Rebuild SDF from graph using params pytree."""

            # Primitives - use pure static sdf() method
            if node.op_type == OpType.PRIMITIVE:
                prim = node.child_sdf
                prim_class = prim.__class__
                prim_name = prim.__class__.__name__.lower()
                node_key = f"{prim_name}_{node.node_id}"

                # Get the pure sdf function from the class
                pure_sdf = prim_class.sdf

                if node_key in params:
                    # Create closure with optimized parameters
                    prim_params = params[node_key]
                    return lambda p: pure_sdf(p, **prim_params)
                else:
                    # Use stored parameter values
                    # Extract parameter values from the instance
                    param_dict = {}
                    for attr_name, attr_value in prim.__dict__.items():
                        if attr_name.endswith('_param'):
                            param_name = attr_name.replace('_param', '')
                            from jaxcad.parameters import Vector
                            if isinstance(attr_value, Vector):
                                param_dict[param_name] = attr_value.xyz
                            else:
                                param_dict[param_name] = attr_value.value
                    return lambda p: pure_sdf(p, **param_dict)

            # Boolean operations
            if node.op_type in BOOLEAN_DISPATCH:
                left_fn = rebuild_sdf(node.children[0])
                right_fn = rebuild_sdf(node.children[1])
                smoothness = node.params.get('smoothness', 0.1)
                op_fn = BOOLEAN_DISPATCH[node.op_type]
                return lambda p: op_fn(left_fn(p), right_fn(p), smoothness)

            # Transforms
            if node.op_type in TRANSFORM_DISPATCH:
                _, param_name, eval_method = TRANSFORM_DISPATCH[node.op_type]
                child_fn = rebuild_sdf(node.children[0])

                # Get parameter value
                param_key = param_map.get(node.node_id)
                if param_key:
                    param_value = params[param_key[0]][param_key[1]]
                else:
                    param_value = node.params[param_name]

                return lambda p: eval_method(child_fn, p, param_value)

            # Fallback
            return lambda _: 0.0

        if graph.nodes:
            child_sdf = rebuild_sdf(graph.nodes[-1])
            return child_sdf(query_point)
        else:
            return jnp.array(0.0)

    return eval_fn
