"""Rebuild evaluation functions from computation graphs."""

from __future__ import annotations

from typing import Callable, Optional, Dict, Any

import jax.numpy as jnp
from jax import Array

from jaxcad.compiler.graph import SDFGraph, GraphNode


def rebuild_from_graph(
    graph: SDFGraph,
    param_values: Optional[Dict[str, Dict[str, Any]]] = None
) -> Callable[[Array], Array]:
    """Rebuild a pure evaluation function from a computation graph.

    Args:
        graph: The computation graph to rebuild from
        param_values: Optional dict of parameter values to use instead of stored values
                     Format: {'sphere_0': {'radius': 1.5}, 'translate_1': {'offset': [1,0,0]}}

    Returns:
        Pure function: (point: Array) -> distance: Array

    Example:
        # Using stored parameter values
        graph = extract_graph(sdf)
        eval_fn = rebuild_from_graph(graph)
        distance = eval_fn(point)

        # Using custom parameter values (for optimization)
        eval_fn = rebuild_from_graph(graph, param_values=optimized_params)
        distance = eval_fn(point)
    """
    # Import base classes
    from jaxcad.primitives.base import Primitive
    from jaxcad.transforms.base import Transform
    from jaxcad.boolean.base import BooleanOp
    from jaxcad.parameters import Vector

    # Map transform classes to their parameter names
    TRANSFORM_PARAMS = {
        'Translate': 'offset',
        'Rotate': 'angle',
        'Scale': 'scale',
        'Twist': 'strength',
    }

    def rebuild_node(node: GraphNode) -> Callable[[Array], Array]:
        """Recursively rebuild evaluation function from graph node."""

        # Primitives - leaf nodes
        if node.child_sdf is not None:
            prim = node.child_sdf
            prim_class = prim.__class__
            prim_name = prim_class.__name__.lower()
            node_key = f"{prim_name}_{node.node_id}"

            # Get the pure sdf function from the class
            pure_sdf = prim_class.sdf

            # Check if we have custom parameter values
            if param_values and node_key in param_values:
                # Use provided parameters
                prim_params = param_values[node_key]
                return lambda p: pure_sdf(p, **prim_params)
            else:
                # Extract stored parameter values from the instance
                param_dict = {}
                for attr_name, attr_value in prim.__dict__.items():
                    if attr_name.endswith('_param'):
                        param_name = attr_name.replace('_param', '')
                        if isinstance(attr_value, Vector):
                            param_dict[param_name] = attr_value.xyz
                        else:
                            param_dict[param_name] = attr_value.value
                return lambda p: pure_sdf(p, **param_dict)

        # Boolean operations - binary nodes
        if issubclass(node.op_class, BooleanOp):
            left_fn = rebuild_node(node.children[0])
            right_fn = rebuild_node(node.children[1])
            smoothness = node.params.get('smoothness', 0.1)
            # Use the operation's static sdf method directly
            return lambda p: node.op_class.sdf(left_fn, right_fn, p, smoothness)

        # Transforms - unary nodes
        if issubclass(node.op_class, Transform):
            child_fn = rebuild_node(node.children[0])
            # Get parameter name for this transform
            param_name = TRANSFORM_PARAMS.get(node.op_class.__name__)
            if param_name:
                param_value = node.params.get(param_name)
                # Use the transform's static sdf method directly
                return lambda p: node.op_class.sdf(child_fn, p, param_value)

        # Fallback
        return lambda _: jnp.array(0.0)

    # Rebuild from root node
    if graph.nodes:
        return rebuild_node(graph.nodes[-1])
    else:
        return lambda _: jnp.array(0.0)
