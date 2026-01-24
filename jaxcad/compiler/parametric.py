"""Parametric compilation - convert SDF expressions to constraint-based form."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.compiler.graph import extract_graph, GraphNode, OpType
from jaxcad.constraints import ConstraintSystem, Parameter
from jaxcad.sdf import SDF
from jaxcad.transforms.functional import (
    translate_eval, rotate_z_eval, twist_z_eval, taper_z_eval, scale_eval
)


def compile_parametric(sdf: SDF, constraint_system: ConstraintSystem) -> Callable:
    """Compile SDF to parametric form with constraints.

    Extracts transform parameters as constraints and creates a function
    that can be differentiated with respect to free parameters.

    Args:
        sdf: The SDF expression tree
        constraint_system: Constraint system to populate with parameters

    Returns:
        Callable that takes (params_vector, query_point) -> sdf_value
    """
    # Extract computation graph
    graph = extract_graph(sdf)

    # Walk graph and extract parameters as constraints
    param_map = {}  # Maps node_id -> constraint

    def extract_constraints(node: GraphNode):
        """Recursively extract constraints from graph."""
        # Process children first
        for child in node.children:
            extract_constraints(child)

        # Extract parameters from primitives
        if node.op_type == OpType.PRIMITIVE and node.child_sdf is not None:
            prim = node.child_sdf
            prim_name = prim.__class__.__name__.lower()

            # Extract primitive-specific parameters
            if hasattr(prim, 'radius'):
                name = f"{prim_name}_radius_{node.node_id}"
                param = constraint_system.distance(float(prim.radius), fixed=False, name=name)
                param_map[(node.node_id, 'radius')] = param

            if hasattr(prim, 'size'):
                # Box size is a vector
                name = f"{prim_name}_size_{node.node_id}"
                param = constraint_system.point(prim.size, fixed=False, name=name)
                param_map[(node.node_id, 'size')] = param

            if hasattr(prim, 'height'):
                name = f"{prim_name}_height_{node.node_id}"
                param = constraint_system.distance(float(prim.height), fixed=False, name=name)
                param_map[(node.node_id, 'height')] = param

            if hasattr(prim, 'major_radius'):
                name = f"{prim_name}_major_radius_{node.node_id}"
                param = constraint_system.distance(float(prim.major_radius), fixed=False, name=name)
                param_map[(node.node_id, 'major_radius')] = param

            if hasattr(prim, 'minor_radius'):
                name = f"{prim_name}_minor_radius_{node.node_id}"
                param = constraint_system.distance(float(prim.minor_radius), fixed=False, name=name)
                param_map[(node.node_id, 'minor_radius')] = param

        # Extract parameters from transforms
        if node.op_type == OpType.TRANSLATE:
            offset = node.params.get("offset")
            if offset is not None:
                name = f"translate_{node.node_id}"
                param = constraint_system.point(offset, fixed=False, name=name)
                param_map[node.node_id] = param

        elif node.op_type == OpType.ROTATE:
            angle = node.params.get("angle")
            if angle is not None:
                name = f"rotate_{node.node_id}"
                param = constraint_system.angle(angle, fixed=False, name=name)
                param_map[node.node_id] = param

        elif node.op_type == OpType.SCALE:
            scale = node.params.get("scale")
            if scale is not None:
                name = f"scale_{node.node_id}"
                # Treat scalar scale as distance
                if isinstance(scale, (int, float)):
                    param = constraint_system.distance(float(scale), fixed=False, name=name)
                    param_map[node.node_id] = param

        elif node.op_type in [OpType.TWIST, OpType.TAPER]:
            strength = node.params.get("strength")
            if strength is not None:
                op_name = node.op_type.value
                name = f"{op_name}_{node.node_id}"
                param = constraint_system.distance(strength, fixed=False, name=name)
                param_map[node.node_id] = param

    # Extract all constraints
    if graph.nodes:
        extract_constraints(graph.nodes[-1])

    # Build evaluation function
    def eval_fn(params_vector: Array, query_point: Array) -> Array:
        """Evaluate SDF with given parameters and query point."""
        # Update constraint values from vector
        constraint_system.from_vector(params_vector)

        # Rebuild and evaluate SDF with updated parameters
        def rebuild_sdf(node: GraphNode) -> Callable:
            """Rebuild SDF evaluation from graph with current parameter values."""
            if node.op_type == OpType.PRIMITIVE:
                # Rebuild primitive with updated parameter values
                prim = node.child_sdf
                prim_class = prim.__class__

                # Get updated parameters from constraints
                updated_kwargs = {}

                radius_param = param_map.get((node.node_id, 'radius'))
                if radius_param:
                    # Keep as JAX array for tracing compatibility
                    updated_kwargs['radius'] = radius_param.value

                size_param = param_map.get((node.node_id, 'size'))
                if size_param:
                    updated_kwargs['size'] = size_param.value

                height_param = param_map.get((node.node_id, 'height'))
                if height_param:
                    updated_kwargs['height'] = height_param.value

                major_radius_param = param_map.get((node.node_id, 'major_radius'))
                if major_radius_param:
                    updated_kwargs['major_radius'] = major_radius_param.value

                minor_radius_param = param_map.get((node.node_id, 'minor_radius'))
                if minor_radius_param:
                    updated_kwargs['minor_radius'] = minor_radius_param.value

                # If we have updated parameters, rebuild the primitive
                if updated_kwargs:
                    return prim_class(**updated_kwargs)
                else:
                    return node.child_sdf

            # Rebuild children
            if node.op_type == OpType.UNION:
                from jaxcad.boolean import smooth_min
                left_fn = rebuild_sdf(node.children[0])
                right_fn = rebuild_sdf(node.children[1])
                smoothness = node.params.get("smoothness", 0.1)

                def union_fn(p):
                    return smooth_min(left_fn(p), right_fn(p), smoothness)
                return union_fn

            elif node.op_type == OpType.INTERSECTION:
                from jaxcad.boolean import smooth_max
                left_fn = rebuild_sdf(node.children[0])
                right_fn = rebuild_sdf(node.children[1])
                smoothness = node.params.get("smoothness", 0.1)

                def intersection_fn(p):
                    return smooth_max(left_fn(p), right_fn(p), smoothness)
                return intersection_fn

            elif node.op_type == OpType.DIFFERENCE:
                from jaxcad.boolean import smooth_max
                left_fn = rebuild_sdf(node.children[0])
                right_fn = rebuild_sdf(node.children[1])
                smoothness = node.params.get("smoothness", 0.1)

                def difference_fn(p):
                    return smooth_max(left_fn(p), -right_fn(p), smoothness)
                return difference_fn

            elif node.op_type == OpType.TRANSLATE:
                child_fn = rebuild_sdf(node.children[0])
                # Get current value from constraint
                param = param_map.get(node.node_id)
                offset = param.value if param else node.params.get("offset")

                def translate_fn(p):
                    return translate_eval(child_fn, p, offset)
                return translate_fn

            elif node.op_type == OpType.ROTATE:
                child_fn = rebuild_sdf(node.children[0])
                param = param_map.get(node.node_id)
                angle = param.value if param else node.params.get("angle")

                def rotate_fn(p):
                    return rotate_z_eval(child_fn, p, angle)
                return rotate_fn

            elif node.op_type == OpType.SCALE:
                child_fn = rebuild_sdf(node.children[0])
                param = param_map.get(node.node_id)
                scale = param.value if param else node.params.get("scale")

                def scale_fn(p):
                    return scale_eval(child_fn, p, scale)
                return scale_fn

            elif node.op_type == OpType.TWIST:
                child_fn = rebuild_sdf(node.children[0])
                param = param_map.get(node.node_id)
                strength = param.value if param else node.params.get("strength")

                def twist_fn(p):
                    return twist_z_eval(child_fn, p, strength)
                return twist_fn

            elif node.op_type == OpType.TAPER:
                child_fn = rebuild_sdf(node.children[0])
                param = param_map.get(node.node_id)
                strength = param.value if param else node.params.get("strength")

                def taper_fn(p):
                    return taper_z_eval(child_fn, p, strength)
                return taper_fn

            else:
                # Fallback - just use original SDF
                return lambda p: node.child_sdf(p) if node.child_sdf else 0.0

        # Rebuild and evaluate
        if graph.nodes:
            child_sdf = rebuild_sdf(graph.nodes[-1])
            return child_sdf(query_point)
        else:
            return jnp.array(0.0)

    return eval_fn


def optimize_parameters(
    sdf: SDF,
    target_points: Array,
    target_values: Array,
    constraint_system: ConstraintSystem,
    num_iterations: int = 100,
    learning_rate: float = 0.01
) -> tuple[Array, list[float]]:
    """Optimize free parameters to match target SDF values at target points.

    Args:
        sdf: The SDF expression
        target_points: Points where we want specific SDF values (N, 3)
        target_values: Desired SDF values at those points (N,)
        constraint_system: Constraint system with free/fixed parameters
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for gradient descent

    Returns:
        Tuple of (optimized_params, loss_history)
    """
    # Compile to parametric form
    eval_fn = compile_parametric(sdf, constraint_system)

    # Define loss function
    def loss_fn(params_vector: Array) -> Array:
        """Compute MSE between predicted and target SDF values."""
        predictions = jax.vmap(
            lambda p: eval_fn(params_vector, p)
        )(target_points)

        return jnp.mean((predictions - target_values) ** 2)

    # Gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))

    # Initialize with current free parameters
    params = constraint_system.to_vector()
    loss_history = []

    # Gradient descent
    for i in range(num_iterations):
        loss = loss_fn(params)
        loss_history.append(float(loss))

        if i % 10 == 0:
            print(f"Iteration {i}: loss = {loss:.6f}")

        grad = grad_fn(params)
        params = params - learning_rate * grad

    # Update constraint system with optimized values
    constraint_system.from_vector(params)

    return params, loss_history
