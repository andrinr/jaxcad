"""Graph optimization passes for SDF expressions."""

from typing import List

from jaxcad.compiler.graph import SDFGraph, GraphNode, OpType


def optimize_graph(graph: SDFGraph) -> SDFGraph:
    """Apply optimization passes to an SDF graph.

    Optimizations include:
    - Constant folding
    - Transform fusion (combining sequential transforms)
    - Dead code elimination
    - Common subexpression elimination

    Args:
        graph: Input computation graph

    Returns:
        Optimized graph
    """
    # Apply optimization passes in sequence
    graph = fuse_transforms(graph)
    graph = eliminate_identity_transforms(graph)
    # graph = common_subexpression_elimination(graph)  # TODO

    return graph


def fuse_transforms(graph: SDFGraph) -> SDFGraph:
    """Fuse sequential transforms of the same type.

    Examples:
    - translate(translate(sdf, a), b) -> translate(sdf, a + b)
    - scale(scale(sdf, a), b) -> scale(sdf, a * b)
    - rotate(rotate(sdf, axis, a), axis, b) -> rotate(sdf, axis, a + b)
    """
    optimized = SDFGraph()

    def fuse_node(node: GraphNode) -> GraphNode:
        """Recursively fuse transforms in a node."""
        if not node.children:
            # Leaf node - copy as is
            return optimized.add_node(
                node.op_type,
                sdf_fn=node.sdf_fn,
                params=node.params
            )

        # Fuse children first
        fused_children = [fuse_node(child) for child in node.children]

        # Check for fusion opportunities
        if node.op_type == OpType.TRANSLATE and len(fused_children) == 1:
            child = fused_children[0]
            if child.op_type == OpType.TRANSLATE:
                # Fuse two translations: offset1 + offset2
                import jax.numpy as jnp
                new_offset = node.params["offset"] + child.params["offset"]
                return optimized.add_node(
                    OpType.TRANSLATE,
                    children=child.children,
                    params={"offset": new_offset}
                )

        elif node.op_type == OpType.SCALE and len(fused_children) == 1:
            child = fused_children[0]
            if child.op_type == OpType.SCALE:
                # Fuse two scales: scale1 * scale2
                new_scale = node.params["scale"] * child.params["scale"]
                is_uniform = node.params.get("is_uniform", True) and child.params.get("is_uniform", True)
                return optimized.add_node(
                    OpType.SCALE,
                    children=child.children,
                    params={"scale": new_scale, "is_uniform": is_uniform}
                )

        # No fusion - return node with fused children
        return optimized.add_node(
            node.op_type,
            sdf_fn=node.sdf_fn,
            children=fused_children,
            params=node.params
        )

    if graph.nodes:
        root = graph.nodes[-1]
        fuse_node(root)

    return optimized


def eliminate_identity_transforms(graph: SDFGraph) -> SDFGraph:
    """Remove identity transforms (e.g., translate by 0, scale by 1).

    These do nothing but add overhead.
    """
    optimized = SDFGraph()

    def is_identity(node: GraphNode) -> bool:
        """Check if a transform is an identity operation."""
        import jax.numpy as jnp

        if node.op_type == OpType.TRANSLATE:
            offset = node.params.get("offset")
            if offset is not None and jnp.allclose(offset, 0.0):
                return True

        elif node.op_type == OpType.SCALE:
            scale = node.params.get("scale")
            if scale is not None:
                if isinstance(scale, (int, float)) and abs(scale - 1.0) < 1e-9:
                    return True
                elif hasattr(scale, 'shape') and jnp.allclose(scale, 1.0):
                    return True

        elif node.op_type == OpType.ROTATE:
            angle = node.params.get("angle")
            if angle is not None and abs(angle) < 1e-9:
                return True

        return False

    def eliminate_node(node: GraphNode) -> GraphNode:
        """Recursively eliminate identity transforms."""
        if not node.children:
            # Leaf node - copy as is
            return optimized.add_node(
                node.op_type,
                sdf_fn=node.sdf_fn,
                params=node.params
            )

        # Process children first
        processed_children = [eliminate_node(child) for child in node.children]

        # Check if this is an identity transform
        if is_identity(node) and len(processed_children) == 1:
            # Skip this node, return the child directly
            return processed_children[0]

        # Not an identity - keep the node
        return optimized.add_node(
            node.op_type,
            sdf_fn=node.sdf_fn,
            children=processed_children,
            params=node.params
        )

    if graph.nodes:
        root = graph.nodes[-1]
        eliminate_node(root)

    return optimized


def estimate_complexity(graph: SDFGraph) -> dict:
    """Estimate computational complexity of the graph.

    Returns metrics like:
    - Total node count
    - Operation counts by type
    - Estimated FLOPs
    - Graph depth
    """
    def count_depth(node: GraphNode) -> int:
        """Compute maximum depth from this node."""
        if not node.children:
            return 1
        return 1 + max(count_depth(child) for child in node.children)

    counts = graph.count_operations()

    if not graph.nodes:
        depth = 0
    else:
        depth = count_depth(graph.nodes[-1])

    # Rough FLOP estimates per operation type
    flop_estimates = {
        OpType.PRIMITIVE: 10,      # Varies by primitive
        OpType.UNION: 5,
        OpType.INTERSECTION: 5,
        OpType.DIFFERENCE: 5,
        OpType.TRANSLATE: 3,
        OpType.ROTATE: 15,         # Matrix multiply
        OpType.SCALE: 3,
        OpType.TWIST: 20,          # Trig functions
        OpType.BEND: 25,
        OpType.TAPER: 5,
        OpType.MIRROR: 2,
        OpType.REPEAT_INFINITE: 5,
        OpType.REPEAT_FINITE: 10,
    }

    total_flops = sum(counts.get(op_type, 0) * flops
                     for op_type, flops in flop_estimates.items())

    return {
        "total_nodes": len(graph.nodes),
        "operations": counts,
        "depth": depth,
        "estimated_flops": total_flops
    }
