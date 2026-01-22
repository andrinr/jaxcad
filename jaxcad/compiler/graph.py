"""SDF expression graph representation and compilation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class OpType(Enum):
    """Operation types in the SDF graph."""
    PRIMITIVE = "primitive"
    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"
    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"
    TWIST = "twist"
    BEND = "bend"
    TAPER = "taper"
    MIRROR = "mirror"
    REPEAT_INFINITE = "repeat_infinite"
    REPEAT_FINITE = "repeat_finite"


@dataclass
class GraphNode:
    """Node in the SDF computation graph."""
    op_type: OpType
    sdf_fn: Optional[Callable] = None  # For primitives
    children: List['GraphNode'] = None  # For composite operations
    params: Dict[str, Any] = None  # Operation parameters
    node_id: Optional[int] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.params is None:
            self.params = {}


class SDFGraph:
    """Expression graph for SDF operations.

    Captures the structure of SDF operations for analysis and optimization
    before JIT compilation.
    """

    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.node_counter = 0

    def add_node(self, op_type: OpType, sdf_fn=None, children=None, params=None) -> GraphNode:
        """Add a node to the graph."""
        node = GraphNode(
            op_type=op_type,
            sdf_fn=sdf_fn,
            children=children or [],
            params=params or {},
            node_id=self.node_counter
        )
        self.nodes.append(node)
        self.node_counter += 1
        return node

    def visualize(self, node: Optional[GraphNode] = None, indent: int = 0) -> str:
        """Generate a text visualization of the graph."""
        if node is None:
            if not self.nodes:
                return "Empty graph"
            node = self.nodes[-1]  # Root is typically the last node

        lines = []
        prefix = "  " * indent

        # Node description
        if node.op_type == OpType.PRIMITIVE:
            lines.append(f"{prefix}{node.op_type.value}: {node.sdf_fn.__class__.__name__}")
        else:
            param_str = ", ".join(f"{k}={v}" for k, v in node.params.items())
            lines.append(f"{prefix}{node.op_type.value}({param_str})")

        # Children
        for child in node.children:
            lines.append(self.visualize(child, indent + 1))

        return "\n".join(lines)

    def count_operations(self) -> Dict[OpType, int]:
        """Count operations by type."""
        counts = {}
        for node in self.nodes:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return counts


def extract_graph(sdf: SDF) -> SDFGraph:
    """Extract computation graph from an SDF object.

    Walks the SDF tree and builds an explicit graph representation.
    """
    graph = SDFGraph()

    def walk(obj: SDF) -> GraphNode:
        """Recursively walk SDF tree."""
        # Check the type and extract structure
        class_name = obj.__class__.__name__

        # Boolean operations
        if class_name == "Union":
            left = walk(obj.sdf1)
            right = walk(obj.sdf2)
            return graph.add_node(
                OpType.UNION,
                children=[left, right],
                params={"smoothness": obj.smoothness}
            )
        elif class_name == "Intersection":
            left = walk(obj.sdf1)
            right = walk(obj.sdf2)
            return graph.add_node(
                OpType.INTERSECTION,
                children=[left, right],
                params={"smoothness": obj.smoothness}
            )
        elif class_name == "Difference":
            left = walk(obj.sdf1)
            right = walk(obj.sdf2)
            return graph.add_node(
                OpType.DIFFERENCE,
                children=[left, right],
                params={"smoothness": obj.smoothness}
            )

        # Transforms
        elif class_name == "Translate":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.TRANSLATE,
                children=[child],
                params={"offset": obj.offset}
            )
        elif class_name == "Rotate":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.ROTATE,
                children=[child],
                params={"angle": obj.angle, "matrix": obj.rotation_matrix}
            )
        elif class_name == "Scale":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.SCALE,
                children=[child],
                params={"scale": obj.scale, "is_uniform": obj.is_uniform}
            )
        elif class_name == "Twist":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.TWIST,
                children=[child],
                params={"axis": obj.axis, "strength": obj.strength}
            )
        elif class_name == "Bend":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.BEND,
                children=[child],
                params={"axis": obj.axis, "strength": obj.strength}
            )
        elif class_name == "Taper":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.TAPER,
                children=[child],
                params={"axis": obj.axis, "strength": obj.strength}
            )
        elif class_name == "Mirror":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.MIRROR,
                children=[child],
                params={"axis": obj.axis, "offset": obj.offset}
            )
        elif class_name == "RepeatInfinite":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.REPEAT_INFINITE,
                children=[child],
                params={"spacing": obj.spacing}
            )
        elif class_name == "RepeatFinite":
            child = walk(obj.sdf)
            return graph.add_node(
                OpType.REPEAT_FINITE,
                children=[child],
                params={"spacing": obj.spacing, "count": obj.count}
            )

        # Primitives (leaf nodes)
        else:
            return graph.add_node(OpType.PRIMITIVE, sdf_fn=obj)

    walk(sdf)
    return graph


def compile_sdf(sdf: SDF, optimize: bool = True, jit: bool = True) -> Callable[[Array], Array]:
    """Compile an SDF with optional graph optimization and JIT.

    This is the main compilation entry point that:
    1. Extracts the computation graph
    2. Optionally optimizes the graph
    3. Generates optimized evaluation code
    4. Optionally JIT compiles with JAX

    Args:
        sdf: The SDF to compile
        optimize: Whether to apply graph optimizations
        jit: Whether to apply JAX JIT compilation

    Returns:
        Compiled evaluation function
    """
    # Extract graph
    graph = extract_graph(sdf)

    # Optimize if requested
    if optimize:
        from jaxcad.compiler.optimize import optimize_graph
        graph = optimize_graph(graph)

    # Generate evaluation function
    eval_fn = sdf  # For now, just use the original SDF

    # Apply JAX JIT if requested
    if jit:
        eval_fn = jax.jit(eval_fn)

    return eval_fn
