"""SDF expression graph representation and compilation."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import jax
from jax import Array

from jaxcad.sdf import SDF


@dataclass
class GraphNode:
    """Node in the SDF computation graph."""
    op_class: Type[SDF] | None  # The SDF class (Translate, Sphere, Union, etc.)
    child_sdf: Optional[Callable] = None  # For primitives: the instance
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

    def add_node(self, op_class: Type[SDF] | None, child_sdf=None, children=None, params=None) -> GraphNode:
        """Add a node to the graph."""
        node = GraphNode(
            op_class=op_class,
            child_sdf=child_sdf,
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
        if node.child_sdf is not None:
            # Primitive node
            lines.append(f"{prefix}{node.op_class.__name__}: {node.child_sdf.__class__.__name__}")
        else:
            # Transform/Boolean node
            param_str = ", ".join(f"{k}={v}" for k, v in node.params.items())
            op_name = node.op_class.__name__ if node.op_class else "Unknown"
            lines.append(f"{prefix}{op_name}({param_str})")

        # Children
        for child in node.children:
            lines.append(self.visualize(child, indent + 1))

        return "\n".join(lines)

    def count_operations(self) -> Dict[Type[SDF], int]:
        """Count operations by type."""
        counts = {}
        for node in self.nodes:
            if node.op_class:
                counts[node.op_class] = counts.get(node.op_class, 0) + 1
        return counts


def extract_graph(sdf: SDF) -> SDFGraph:
    """Extract computation graph from an SDF object.

    Walks the SDF tree and builds an explicit graph representation.
    """
    graph = SDFGraph()

    def walk(obj: SDF) -> GraphNode:
        """Recursively walk SDF tree.

        Each SDF knows how to serialize itself via to_graph_node().
        """
        return obj.to_graph_node(graph, walk)

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
