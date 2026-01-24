"""Base class for primitive SDFs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxcad.sdf import SDF

if TYPE_CHECKING:
    from jaxcad.compiler.graph import SDFGraph, GraphNode


class Primitive(SDF):
    """Base class for primitive SDFs (leaf nodes in the computation graph).

    Primitives are the basic building blocks - spheres, boxes, cylinders, etc.
    They have no children in the computation graph.

    Subclasses must implement:
    - @staticmethod def sdf(p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable
    """

    def to_graph_node(self, graph: SDFGraph, walk_fn) -> GraphNode:
        """Add this primitive to the computation graph.

        Primitives are leaf nodes with no children.
        """
        return graph.add_node(self.__class__, child_sdf=self)
