"""Base class for boolean operation SDFs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxcad.sdf import SDF

if TYPE_CHECKING:
    from jaxcad.compiler.graph import SDFGraph, GraphNode


class BooleanOp(SDF):
    """Base class for boolean operation SDFs (binary operations in the computation graph).

    Boolean operations combine two SDFs - union, intersection, difference, etc.
    They have two children in the computation graph.

    Subclasses must implement:
    - @staticmethod def sdf(child_sdf1, child_sdf2, p: Array, smoothness: float) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable

    Subclasses should store:
    - self.sdf1: The first child SDF
    - self.sdf2: The second child SDF
    - self.smoothness: The smoothness parameter
    """

    def to_graph_node(self, graph: SDFGraph, walk_fn) -> GraphNode:
        """Add this boolean operation to the computation graph.

        Boolean operations have two children and store smoothness parameter.
        """
        left = walk_fn(self.sdf1)
        right = walk_fn(self.sdf2)

        params = {}
        if hasattr(self, 'smoothness'):
            params['smoothness'] = self.smoothness

        return graph.add_node(self.__class__, children=[left, right], params=params)
