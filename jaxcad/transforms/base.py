"""Base class for transform SDFs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxcad.sdf import SDF

if TYPE_CHECKING:
    from jaxcad.compiler.graph import SDFGraph, GraphNode


class Transform(SDF):
    """Base class for transform SDFs (unary operations in the computation graph).

    Transforms modify a child SDF - translate, rotate, scale, twist, etc.
    They have one child in the computation graph.

    Subclasses must implement:
    - @staticmethod def sdf(child_sdf, p: Array, **params) -> Array
    - __call__(self, p: Array) -> Array
    - to_functional(self) -> Callable

    Subclasses should store:
    - self.sdf: The child SDF being transformed
    - Parameters as *_param attributes
    """

    def to_graph_node(self, graph: SDFGraph, walk_fn) -> GraphNode:
        """Add this transform to the computation graph.

        Transforms have one child and store their parameters.
        """
        child = walk_fn(self.sdf)

        # Extract parameters from *_param attributes
        params = {}
        for attr_name in dir(self):
            if attr_name.endswith('_param') and not attr_name.startswith('_'):
                param = getattr(self, attr_name)
                param_name = attr_name.replace('_param', '')

                # Extract value, handling Vector vs Scalar
                from jaxcad.parameters import Vector
                if isinstance(param, Vector):
                    params[param_name] = param.xyz
                else:
                    params[param_name] = param.value

        # Add any non-parameter attributes (like is_uniform for Scale)
        for attr_name in ['is_uniform', 'axis']:
            if hasattr(self, attr_name):
                params[attr_name] = getattr(self, attr_name)

        return graph.add_node(self.__class__, children=[child], params=params)
