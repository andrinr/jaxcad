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

        # Extract parameter values from self.params dictionary
        params = {}
        from jaxcad.parameters import Vector, Parameter
        for param_name, param_value in self.params.items():
            # Extract value from Parameter objects, pass through other values (like axis)
            if isinstance(param_value, Parameter):
                if isinstance(param_value, Vector):
                    params[param_name] = param_value.xyz
                else:
                    params[param_name] = param_value.value
            else:
                # Non-parameter metadata (like axis in Rotate)
                params[param_name] = param_value

        return graph.add_node(self.__class__, children=[child], params=params)
