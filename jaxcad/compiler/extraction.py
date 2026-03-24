"""Parameter extraction from SDF trees."""

from typing import Any, Dict

from jaxcad.sdf import SDF


def extract_parameters(sdf: SDF) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract free and fixed parameters from an SDF tree.

    Args:
        sdf: The SDF to extract parameters from

    Returns:
        Tuple of (free_params, fixed_params) where each is a dict mapping
        parameter paths to Parameter objects.
        Parameter paths are in format: "node_id.param_name" (e.g., "sphere_0.radius")
    """
    from jaxcad.geometry.parameters import Parameter
    from jaxcad.sdf.transforms.base import Transform
    from jaxcad.sdf.boolean.base import BooleanOp

    free_params = {}
    fixed_params = {}
    node_counter = {'count': 0}

    def walk(obj: SDF) -> None:
        """Recursively walk SDF tree and collect parameters."""
        # Generate node ID for this SDF
        class_name = obj.__class__.__name__.lower()
        node_id = f"{class_name}_{node_counter['count']}"
        node_counter['count'] += 1

        # Extract parameters from this node
        if hasattr(obj, 'params'):
            for param_name, param in obj.params.items():
                param_path = f"{node_id}.{param_name}"
                if param.free:
                    free_params[param_path] = param
                else:
                    fixed_params[param_path] = param

        # Recursively walk children based on SDF type
        if isinstance(obj, Transform):
            walk(obj.sdf)
        elif isinstance(obj, BooleanOp):
            for child in obj.sdfs:
                walk(child)
        # Primitives have no children

    walk(sdf)
    return free_params, fixed_params
