"""Parameter extraction from Fluent trees."""

from jaxcad.fluent import Fluent


def extract_parameters(root: Fluent) -> tuple[dict, dict, dict]:
    """Extract parameters from a Fluent tree.

    Args:
        root: The root of the Fluent tree to extract parameters from.

    Returns:
        free_params (dict[name, Array]): Name-keyed, deduplicated plain arrays for free params.
        fixed_params (dict[path, Array]): Path-keyed plain arrays for fixed params.
        metadata (dict[name, Parameter]): Name-keyed Parameter objects for free params
            (carries constraints, bounds, type info).

    Raises:
        ValueError: If a free parameter has no name, or two free parameters share a name.
    """
    free_params = {}  # name → Array
    fixed_params = {}  # path → Array
    metadata = {}  # name → Parameter
    seen_free_ids = {}  # id(param) → name (for dedup)
    seen_free_names = set()
    node_counter = {"count": 0}

    def walk(obj: Fluent) -> None:
        class_name = obj.__class__.__name__.lower()
        node_id = f"{class_name}_{node_counter['count']}"
        node_counter["count"] += 1

        if hasattr(obj, "params"):
            for attr_name, param in obj.params.items():
                path = f"{node_id}.{attr_name}"
                if param.free:
                    if id(param) in seen_free_ids:
                        continue  # deduplicate by identity
                    if param.name is None:
                        raise ValueError(f"Free parameter at '{path}' has no name.")
                    if param.name in seen_free_names:
                        raise ValueError(f"Two free parameters share name '{param.name}'.")
                    seen_free_ids[id(param)] = param.name
                    seen_free_names.add(param.name)
                    free_params[param.name] = param.value
                    metadata[param.name] = param
                else:
                    fixed_params[path] = param.value

        for child in obj.children():
            walk(child)

    walk(root)
    return free_params, fixed_params, metadata
