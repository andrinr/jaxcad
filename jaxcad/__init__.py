"""JAX-based differentiable CAD system."""

from jaxcad.boolean import merge, smooth_vertices, subdivide_faces
from jaxcad.core import Solid
from jaxcad.modifications import compute_vertex_normals, taper, thicken, twist
from jaxcad.operations import array_circular, array_linear, extrude, loft, revolve, sweep
from jaxcad.primitives import box, cylinder, sphere
from jaxcad.sketch import Profile2D, circle, offset_profile, polygon, rectangle, regular_polygon
from jaxcad.transforms import rotate, scale, transform, translate

# Visualization is optional
try:
    from jaxcad.viz import plot_solid, plot_solids  # noqa: F401

    __all_viz__ = ["plot_solid", "plot_solids"]
except ImportError:
    __all_viz__ = []

__version__ = "0.1.0"
__all__ = [
    # Core
    "Solid",
    "Profile2D",
    # Primitives
    "box",
    "sphere",
    "cylinder",
    # Transforms
    "translate",
    "rotate",
    "scale",
    "transform",
    # Sketch
    "rectangle",
    "circle",
    "polygon",
    "regular_polygon",
    "offset_profile",
    # Operations
    "extrude",
    "revolve",
    "loft",
    "sweep",
    "array_linear",
    "array_circular",
    # Boolean
    "merge",
    "smooth_vertices",
    "subdivide_faces",
    # Modifications
    "twist",
    "taper",
    "thicken",
    "compute_vertex_normals",
] + __all_viz__
