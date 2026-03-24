"""Extrude a rectangle to create an oriented box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxcad.geometry.parameters import Scalar, Vector, as_parameter
from jaxcad.geometry.primitives import Rectangle

if TYPE_CHECKING:
    from jaxcad.sdf.primitives.box import Box


def extrude(rectangle: Rectangle, depth: float | Scalar) -> Box:
    """Extrude a rectangle to create an oriented box.

    The box is centered on the rectangle's plane and extends depth/2 in each
    direction along the rectangle's normal.

    Args:
        rectangle: Rectangle to extrude
        depth: Extrusion depth (total, not half-depth)

    Returns:
        Box SDF with shared parameter references

    Example:
        ```python
        rect = Rectangle(center=[0, 0, 0], width=2.0, height=1.0)
        box = extrude(rect, depth=3.0)
        ```
    """
    from jaxcad.sdf.primitives.box import Box

    depth = as_parameter(depth)

    # Box size: [width/2, height/2, depth/2]
    # We need to create a Vector that combines width, height, depth
    # Since rectangle params might be free, we need to handle this carefully

    # For now, create a new box centered at rectangle center
    # The size is [width/2, height/2, depth/2]
    # Note: This is a simplified version. Full implementation would need
    # proper coordinate frame handling for arbitrary rectangle orientations

    # Create size vector
    size_xyz = jnp.array([rectangle.width.value / 2, rectangle.height.value / 2, depth.value / 2])

    size = Vector(value=size_xyz, free=False, name=f"{rectangle.center.name}_box_size")

    # Create box
    box = Box(size=size)

    # Store reference to source geometry for potential future use
    box._source_geometry = rectangle
    box._construction_method = "extrude"

    return box
