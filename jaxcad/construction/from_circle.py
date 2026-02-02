"""Create a cylinder from a circle."""

from __future__ import annotations

from typing import Union

from jaxcad.geometry.parameters import Scalar, as_parameter
from jaxcad.geometry.primitives import Circle


def from_circle(circle: Circle, height: Union[float, Scalar]) -> 'Cylinder':
    """Create a cylinder from a circle.

    The cylinder's base is the circle, and it extends along the circle's normal
    by the specified height.

    Args:
        circle: Circle defining the cylinder's cross-section
        height: Cylinder height

    Returns:
        Cylinder SDF with shared parameter references

    Example:
        center = Vector([0, 0, 0], free=True, name='center')
        radius = Scalar(1.0, free=True, name='radius')
        circle = Circle(center=center, radius=radius)
        cylinder = from_circle(circle, height=5.0)
    """
    from jaxcad.sdf.primitives.cylinder import Cylinder

    height = as_parameter(height)

    # Cylinder parameters: radius, height
    # The cylinder is axis-aligned in its local frame
    # For a general circle, we'd need to apply a transform
    # For now, assume the circle is in the XY plane

    cylinder = Cylinder(radius=circle.radius, height=height)

    # Store reference to source geometry
    cylinder._source_geometry = circle
    cylinder._construction_method = 'from_circle'

    return cylinder
