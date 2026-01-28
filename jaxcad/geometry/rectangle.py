"""Rectangle geometric entity.

A Rectangle is a parametric geometric entity defined by a center point,
dimensions (width and height), and an orientation (normal vector).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector, Scalar


@dataclass
class Rectangle:
    """Parametric rectangle in 3D space.

    A Rectangle is a first-class geometric entity that can be used to:
    - Define planar constraints and relationships
    - Construct oriented primitives (e.g., Box.extrude(rectangle, depth))
    - Serve as a surface for operations (e.g., Array pattern on surface)
    - Render as wireframe/surface in 3D viewers

    Args:
        center: Center point (Vector or 3D array)
        width: Width along the local X axis (Scalar or float)
        height: Height along the local Y axis (Scalar or float)
        normal: Normal vector defining the plane orientation (Vector or 3D array, default: [0, 0, 1])
        u_axis: Optional explicit U (width) axis. If not provided, computed from normal.

    The rectangle lies in a plane defined by the normal vector, with width
    and height defining the extent in the local coordinate frame.

    Example:
        # Rectangle in XY plane
        rect = Rectangle(center=[0, 0, 0], width=2.0, height=1.0)

        # Tilted rectangle with free parameters
        center = Vector([0, 0, 0], free=True, name='center')
        normal = Vector([1, 1, 1, 0], free=False)  # w=0 for direction
        rect = Rectangle(center=center, width=2.0, height=1.0, normal=normal)

        # Sample points on the rectangle
        corner = rect.corner(0)  # Get first corner
        point = rect.sample(0.5, 0.5)  # Center point
    """

    center: Vector
    width: Scalar
    height: Scalar
    normal: Vector = None
    u_axis: Vector = None

    def __post_init__(self):
        """Convert raw values to parameter types and compute local frame."""
        from jaxcad.geometry.parameters import as_parameter

        self.center = as_parameter(self.center)
        self.width = as_parameter(self.width)
        self.height = as_parameter(self.height)

        # Default normal is +Z
        if self.normal is None:
            self.normal = as_parameter([0.0, 0.0, 1.0, 0.0])
        else:
            self.normal = as_parameter(self.normal)

        # Compute local coordinate frame if u_axis not provided
        if self.u_axis is None:
            self._compute_local_frame()
        else:
            self.u_axis = as_parameter(self.u_axis)

    def _compute_local_frame(self):
        """Compute orthonormal local frame (U, V, N) from normal vector.

        U: width direction
        V: height direction
        N: normal direction
        """
        normal = self.normal.xyz
        normal = normal / jnp.linalg.norm(normal)

        # Find a vector not parallel to normal
        up = jnp.array([0.0, 0.0, 1.0])
        if jnp.abs(jnp.dot(normal, up)) > 0.99:
            up = jnp.array([1.0, 0.0, 0.0])

        # Compute U (width) axis
        u = jnp.cross(up, normal)
        u = u / jnp.linalg.norm(u)

        # Store as Vector parameter
        from jaxcad.geometry.parameters import as_parameter
        self.u_axis = as_parameter(jnp.append(u, 0.0))  # Direction vector (w=0)

    def v_axis(self) -> Array:
        """Compute V (height) axis from normal and U axis.

        Returns:
            3D direction vector for height axis
        """
        normal = self.normal.xyz / jnp.linalg.norm(self.normal.xyz)
        u = self.u_axis.xyz
        v = jnp.cross(normal, u)
        return v / jnp.linalg.norm(v)

    def sample(self, u: Union[float, Array], v: Union[float, Array]) -> Array:
        """Sample point on the rectangle at parameters (u, v).

        Args:
            u: Parameter along width in [0, 1] (0.5 = center)
            v: Parameter along height in [0, 1] (0.5 = center)

        Returns:
            3D point on the rectangle
        """
        u = jnp.asarray(u)
        v = jnp.asarray(v)

        # Convert from [0, 1] to [-0.5, 0.5]
        u_offset = (u - 0.5) * self.width.value
        v_offset = (v - 0.5) * self.height.value

        u_vec = self.u_axis.xyz
        v_vec = self.v_axis()

        return self.center.xyz + u_offset * u_vec + v_offset * v_vec

    def corner(self, index: int) -> Array:
        """Get one of the four corners of the rectangle.

        Args:
            index: Corner index in [0, 3]
                   0: (-w/2, -h/2)
                   1: (+w/2, -h/2)
                   2: (+w/2, +h/2)
                   3: (-w/2, +h/2)

        Returns:
            3D corner position
        """
        corners_uv = [
            (0.0, 0.0),  # Bottom-left
            (1.0, 0.0),  # Bottom-right
            (1.0, 1.0),  # Top-right
            (0.0, 1.0),  # Top-left
        ]
        u, v = corners_uv[index % 4]
        return self.sample(u, v)

    def corners(self) -> Array:
        """Get all four corners of the rectangle.

        Returns:
            Array of shape (4, 3) with corner positions
        """
        return jnp.stack([self.corner(i) for i in range(4)])

    def area(self) -> float:
        """Get the area of the rectangle.

        Returns:
            Area (width * height)
        """
        return self.width.value * self.height.value

    def perimeter(self) -> float:
        """Get the perimeter of the rectangle.

        Returns:
            Perimeter (2 * (width + height))
        """
        return 2 * (self.width.value + self.height.value)

    def contains_point(self, p: Array, tolerance: float = 1e-6) -> bool:
        """Check if a point lies within the rectangle (in its plane).

        Args:
            p: Query point [x, y, z]
            tolerance: Distance tolerance for plane and bounds checking

        Returns:
            True if point is within rectangle bounds
        """
        p = jnp.asarray(p)

        # Project point onto rectangle's local frame
        p_rel = p - self.center.xyz
        u_vec = self.u_axis.xyz
        v_vec = self.v_axis()
        normal = self.normal.xyz / jnp.linalg.norm(self.normal.xyz)

        # Check if point is in plane
        dist_to_plane = jnp.abs(jnp.dot(p_rel, normal))
        if dist_to_plane > tolerance:
            return False

        # Check if within bounds
        u_coord = jnp.dot(p_rel, u_vec)
        v_coord = jnp.dot(p_rel, v_vec)

        u_in_bounds = jnp.abs(u_coord) <= (self.width.value / 2 + tolerance)
        v_in_bounds = jnp.abs(v_coord) <= (self.height.value / 2 + tolerance)

        return u_in_bounds and v_in_bounds

    def __repr__(self) -> str:
        return f"Rectangle(center={self.center.xyz}, width={self.width.value}, height={self.height.value})"
