"""Differentiable 2D sketch operations for profile creation."""

from typing import NamedTuple

import jax.numpy as jnp


class Profile2D(NamedTuple):
    """Represents a 2D profile/sketch.

    Attributes:
        points: Array of shape (N, 2) containing 2D points defining the profile
        closed: Boolean indicating if the profile forms a closed loop
    """

    points: jnp.ndarray  # (N, 2)
    closed: bool = True

    def __repr__(self):
        return f"Profile2D(points={self.points.shape}, closed={self.closed})"


def rectangle(center: jnp.ndarray, width: float, height: float) -> Profile2D:
    """Create a rectangular 2D profile.

    Args:
        center: Array of shape (2,) specifying rectangle center [x, y]
        width: Rectangle width
        height: Rectangle height

    Returns:
        Profile2D representing a rectangle

    Example:
        >>> center = jnp.array([0., 0.])
        >>> profile = rectangle(center, width=2.0, height=1.0)
    """
    half_w = width / 2.0
    half_h = height / 2.0

    points = (
        jnp.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ]
        )
        + center
    )

    return Profile2D(points=points, closed=True)


def circle(center: jnp.ndarray, radius: float, resolution: int = 32) -> Profile2D:
    """Create a circular 2D profile.

    Args:
        center: Array of shape (2,) specifying circle center [x, y]
        radius: Circle radius
        resolution: Number of points to approximate the circle

    Returns:
        Profile2D representing a circle

    Example:
        >>> center = jnp.array([0., 0.])
        >>> profile = circle(center, radius=1.0)
    """
    theta = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    x = radius * jnp.cos(theta) + center[0]
    y = radius * jnp.sin(theta) + center[1]

    points = jnp.stack([x, y], axis=1)

    return Profile2D(points=points, closed=True)


def polygon(points: jnp.ndarray, closed: bool = True) -> Profile2D:
    """Create a polygonal 2D profile from arbitrary points.

    Args:
        points: Array of shape (N, 2) containing 2D points
        closed: Whether the polygon is closed

    Returns:
        Profile2D representing the polygon

    Example:
        >>> points = jnp.array([[0., 0.], [1., 0.], [0.5, 1.]])
        >>> profile = polygon(points)  # Triangle
    """
    return Profile2D(points=points, closed=closed)


def regular_polygon(center: jnp.ndarray, radius: float, n_sides: int) -> Profile2D:
    """Create a regular polygon.

    Args:
        center: Array of shape (2,) specifying polygon center [x, y]
        radius: Distance from center to vertices
        n_sides: Number of sides

    Returns:
        Profile2D representing a regular polygon

    Example:
        >>> center = jnp.array([0., 0.])
        >>> profile = regular_polygon(center, radius=1.0, n_sides=6)  # Hexagon
    """
    theta = jnp.linspace(0, 2 * jnp.pi, n_sides, endpoint=False)
    x = radius * jnp.cos(theta) + center[0]
    y = radius * jnp.sin(theta) + center[1]

    points = jnp.stack([x, y], axis=1)

    return Profile2D(points=points, closed=True)


def offset_profile(profile: Profile2D, distance: float, resolution: int = 16) -> Profile2D:  # noqa: ARG001
    """Offset a 2D profile by a given distance (inward/outward).

    Uses simple per-vertex normal offsetting. For complex shapes, this is approximate.

    Args:
        profile: Input profile to offset
        distance: Offset distance (positive = outward, negative = inward)
        resolution: Number of points for smoothing corners

    Returns:
        Offset profile

    Example:
        >>> rect = rectangle(jnp.zeros(2), 2.0, 2.0)
        >>> expanded = offset_profile(rect, 0.5)
    """
    points = profile.points

    if not profile.closed:
        # For open profiles, just offset perpendicular
        tangents = jnp.diff(points, axis=0, prepend=points[:1])
        normals = jnp.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        normals = normals / (jnp.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        offset_points = points + distance * normals
        return Profile2D(points=offset_points, closed=False)

    # For closed profiles, compute normals at each vertex
    # Get vectors to next and previous points
    next_points = jnp.roll(points, -1, axis=0)
    prev_points = jnp.roll(points, 1, axis=0)

    # Compute edge normals (perpendicular to edges)
    edge1 = points - prev_points
    edge2 = next_points - points

    # Perpendicular vectors (2D)
    normal1 = jnp.stack([-edge1[:, 1], edge1[:, 0]], axis=1)
    normal2 = jnp.stack([-edge2[:, 1], edge2[:, 0]], axis=1)

    # Normalize
    normal1 = normal1 / (jnp.linalg.norm(normal1, axis=1, keepdims=True) + 1e-8)
    normal2 = normal2 / (jnp.linalg.norm(normal2, axis=1, keepdims=True) + 1e-8)

    # Average normals at vertices
    avg_normals = (normal1 + normal2) / 2.0
    avg_normals = avg_normals / (jnp.linalg.norm(avg_normals, axis=1, keepdims=True) + 1e-8)

    # Offset points
    offset_points = points + distance * avg_normals

    return Profile2D(points=offset_points, closed=True)
