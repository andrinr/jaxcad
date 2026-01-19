"""Complex geometric operations for creating advanced shapes."""

import jax.numpy as jnp

from jaxcad.core import Solid
from jaxcad.sketch import Profile2D


def extrude(profile: Profile2D, height: float) -> Solid:
    """Extrude a 2D profile along the z-axis to create a 3D solid.

    Fully differentiable with respect to profile points and height.

    Args:
        profile: 2D profile to extrude
        height: Extrusion height along z-axis

    Returns:
        Extruded solid

    Example:
        >>> from jaxcad.sketch import rectangle
        >>> profile = rectangle(jnp.zeros(2), 2.0, 1.0)
        >>> solid = extrude(profile, height=3.0)
    """
    points_2d = profile.points
    n = len(points_2d)

    # Create bottom and top vertices
    bottom_vertices = jnp.concatenate([points_2d, jnp.zeros((n, 1))], axis=1)

    top_vertices = jnp.concatenate([points_2d, jnp.full((n, 1), height)], axis=1)

    # Combine all vertices
    vertices = jnp.concatenate([bottom_vertices, top_vertices], axis=0)

    # Generate faces
    faces = []

    if profile.closed:
        # Side faces (quads split into triangles)
        for i in range(n):
            next_i = (i + 1) % n
            # Bottom edge: i, next_i
            # Top edge: i+n, next_i+n

            # Triangle 1
            faces.append([i, next_i, next_i + n])
            # Triangle 2
            faces.append([i, next_i + n, i + n])

        # Bottom cap (fan triangulation from first vertex)
        for i in range(1, n - 1):
            faces.append([0, i + 1, i])

        # Top cap (fan triangulation from first vertex)
        for i in range(1, n - 1):
            faces.append([n, n + i, n + i + 1])
    else:
        # For open profiles, only create side faces
        for i in range(n - 1):
            faces.append([i, i + 1, i + 1 + n])
            faces.append([i, i + 1 + n, i + n])

    faces = jnp.array(faces, dtype=jnp.int32)

    return Solid(vertices=vertices, faces=faces)


def revolve(
    profile: Profile2D,
    axis: str = "z",  # noqa: ARG001
    angle: float = 2 * jnp.pi,
    resolution: int = 32,
) -> Solid:
    """Revolve a 2D profile around an axis to create a 3D solid of revolution.

    The profile is assumed to be in the x-z plane and is revolved around the z-axis.

    Fully differentiable with respect to profile points and angle.

    Args:
        profile: 2D profile to revolve (interpreted as x-z coordinates)
        axis: Axis to revolve around ('z' or 'y')
        angle: Angle of revolution in radians (default: 2π for full revolution)
        resolution: Number of steps around the revolution

    Returns:
        Revolved solid

    Example:
        >>> from jaxcad.sketch import polygon
        >>> # Create a profile for a cone
        >>> points = jnp.array([[0., 0.], [1., 0.], [0., 2.]])
        >>> profile = polygon(points, closed=False)
        >>> solid = revolve(profile, angle=2*jnp.pi)
    """
    points_2d = profile.points  # (N, 2) - treat as (r, z) coordinates
    n_profile = len(points_2d)

    # Generate revolution steps
    theta = jnp.linspace(0, angle, resolution, endpoint=angle < 2 * jnp.pi - 0.01)
    n_theta = len(theta)

    # Create vertices by revolving each profile point
    vertices = []
    for i in range(n_profile):
        r = points_2d[i, 0]  # radial distance
        z = points_2d[i, 1]  # height

        # Generate points around circle at this height
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        z_coords = jnp.full_like(theta, z)

        profile_vertices = jnp.stack([x, y, z_coords], axis=1)
        vertices.append(profile_vertices)

    vertices = jnp.concatenate(vertices, axis=0)

    # Generate faces
    faces = []
    for i in range(n_profile - 1):
        for j in range(n_theta):
            next_j = (j + 1) % n_theta

            # Current profile ring starts at index i * n_theta
            # Next profile ring starts at index (i + 1) * n_theta
            v0 = i * n_theta + j
            v1 = i * n_theta + next_j
            v2 = (i + 1) * n_theta + next_j
            v3 = (i + 1) * n_theta + j

            # Create two triangles
            if angle >= 2 * jnp.pi - 0.01 or next_j != 0:  # Don't close if partial revolution
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

    faces = jnp.array(faces, dtype=jnp.int32)

    return Solid(vertices=vertices, faces=faces)


def loft(profiles: list[Profile2D], heights: jnp.ndarray) -> Solid:
    """Create a solid by lofting between multiple 2D profiles at different heights.

    All profiles must have the same number of points.

    Args:
        profiles: List of 2D profiles
        heights: Array of z-coordinates for each profile

    Returns:
        Lofted solid

    Example:
        >>> from jaxcad.sketch import circle
        >>> profile1 = circle(jnp.zeros(2), 1.0, resolution=16)
        >>> profile2 = circle(jnp.zeros(2), 0.5, resolution=16)
        >>> heights = jnp.array([0., 2.])
        >>> solid = loft([profile1, profile2], heights)
    """
    n_profiles = len(profiles)
    n_points = len(profiles[0].points)

    # Check all profiles have same number of points
    for p in profiles:
        assert len(p.points) == n_points, "All profiles must have same number of points"

    # Create vertices at each height
    vertices = []
    for profile, height in zip(profiles, heights):
        points_3d = jnp.concatenate([profile.points, jnp.full((n_points, 1), height)], axis=1)
        vertices.append(points_3d)

    vertices = jnp.concatenate(vertices, axis=0)

    # Generate faces between consecutive profiles
    faces = []
    for i in range(n_profiles - 1):
        for j in range(n_points):
            next_j = (j + 1) % n_points

            # Current profile ring starts at i * n_points
            # Next profile ring starts at (i + 1) * n_points
            v0 = i * n_points + j
            v1 = i * n_points + next_j
            v2 = (i + 1) * n_points + next_j
            v3 = (i + 1) * n_points + j

            # Create two triangles
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Add caps if profiles are closed
    if profiles[0].closed:
        # Bottom cap
        for j in range(1, n_points - 1):
            faces.append([0, j + 1, j])

        # Top cap
        base = (n_profiles - 1) * n_points
        for j in range(1, n_points - 1):
            faces.append([base, base + j, base + j + 1])

    faces = jnp.array(faces, dtype=jnp.int32)

    return Solid(vertices=vertices, faces=faces)


def sweep(profile: Profile2D, path: jnp.ndarray) -> Solid:
    """Sweep a 2D profile along a 3D path.

    Args:
        profile: 2D profile to sweep
        path: Array of shape (M, 3) defining the 3D path

    Returns:
        Swept solid

    Example:
        >>> from jaxcad.sketch import circle
        >>> profile = circle(jnp.zeros(2), 0.5, resolution=8)
        >>> # Create a helix path
        >>> t = jnp.linspace(0, 4*jnp.pi, 32)
        >>> path = jnp.stack([jnp.cos(t), jnp.sin(t), t/2], axis=1)
        >>> solid = sweep(profile, path)
    """
    points_2d = profile.points
    n_profile = len(points_2d)
    n_path = len(path)

    # For each point on path, place profile perpendicular to path direction
    vertices = []

    for i in range(n_path):
        # Get path position and tangent
        pos = path[i]

        # Compute tangent (forward difference)
        tangent = path[i + 1] - path[i] if i < n_path - 1 else path[i] - path[i - 1]

        tangent = tangent / (jnp.linalg.norm(tangent) + 1e-8)

        # Create a local coordinate system
        # Use arbitrary perpendicular vector
        up = jnp.array([0.0, 0.0, 1.0]) if jnp.abs(tangent[2]) < 0.9 else jnp.array([1.0, 0.0, 0.0])

        # Gram-Schmidt to get perpendicular vectors
        right = jnp.cross(up, tangent)
        right = right / (jnp.linalg.norm(right) + 1e-8)
        up = jnp.cross(tangent, right)

        # Transform profile points to 3D
        for j in range(n_profile):
            pt_2d = points_2d[j]
            pt_3d = pos + pt_2d[0] * right + pt_2d[1] * up
            vertices.append(pt_3d)

    vertices = jnp.array(vertices)

    # Generate faces
    faces = []
    for i in range(n_path - 1):
        for j in range(n_profile):
            next_j = (j + 1) % n_profile

            v0 = i * n_profile + j
            v1 = i * n_profile + next_j
            v2 = (i + 1) * n_profile + next_j
            v3 = (i + 1) * n_profile + j

            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = jnp.array(faces, dtype=jnp.int32)

    return Solid(vertices=vertices, faces=faces)


def array_linear(solid: Solid, direction: jnp.ndarray, count: int, spacing: float) -> Solid:
    """Create a linear array of copies of a solid.

    Args:
        solid: Solid to array
        direction: Direction vector for array
        count: Number of copies
        spacing: Spacing between copies

    Returns:
        Combined solid with all copies

    Example:
        >>> box_solid = box(jnp.zeros(3), jnp.ones(3))
        >>> arrayed = array_linear(box_solid, jnp.array([1., 0., 0.]), count=5, spacing=2.0)
    """
    direction_normalized = direction / jnp.linalg.norm(direction)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i in range(count):
        offset = direction_normalized * spacing * i
        vertices = solid.vertices + offset
        faces = solid.faces + vertex_offset

        all_vertices.append(vertices)
        all_faces.append(faces)
        vertex_offset += len(vertices)

    combined_vertices = jnp.concatenate(all_vertices, axis=0)
    combined_faces = jnp.concatenate(all_faces, axis=0)

    return Solid(vertices=combined_vertices, faces=combined_faces)


def array_circular(
    solid: Solid, axis: jnp.ndarray, center: jnp.ndarray, count: int, angle: float = None
) -> Solid:
    """Create a circular array of copies of a solid.

    Args:
        solid: Solid to array
        axis: Axis of rotation
        center: Center point for rotation
        count: Number of copies
        angle: Total angle to span (default: 2π for full circle)

    Returns:
        Combined solid with all copies

    Example:
        >>> from jaxcad.primitives import box
        >>> box_solid = box(jnp.array([2., 0., 0.]), jnp.ones(3))
        >>> arrayed = array_circular(box_solid, jnp.array([0., 0., 1.]),
        ...                          jnp.zeros(3), count=6)
    """
    from jaxcad.transforms import rotate

    if angle is None:
        angle = 2 * jnp.pi

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i in range(count):
        rotation_angle = (angle / count) * i
        rotated = rotate(solid, axis, rotation_angle, origin=center)

        faces = rotated.faces + vertex_offset

        all_vertices.append(rotated.vertices)
        all_faces.append(faces)
        vertex_offset += len(rotated.vertices)

    combined_vertices = jnp.concatenate(all_vertices, axis=0)
    combined_faces = jnp.concatenate(all_faces, axis=0)

    return Solid(vertices=combined_vertices, faces=combined_faces)
