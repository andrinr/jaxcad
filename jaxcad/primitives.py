"""Differentiable primitive shape generators."""

import jax.numpy as jnp

from jaxcad.core import Solid


def box(center: jnp.ndarray, size: jnp.ndarray) -> Solid:
    """Create a box primitive.

    All operations are differentiable with respect to center and size parameters.

    Args:
        center: Array of shape (3,) specifying box center [x, y, z]
        size: Array of shape (3,) specifying box dimensions [width, height, depth]

    Returns:
        Solid with 8 vertices and 12 triangular faces

    Example:
        >>> center = jnp.array([0., 0., 0.])
        >>> size = jnp.array([2., 2., 2.])
        >>> solid = box(center, size)
    """
    # Compute half extents
    half_size = size / 2.0

    # Define 8 vertices of a box centered at origin
    vertices = jnp.array(
        [
            [-1, -1, -1],  # 0
            [1, -1, -1],  # 1
            [1, 1, -1],  # 2
            [-1, 1, -1],  # 3
            [-1, -1, 1],  # 4
            [1, -1, 1],  # 5
            [1, 1, 1],  # 6
            [-1, 1, 1],  # 7
        ],
        dtype=jnp.float32,
    )

    # Scale by half_size and translate to center
    vertices = vertices * half_size + center

    # Define 12 triangular faces (2 per box face)
    faces = jnp.array(
        [
            # Bottom face (z = -half_size[2])
            [0, 1, 2],
            [0, 2, 3],
            # Top face (z = half_size[2])
            [4, 6, 5],
            [4, 7, 6],
            # Front face (y = -half_size[1])
            [0, 5, 1],
            [0, 4, 5],
            # Back face (y = half_size[1])
            [2, 7, 3],
            [2, 6, 7],
            # Left face (x = -half_size[0])
            [0, 3, 7],
            [0, 7, 4],
            # Right face (x = half_size[0])
            [1, 6, 2],
            [1, 5, 6],
        ],
        dtype=jnp.int32,
    )

    return Solid(vertices=vertices, faces=faces)


def sphere(center: jnp.ndarray, radius: float, resolution: int = 16) -> Solid:
    """Create a sphere primitive using UV sphere tessellation.

    All operations are differentiable with respect to center and radius.

    Args:
        center: Array of shape (3,) specifying sphere center [x, y, z]
        radius: Sphere radius
        resolution: Number of subdivisions (higher = smoother sphere)

    Returns:
        Solid with tessellated sphere mesh

    Example:
        >>> center = jnp.array([0., 0., 0.])
        >>> solid = sphere(center, radius=1.0, resolution=16)
    """
    # Create UV sphere parametrization
    u = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)
    v = jnp.linspace(0, jnp.pi, resolution // 2 + 1)

    # Generate vertices using spherical coordinates
    u_grid, v_grid = jnp.meshgrid(u, v)
    x = radius * jnp.sin(v_grid) * jnp.cos(u_grid)
    y = radius * jnp.sin(v_grid) * jnp.sin(u_grid)
    z = radius * jnp.cos(v_grid)

    # Flatten and stack coordinates
    vertices = jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=1) + center

    # Generate face indices
    faces = []
    n_lat = resolution // 2 + 1  # number of latitude lines
    n_lon = resolution  # number of longitude lines

    for i in range(n_lat - 1):
        for j in range(n_lon):
            # Current quad vertices
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + (j + 1) % n_lon
            v3 = (i + 1) * n_lon + j

            # Create two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = jnp.array(faces, dtype=jnp.int32)

    return Solid(vertices=vertices, faces=faces)


def cylinder(center: jnp.ndarray, radius: float, height: float, resolution: int = 32) -> Solid:
    """Create a cylinder primitive.

    All operations are differentiable with respect to center, radius, and height.

    Args:
        center: Array of shape (3,) specifying cylinder center [x, y, z]
        radius: Cylinder radius
        height: Cylinder height (along z-axis)
        resolution: Number of radial subdivisions

    Returns:
        Solid with tessellated cylinder mesh

    Example:
        >>> center = jnp.array([0., 0., 0.])
        >>> solid = cylinder(center, radius=1.0, height=2.0, resolution=32)
    """
    # Create circle vertices at top and bottom
    theta = jnp.linspace(0, 2 * jnp.pi, resolution, endpoint=False)

    # Bottom circle (z = -height/2)
    x_bottom = radius * jnp.cos(theta)
    y_bottom = radius * jnp.sin(theta)
    z_bottom = jnp.full_like(theta, -height / 2.0)
    bottom_vertices = jnp.stack([x_bottom, y_bottom, z_bottom], axis=1)

    # Top circle (z = height/2)
    x_top = radius * jnp.cos(theta)
    y_top = radius * jnp.sin(theta)
    z_top = jnp.full_like(theta, height / 2.0)
    top_vertices = jnp.stack([x_top, y_top, z_top], axis=1)

    # Center vertices for caps
    center_bottom = jnp.array([[0.0, 0.0, -height / 2.0]])
    center_top = jnp.array([[0.0, 0.0, height / 2.0]])

    # Combine all vertices and translate to center
    vertices = (
        jnp.concatenate(
            [
                bottom_vertices,  # 0 to resolution-1
                top_vertices,  # resolution to 2*resolution-1
                center_bottom,  # 2*resolution
                center_top,  # 2*resolution+1
            ],
            axis=0,
        )
        + center
    )

    # Generate faces
    faces = []

    # Side faces
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Bottom to top
        faces.append([i, next_i, resolution + next_i])
        faces.append([i, resolution + next_i, resolution + i])

    # Bottom cap
    center_bottom_idx = 2 * resolution
    for i in range(resolution):
        next_i = (i + 1) % resolution
        faces.append([center_bottom_idx, next_i, i])

    # Top cap
    center_top_idx = 2 * resolution + 1
    for i in range(resolution):
        next_i = (i + 1) % resolution
        faces.append([center_top_idx, resolution + i, resolution + next_i])

    faces = jnp.array(faces, dtype=jnp.int32)

    return Solid(vertices=vertices, faces=faces)
