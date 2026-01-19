"""Differentiable operations for modifying solid geometry (fillets, chamfers, etc.)."""

import jax.numpy as jnp

from jaxcad.core import Solid


def chamfer_vertex(solid: Solid, vertex_idx: int, distance: float) -> Solid:
    """Apply a chamfer to a specific vertex by moving it inward.

    This is a simplified chamfer that moves a vertex toward the centroid
    of its neighbors. Fully differentiable.

    Args:
        solid: Input solid
        vertex_idx: Index of vertex to chamfer
        distance: Chamfer distance (0 to 1, fraction of movement toward neighbors)

    Returns:
        Modified solid

    Example:
        >>> s = box(jnp.zeros(3), jnp.ones(3))
        >>> chamfered = chamfer_vertex(s, vertex_idx=0, distance=0.2)
    """
    vertices = solid.vertices

    # Find neighboring vertices (vertices that share a face with this vertex)
    neighbor_indices = []
    for face in solid.faces:
        face_list = [int(face[0]), int(face[1]), int(face[2])]
        if vertex_idx in face_list:
            neighbor_indices.extend([v for v in face_list if v != vertex_idx])

    # Get unique neighbors
    neighbor_indices = list(set(neighbor_indices))

    if len(neighbor_indices) > 0:
        # Compute centroid of neighbors
        neighbor_positions = vertices[jnp.array(neighbor_indices)]
        centroid = jnp.mean(neighbor_positions, axis=0)

        # Move vertex toward centroid
        new_position = vertices[vertex_idx] + distance * (centroid - vertices[vertex_idx])
        vertices = vertices.at[vertex_idx].set(new_position)

    return Solid(vertices=vertices, faces=solid.faces)


def fillet_vertex(solid: Solid, vertex_idx: int, radius: float) -> Solid:
    """Apply a fillet to a specific vertex.

    This is a simplified fillet using the same approach as chamfer.
    For true fillets, we would need to add new geometry.

    Args:
        solid: Input solid
        vertex_idx: Index of vertex to fillet
        radius: Fillet radius (0 to 1, fraction of movement)

    Returns:
        Modified solid
    """
    # For simplicity, use the same implementation as chamfer
    # A proper fillet would add new faces
    return chamfer_vertex(solid, vertex_idx, radius)


def shell(solid: Solid, thickness: float, inward: bool = True) -> Solid:
    """Create a hollow shell from a solid by offsetting vertices.

    This creates an approximate shell by moving vertices along their normals.
    Fully differentiable.

    Args:
        solid: Input solid
        thickness: Shell wall thickness
        inward: If True, offset inward; if False, offset outward

    Returns:
        Shelled solid (hollow)

    Example:
        >>> s = sphere(jnp.zeros(3), 2.0, resolution=16)
        >>> shelled = shell(s, thickness=0.2, inward=True)
    """
    vertices = solid.vertices
    faces = solid.faces

    # Compute vertex normals
    vertex_normals = compute_vertex_normals(solid)

    # Offset vertices along normals
    offset_direction = -1.0 if inward else 1.0
    inner_vertices = vertices + offset_direction * thickness * vertex_normals

    # Create new solid with both inner and outer shells
    # Outer shell: original vertices and faces
    # Inner shell: offset vertices with reversed faces

    n_vertices = len(vertices)

    # Combine vertices
    combined_vertices = jnp.concatenate([vertices, inner_vertices], axis=0)

    # Create inner faces (reversed winding)
    inner_faces = faces + n_vertices
    inner_faces_reversed = jnp.stack(
        [inner_faces[:, 0], inner_faces[:, 2], inner_faces[:, 1]], axis=1
    )

    # Create side faces connecting outer and inner shells at edges
    # For simplicity, we'll just use outer and inner faces
    # A proper implementation would connect them at edges

    combined_faces = jnp.concatenate([faces, inner_faces_reversed], axis=0)

    return Solid(vertices=combined_vertices, faces=combined_faces)


def compute_vertex_normals(solid: Solid) -> jnp.ndarray:
    """Compute per-vertex normals by averaging face normals.

    Fully differentiable.

    Args:
        solid: Input solid

    Returns:
        Array of shape (N, 3) with vertex normals
    """
    vertices = solid.vertices
    faces = solid.faces

    # Initialize normal accumulator
    normals = jnp.zeros_like(vertices)

    # Compute face normals and accumulate at vertices
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

        # Compute face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = jnp.cross(edge1, edge2)
        face_normal = face_normal / (jnp.linalg.norm(face_normal) + 1e-8)

        # Accumulate to vertex normals
        normals = normals.at[face[0]].add(face_normal)
        normals = normals.at[face[1]].add(face_normal)
        normals = normals.at[face[2]].add(face_normal)

    # Normalize vertex normals
    normals = normals / (jnp.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    return normals


def thicken(solid: Solid, thickness: float) -> Solid:
    """Thicken a surface by offsetting vertices along normals in both directions.

    Fully differentiable.

    Args:
        solid: Input solid (typically a surface)
        thickness: Total thickness

    Returns:
        Thickened solid

    Example:
        >>> # Create a flat rectangle surface
        >>> from jaxcad.sketch import rectangle
        >>> from jaxcad.operations import extrude
        >>> profile = rectangle(jnp.zeros(2), 2.0, 2.0)
        >>> flat = extrude(profile, 0.01)
        >>> thickened = thicken(flat, thickness=0.5)
    """
    vertices = solid.vertices
    faces = solid.faces

    # Compute vertex normals
    vertex_normals = compute_vertex_normals(solid)

    # Create offset vertices in both directions
    half_thickness = thickness / 2.0
    top_vertices = vertices + half_thickness * vertex_normals
    bottom_vertices = vertices - half_thickness * vertex_normals

    n_vertices = len(vertices)

    # Combine vertices
    combined_vertices = jnp.concatenate([top_vertices, bottom_vertices], axis=0)

    # Top faces (original orientation)
    top_faces = faces

    # Bottom faces (reversed winding)
    bottom_faces = faces + n_vertices
    bottom_faces_reversed = jnp.stack(
        [bottom_faces[:, 0], bottom_faces[:, 2], bottom_faces[:, 1]], axis=1
    )

    combined_faces = jnp.concatenate([top_faces, bottom_faces_reversed], axis=0)

    return Solid(vertices=combined_vertices, faces=combined_faces)


def twist(solid: Solid, axis: jnp.ndarray, angle: float, height_range: tuple = None) -> Solid:
    """Apply a twist deformation along an axis.

    Fully differentiable.

    Args:
        solid: Input solid
        axis: Twist axis (will be normalized)
        angle: Total twist angle in radians
        height_range: Optional (min_height, max_height) for normalized twist

    Returns:
        Twisted solid

    Example:
        >>> s = box(jnp.zeros(3), jnp.array([1., 1., 4.]))
        >>> twisted = twist(s, jnp.array([0., 0., 1.]), jnp.pi/2)
    """

    vertices = solid.vertices
    axis = axis / jnp.linalg.norm(axis)

    # Determine height range along axis
    if height_range is None:
        projections = jnp.dot(vertices, axis)
        min_h = jnp.min(projections)
        max_h = jnp.max(projections)
    else:
        min_h, max_h = height_range

    height_span = max_h - min_h + 1e-8

    # For each vertex, compute twist amount based on height
    twisted_vertices = []
    for vertex in vertices:
        # Project onto axis
        h = jnp.dot(vertex, axis)
        t = (h - min_h) / height_span  # Normalized height [0, 1]

        # Compute twist angle for this vertex
        vertex_angle = angle * t

        # Rotate vertex around axis by this angle
        # Use rotation matrix around axis
        cos_a = jnp.cos(vertex_angle)
        sin_a = jnp.sin(vertex_angle)
        ux, uy, uz = axis[0], axis[1], axis[2]

        # Rodrigues rotation matrix
        rotation_matrix = jnp.array(
            [
                [
                    cos_a + ux * ux * (1 - cos_a),
                    ux * uy * (1 - cos_a) - uz * sin_a,
                    ux * uz * (1 - cos_a) + uy * sin_a,
                ],
                [
                    uy * ux * (1 - cos_a) + uz * sin_a,
                    cos_a + uy * uy * (1 - cos_a),
                    uy * uz * (1 - cos_a) - ux * sin_a,
                ],
                [
                    uz * ux * (1 - cos_a) - uy * sin_a,
                    uz * uy * (1 - cos_a) + ux * sin_a,
                    cos_a + uz * uz * (1 - cos_a),
                ],
            ]
        )

        twisted_vertex = jnp.dot(rotation_matrix, vertex)
        twisted_vertices.append(twisted_vertex)

    twisted_vertices = jnp.array(twisted_vertices)

    return Solid(vertices=twisted_vertices, faces=solid.faces)


def taper(solid: Solid, axis: jnp.ndarray, scale_top: float, scale_bottom: float = 1.0) -> Solid:
    """Apply a taper (scale gradient) along an axis.

    Fully differentiable.

    Args:
        solid: Input solid
        axis: Taper axis (will be normalized)
        scale_top: Scale factor at the top
        scale_bottom: Scale factor at the bottom (default: 1.0)

    Returns:
        Tapered solid

    Example:
        >>> s = cylinder(jnp.zeros(3), 1.0, 4.0)
        >>> tapered = taper(s, jnp.array([0., 0., 1.]), scale_top=0.5)
    """
    vertices = solid.vertices
    axis = axis / jnp.linalg.norm(axis)

    # Find height range along axis
    projections = jnp.dot(vertices, axis)
    min_h = jnp.min(projections)
    max_h = jnp.max(projections)
    height_span = max_h - min_h + 1e-8

    # For each vertex, scale perpendicular components based on height
    tapered_vertices = []
    for vertex in vertices:
        h = jnp.dot(vertex, axis)
        t = (h - min_h) / height_span  # Normalized height [0, 1]

        # Interpolate scale factor
        scale_factor = scale_bottom + t * (scale_top - scale_bottom)

        # Decompose vertex into parallel and perpendicular components
        parallel_component = h * axis
        perpendicular_component = vertex - parallel_component

        # Scale perpendicular component
        tapered_vertex = parallel_component + scale_factor * perpendicular_component
        tapered_vertices.append(tapered_vertex)

    tapered_vertices = jnp.array(tapered_vertices)

    return Solid(vertices=tapered_vertices, faces=solid.faces)
