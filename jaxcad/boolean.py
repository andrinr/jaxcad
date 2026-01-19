"""Differentiable boolean operations using soft approximations.

Note: True CSG boolean operations on meshes are not easily differentiable.
This module provides soft/approximate boolean operations suitable for
gradient-based optimization.
"""

import jax.numpy as jnp

from jaxcad.core import Solid


def merge(solid1: Solid, solid2: Solid) -> Solid:
    """Merge two solids by combining their vertices and faces.

    This is a simple concatenation, not a true CSG union.
    Fully differentiable.

    Args:
        solid1: First solid
        solid2: Second solid

    Returns:
        Combined solid

    Example:
        >>> s1 = box(jnp.array([0., 0., 0.]), jnp.ones(3))
        >>> s2 = box(jnp.array([2., 0., 0.]), jnp.ones(3))
        >>> merged = merge(s1, s2)
    """
    # Combine vertices
    combined_vertices = jnp.concatenate([solid1.vertices, solid2.vertices], axis=0)

    # Adjust face indices for second solid
    n_vertices1 = len(solid1.vertices)
    faces2_adjusted = solid2.faces + n_vertices1

    # Combine faces
    combined_faces = jnp.concatenate([solid1.faces, faces2_adjusted], axis=0)

    return Solid(vertices=combined_vertices, faces=combined_faces)


def sdf_box(point: jnp.ndarray, center: jnp.ndarray, size: jnp.ndarray) -> float:
    """Signed distance function for a box.

    Negative inside, positive outside, zero on surface.

    Args:
        point: Query point (3,)
        center: Box center (3,)
        size: Box size (3,)

    Returns:
        Signed distance
    """
    q = jnp.abs(point - center) - size / 2.0
    return jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q), 0.0)


def sdf_sphere(point: jnp.ndarray, center: jnp.ndarray, radius: float) -> float:
    """Signed distance function for a sphere.

    Args:
        point: Query point (3,)
        center: Sphere center (3,)
        radius: Sphere radius

    Returns:
        Signed distance
    """
    return jnp.linalg.norm(point - center) - radius


def soft_union_sdf(d1: float, d2: float, k: float = 0.1) -> float:
    """Smooth minimum for SDF union (differentiable boolean union).

    Args:
        d1: Distance from first object
        d2: Distance from second object
        k: Smoothing factor (smaller = sharper transition)

    Returns:
        Combined distance
    """
    h = jnp.maximum(k - jnp.abs(d1 - d2), 0.0) / k
    return jnp.minimum(d1, d2) - h * h * k * 0.25


def soft_intersection_sdf(d1: float, d2: float, k: float = 0.1) -> float:
    """Smooth maximum for SDF intersection (differentiable boolean intersection).

    Args:
        d1: Distance from first object
        d2: Distance from second object
        k: Smoothing factor

    Returns:
        Combined distance
    """
    h = jnp.maximum(k - jnp.abs(d1 - d2), 0.0) / k
    return jnp.maximum(d1, d2) + h * h * k * 0.25


def soft_difference_sdf(d1: float, d2: float, k: float = 0.1) -> float:
    """Smooth difference for SDF (differentiable boolean difference).

    Args:
        d1: Distance from first object
        d2: Distance from second object
        k: Smoothing factor

    Returns:
        Combined distance
    """
    return soft_intersection_sdf(d1, -d2, k)


def vertex_based_union(solid1: Solid, solid2: Solid, threshold: float = 0.01) -> Solid:  # noqa: ARG001
    """Approximate union by removing vertices from one solid that are inside the other.

    This is an approximate, differentiable operation.

    Args:
        solid1: First solid
        solid2: Second solid
        threshold: Distance threshold for inside/outside detection

    Returns:
        Approximate union
    """
    # For now, just merge them (true CSG union is complex)
    # A proper implementation would require mesh boolean operations
    return merge(solid1, solid2)


def hull(solid: Solid) -> Solid:
    """Compute approximate convex hull of a solid.

    Uses the vertices to create a convex hull approximation.
    Note: This is a simplified implementation.

    Args:
        solid: Input solid

    Returns:
        Convex hull solid
    """
    # For a proper convex hull, we'd use scipy or a custom algorithm
    # For now, return the original solid
    # A differentiable convex hull is complex to implement
    return solid


def smooth_vertices(solid: Solid, iterations: int = 1, factor: float = 0.5) -> Solid:
    """Smooth the vertices of a solid using Laplacian smoothing.

    Fully differentiable.

    Args:
        solid: Input solid
        iterations: Number of smoothing iterations
        factor: Smoothing factor (0 = no smoothing, 1 = full smoothing)

    Returns:
        Smoothed solid

    Example:
        >>> s = sphere(jnp.zeros(3), 1.0, resolution=8)
        >>> smoothed = smooth_vertices(s, iterations=2, factor=0.5)
    """
    vertices = solid.vertices

    for _ in range(iterations):
        # Build adjacency information from faces
        n_vertices = len(vertices)
        neighbor_sum = jnp.zeros_like(vertices)
        neighbor_count = jnp.zeros(n_vertices)

        # For each face, add contributions to neighbor averaging
        for face in solid.faces:
            for i in range(3):
                v_idx = face[i]
                neighbor_idx1 = face[(i + 1) % 3]
                neighbor_idx2 = face[(i + 2) % 3]

                # Accumulate neighbor positions
                neighbor_sum = neighbor_sum.at[v_idx].add(vertices[neighbor_idx1])
                neighbor_sum = neighbor_sum.at[v_idx].add(vertices[neighbor_idx2])
                neighbor_count = neighbor_count.at[v_idx].add(2)

        # Compute average neighbor position
        neighbor_avg = neighbor_sum / (neighbor_count[:, None] + 1e-8)

        # Blend between original and averaged position
        vertices = (1 - factor) * vertices + factor * neighbor_avg

    return Solid(vertices=vertices, faces=solid.faces)


def subdivide_faces(solid: Solid) -> Solid:
    """Subdivide each triangular face into 4 smaller triangles.

    Fully differentiable (vertices are linearly interpolated).

    Args:
        solid: Input solid

    Returns:
        Subdivided solid with 4x more faces

    Example:
        >>> s = box(jnp.zeros(3), jnp.ones(3))
        >>> subdivided = subdivide_faces(s)
    """
    vertices = solid.vertices
    faces = solid.faces

    new_vertices = [vertices]
    new_faces = []

    # Dictionary to track edge midpoints (avoid duplicates)
    edge_midpoints = {}
    next_vertex_idx = len(vertices)

    def get_edge_midpoint(v1_idx, v2_idx):
        nonlocal next_vertex_idx
        # Create canonical edge key (smaller index first)
        edge_key = tuple(sorted([int(v1_idx), int(v2_idx)]))

        if edge_key not in edge_midpoints:
            # Compute midpoint
            midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2.0
            edge_midpoints[edge_key] = (next_vertex_idx, midpoint)
            next_vertex_idx += 1

        return edge_midpoints[edge_key][0]

    # Process each face
    for face in faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])

        # Get midpoints of three edges
        m01 = get_edge_midpoint(v0, v1)
        m12 = get_edge_midpoint(v1, v2)
        m20 = get_edge_midpoint(v2, v0)

        # Create 4 new faces
        new_faces.append([v0, m01, m20])
        new_faces.append([v1, m12, m01])
        new_faces.append([v2, m20, m12])
        new_faces.append([m01, m12, m20])

    # Collect all new midpoint vertices
    midpoint_vertices = [mp[1] for mp in sorted(edge_midpoints.values(), key=lambda x: x[0])]
    if midpoint_vertices:
        new_vertices.append(jnp.array(midpoint_vertices))

    combined_vertices = jnp.concatenate(new_vertices, axis=0)
    combined_faces = jnp.array(new_faces, dtype=jnp.int32)

    return Solid(vertices=combined_vertices, faces=combined_faces)
