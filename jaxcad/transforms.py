"""Differentiable geometric transformations."""

import jax.numpy as jnp

from jaxcad.core import Solid


def translate(solid: Solid, offset: jnp.ndarray) -> Solid:
    """Translate a solid by an offset vector.

    Fully differentiable with respect to offset.

    Args:
        solid: Input solid to translate
        offset: Array of shape (3,) specifying translation [dx, dy, dz]

    Returns:
        Translated solid

    Example:
        >>> solid = box(jnp.zeros(3), jnp.ones(3))
        >>> translated = translate(solid, jnp.array([1., 0., 0.]))
    """
    return Solid(vertices=solid.vertices + offset, faces=solid.faces)


def scale(solid: Solid, factors: jnp.ndarray, origin: jnp.ndarray = None) -> Solid:
    """Scale a solid by given factors around an origin point.

    Fully differentiable with respect to factors and origin.

    Args:
        solid: Input solid to scale
        factors: Array of shape (3,) specifying scale factors [sx, sy, sz]
        origin: Array of shape (3,) specifying scale origin. Defaults to [0, 0, 0]

    Returns:
        Scaled solid

    Example:
        >>> solid = box(jnp.zeros(3), jnp.ones(3))
        >>> scaled = scale(solid, jnp.array([2., 1., 1.]))
    """
    if origin is None:
        origin = jnp.zeros(3)

    # Translate to origin, scale, translate back
    vertices = solid.vertices - origin
    vertices = vertices * factors
    vertices = vertices + origin

    return Solid(vertices=vertices, faces=solid.faces)


def rotate(solid: Solid, axis: jnp.ndarray, angle: float, origin: jnp.ndarray = None) -> Solid:
    """Rotate a solid around an axis by an angle.

    Uses Rodrigues' rotation formula. Fully differentiable with respect to
    axis, angle, and origin.

    Args:
        solid: Input solid to rotate
        axis: Array of shape (3,) specifying rotation axis (will be normalized)
        angle: Rotation angle in radians
        origin: Array of shape (3,) specifying rotation origin. Defaults to [0, 0, 0]

    Returns:
        Rotated solid

    Example:
        >>> solid = box(jnp.zeros(3), jnp.ones(3))
        >>> rotated = rotate(solid, jnp.array([0., 0., 1.]), jnp.pi / 4)
    """
    if origin is None:
        origin = jnp.zeros(3)

    # Normalize axis
    axis = axis / jnp.linalg.norm(axis)

    # Build rotation matrix using Rodrigues' formula
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    x, y, z = axis[0], axis[1], axis[2]

    rotation_matrix = jnp.array(
        [
            [
                cos_angle + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_angle,
                x * z * one_minus_cos + y * sin_angle,
            ],
            [
                y * x * one_minus_cos + z * sin_angle,
                cos_angle + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_angle,
            ],
            [
                z * x * one_minus_cos - y * sin_angle,
                z * y * one_minus_cos + x * sin_angle,
                cos_angle + z * z * one_minus_cos,
            ],
        ]
    )

    # Translate to origin, rotate, translate back
    vertices = solid.vertices - origin
    vertices = jnp.dot(vertices, rotation_matrix.T)
    vertices = vertices + origin

    return Solid(vertices=vertices, faces=solid.faces)


def transform(solid: Solid, matrix: jnp.ndarray) -> Solid:
    """Apply a general 4x4 transformation matrix to a solid.

    Fully differentiable with respect to the transformation matrix.

    Args:
        solid: Input solid to transform
        matrix: Array of shape (4, 4) specifying homogeneous transformation matrix

    Returns:
        Transformed solid

    Example:
        >>> solid = box(jnp.zeros(3), jnp.ones(3))
        >>> matrix = jnp.eye(4)
        >>> transformed = transform(solid, matrix)
    """
    # Convert vertices to homogeneous coordinates
    n_vertices = solid.vertices.shape[0]
    ones = jnp.ones((n_vertices, 1))
    vertices_homogeneous = jnp.concatenate([solid.vertices, ones], axis=1)

    # Apply transformation
    vertices_transformed = jnp.dot(vertices_homogeneous, matrix.T)

    # Convert back to 3D coordinates
    vertices = vertices_transformed[:, :3] / vertices_transformed[:, 3:4]

    return Solid(vertices=vertices, faces=solid.faces)
