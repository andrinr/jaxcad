"""Affine transformations for SDFs."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Translate(SDF):
    """Translate an SDF by a vector offset.

    Note: For SDFs, we translate by applying the *inverse* transform to the
    query point. This moves the geometry in the opposite direction.

    Args:
        sdf: The SDF to translate
        offset: Translation vector (3D or 2D depending on SDF)
    """

    def __init__(self, sdf: SDF, offset: Array):
        self.sdf = sdf
        self.offset = jnp.asarray(offset)

    def __call__(self, p: Array) -> Array:
        """Evaluate translated SDF."""
        # Apply inverse translation to query point
        return self.sdf(p - self.offset)


class Scale(SDF):
    """Scale an SDF uniformly or non-uniformly.

    Note: Non-uniform scaling doesn't produce exact SDFs. For uniform scaling,
    we can divide the distance by the scale factor to maintain correctness.

    Args:
        sdf: The SDF to scale
        scale: Uniform scale factor (float) or per-axis scale (Array)
    """

    def __init__(self, sdf: SDF, scale: float | Array):
        self.sdf = sdf
        self.scale = jnp.asarray(scale) if not isinstance(scale, (int, float)) else scale
        self.is_uniform = isinstance(scale, (int, float))

    def __call__(self, p: Array) -> Array:
        """Evaluate scaled SDF."""
        if self.is_uniform:
            # Uniform scaling: divide point by scale, multiply distance by scale
            return self.sdf(p / self.scale) * self.scale
        else:
            # Non-uniform scaling: approximate (not exact SDF)
            return self.sdf(p / self.scale)


class Rotate(SDF):
    """Rotate an SDF around an axis.

    Args:
        sdf: The SDF to rotate
        axis: Rotation axis ('x', 'y', 'z') or custom axis vector
        angle: Rotation angle in radians
    """

    def __init__(self, sdf: SDF, axis: str | Array, angle: float):
        self.sdf = sdf
        self.angle = angle

        # Compute rotation matrix
        if isinstance(axis, str):
            if axis == 'x':
                self.rotation_matrix = self._rotation_matrix_x(angle)
            elif axis == 'y':
                self.rotation_matrix = self._rotation_matrix_y(angle)
            elif axis == 'z':
                self.rotation_matrix = self._rotation_matrix_z(angle)
            else:
                raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', 'z' or provide axis vector")
        else:
            # Custom axis rotation using Rodrigues' formula
            self.rotation_matrix = self._rotation_matrix_axis(jnp.asarray(axis), angle)

    @staticmethod
    def _rotation_matrix_x(angle: float) -> Array:
        """Rotation matrix around X axis."""
        c, s = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    @staticmethod
    def _rotation_matrix_y(angle: float) -> Array:
        """Rotation matrix around Y axis."""
        c, s = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    @staticmethod
    def _rotation_matrix_z(angle: float) -> Array:
        """Rotation matrix around Z axis."""
        c, s = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def _rotation_matrix_axis(axis: Array, angle: float) -> Array:
        """Rotation matrix around arbitrary axis using Rodrigues' formula."""
        axis = axis / jnp.linalg.norm(axis)
        c, s = jnp.cos(angle), jnp.sin(angle)
        t = 1 - c
        x, y, z = axis[0], axis[1], axis[2]

        return jnp.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])

    def __call__(self, p: Array) -> Array:
        """Evaluate rotated SDF."""
        # Apply inverse rotation (transpose) to query point
        # rotation_matrix.T @ p for each point
        if p.ndim == 1:
            # Single point
            p_rotated = self.rotation_matrix.T @ p
        else:
            # Multiple points: (N, 3) @ (3, 3).T = (N, 3)
            p_rotated = p @ self.rotation_matrix

        return self.sdf(p_rotated)


# Convenience functions
def translate(sdf: SDF, offset: Array) -> Translate:
    """Translate an SDF by offset vector.

    Args:
        sdf: SDF to translate
        offset: Translation vector

    Returns:
        Translated SDF
    """
    return Translate(sdf, offset)


def scale(sdf: SDF, scale: float | Array) -> Scale:
    """Scale an SDF uniformly or non-uniformly.

    Args:
        sdf: SDF to scale
        scale: Scale factor (uniform) or scale vector (non-uniform)

    Returns:
        Scaled SDF
    """
    return Scale(sdf, scale)


def rotate(sdf: SDF, axis: str | Array, angle: float) -> Rotate:
    """Rotate an SDF around an axis.

    Args:
        sdf: SDF to rotate
        axis: Rotation axis ('x', 'y', 'z') or custom axis vector
        angle: Rotation angle in radians

    Returns:
        Rotated SDF
    """
    return Rotate(sdf, axis, angle)
