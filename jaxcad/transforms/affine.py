"""Affine transformations for SDFs."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Vector, Scalar
from jaxcad.sdf import SDF
from jaxcad.transforms.base import Transform


class Translate(Transform):
    """Translate an SDF by a vector offset.

    Note: For SDFs, we translate by applying the *inverse* transform to the
    query point. This moves the geometry in the opposite direction.

    Args:
        sdf: The SDF to translate
        offset: Translation vector (Array or Vector constraint)
    """

    def __init__(self, sdf: SDF, offset: Union[Array, Vector]):
        self.sdf = sdf
        self.params = {'offset': offset}

    @staticmethod
    def sdf(child_sdf, p: Array, offset: Array) -> Array:
        """Pure function for translation.

        Args:
            child_sdf: SDF function to translate
            p: Query point(s)
            offset: Translation vector [x, y, z]

        Returns:
            Translated SDF value
        """
        return child_sdf(p - offset)

    def __call__(self, p: Array) -> Array:
        """Evaluate translated SDF."""
        return Translate.sdf(self.sdf, p, self.params['offset'].xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Translate.sdf


class Scale(Transform):
    """Scale an SDF component-wise.

    Note: Non-uniform scaling doesn't produce exact SDFs. For uniform scaling,
    we can divide the distance by the scale factor to maintain correctness.

    Args:
        sdf: The SDF to scale
        scale: Per-axis scale as Array [sx, sy, sz], Vector parameter, or float for uniform scaling
    """

    def __init__(self, sdf: SDF, scale: Union[float, Array, Vector]):
        self.sdf = sdf
        # Convert scalar to uniform 3D scale vector before auto-cast
        if isinstance(scale, (int, float)):
            scale = jnp.array([scale, scale, scale])
        self.params = {'scale': scale}

    @staticmethod
    def sdf(child_sdf, p: Array, scale: Array) -> Array:
        """Pure function for component-wise scaling.

        Args:
            child_sdf: SDF function to scale
            p: Query point(s)
            scale: Scale vector [sx, sy, sz]

        Returns:
            Scaled SDF value
        """
        # Check if uniform by comparing all components to first
        is_uniform = jnp.allclose(scale, scale[0])

        # Use jnp.where for JAX-compatible branching
        def uniform_scale():
            s = scale[0]
            return child_sdf(p / s) * s

        def nonuniform_scale():
            return child_sdf(p / scale)

        return jnp.where(is_uniform, uniform_scale(), nonuniform_scale())

    def __call__(self, p: Array) -> Array:
        """Evaluate scaled SDF."""
        return Scale.sdf(self.sdf, p, self.params['scale'].xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Scale.sdf


class Rotate(Transform):
    """Rotate an SDF around an axis.

    Args:
        sdf: The SDF to rotate
        axis: Rotation axis as Array [x, y, z], Vector parameter, or string ('x', 'y', 'z')
        angle: Rotation angle in radians (float or Scalar parameter)
    """

    def __init__(self, sdf: SDF, axis: Union[str, Array, Vector], angle: Union[float, Scalar]):
        self.sdf = sdf
        # Convert string axis to vector before auto-cast
        if isinstance(axis, str):
            axis_map = {
                'x': jnp.array([1.0, 0.0, 0.0]),
                'y': jnp.array([0.0, 1.0, 0.0]),
                'z': jnp.array([0.0, 0.0, 1.0])
            }
            axis = axis_map.get(axis.lower(), jnp.array([0.0, 0.0, 1.0]))
        self.params = {
            'axis': axis,
            'angle': angle
        }

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

    @staticmethod
    def sdf(child_sdf, p: Array, axis: Array, angle: float) -> Array:
        """Pure function for rotation around arbitrary axis.

        Args:
            child_sdf: SDF function to rotate
            p: Query point(s)
            axis: Rotation axis [x, y, z]
            angle: Rotation angle in radians

        Returns:
            Rotated SDF value
        """
        # Normalize axis
        axis = axis / jnp.linalg.norm(axis)

        # Rodrigues' rotation formula for arbitrary axis
        c, s = jnp.cos(angle), jnp.sin(angle)
        t = 1 - c
        x, y, z = axis[0], axis[1], axis[2]

        # Build rotation matrix
        R = jnp.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
        ])

        # Apply inverse rotation to point
        if p.ndim == 1:
            p_rotated = R.T @ p
        else:
            p_rotated = jnp.einsum('ij,...j->...i', R.T, p)

        return child_sdf(p_rotated)

    def __call__(self, p: Array) -> Array:
        """Evaluate rotated SDF."""
        return Rotate.sdf(self.sdf, p, self.params['axis'].xyz, self.params['angle'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Rotate.sdf
