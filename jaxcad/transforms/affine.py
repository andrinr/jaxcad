"""Affine transformations for SDFs."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Vector, Scalar
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
        # Accept both raw values and constraints
        if isinstance(offset, Vector):
            self.offset_param = offset
        else:
            # Wrap raw value in a fixed Vector constraint
            self.offset_param = Vector(value=jnp.asarray(offset), free=False)

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
        return Translate.sdf(self.sdf, p, self.offset_param.xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Translate.sdf


class Scale(Transform):
    """Scale an SDF uniformly or non-uniformly.

    Note: Non-uniform scaling doesn't produce exact SDFs. For uniform scaling,
    we can divide the distance by the scale factor to maintain correctness.

    Args:
        sdf: The SDF to scale
        scale: Uniform scale factor (float/Distance) or per-axis scale (Array/Vector)
    """

    def __init__(self, sdf: SDF, scale: Union[float, Array, Scalar, Vector]):
        self.sdf = sdf
        # Accept both raw values and parameters
        if isinstance(scale, (Scalar, Vector)):
            self.scale_param = scale
        elif isinstance(scale, (int, float)):
            # Uniform scale - wrap in Scalar parameter
            self.scale_param = Scalar(value=float(scale), free=False)
        else:
            # Non-uniform scale - wrap in Vector parameter
            self.scale_param = Vector(value=jnp.asarray(scale), free=False)

        self.is_uniform = isinstance(self.scale_param, Scalar)

    @staticmethod
    def sdf(child_sdf, p: Array, scale: float | Array) -> Array:
        """Pure function for scaling.

        Args:
            child_sdf: SDF function to scale
            p: Query point(s)
            scale: Scale factor (uniform) or scale vector (non-uniform)

        Returns:
            Scaled SDF value
        """
        is_uniform = isinstance(scale, (int, float)) or scale.ndim == 0
        if is_uniform:
            # Uniform scaling: divide point by scale, multiply distance by scale
            return child_sdf(p / scale) * scale
        else:
            # Non-uniform scaling: approximate (not exact SDF)
            return child_sdf(p / scale)

    def __call__(self, p: Array) -> Array:
        """Evaluate scaled SDF."""
        scale = self.scale_param.xyz if isinstance(self.scale_param, Vector) else self.scale_param.value
        return Scale.sdf(self.sdf, p, scale)

    def to_functional(self):
        """Return pure function for compilation."""
        return Scale.sdf


class Rotate(Transform):
    """Rotate an SDF around an axis.

    Args:
        sdf: The SDF to rotate
        axis: Rotation axis ('x', 'y', 'z') or custom axis vector
        angle: Rotation angle in radians (float or Angle constraint)
    """

    def __init__(self, sdf: SDF, axis: str | Array, angle: Union[float, Scalar]):
        self.sdf = sdf
        self.axis = axis

        # Accept both raw values and Scalar parameters
        if isinstance(angle, Scalar):
            self.angle_param = angle
        else:
            # Wrap raw value in a fixed Scalar parameter
            self.angle_param = Scalar(value=float(angle), free=False)

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
    def sdf(child_sdf, p: Array, angle: float) -> Array:
        """Pure function for Z-axis rotation.

        Args:
            child_sdf: SDF function to rotate
            p: Query point(s)
            angle: Rotation angle in radians

        Returns:
            Rotated SDF value
        """
        c, s = jnp.cos(angle), jnp.sin(angle)
        # Apply inverse rotation to point
        if p.ndim == 1:
            x = p[0] * c + p[1] * s
            y = -p[0] * s + p[1] * c
            z = p[2]
            p_rotated = jnp.array([x, y, z])
        else:
            x = p[..., 0] * c + p[..., 1] * s
            y = -p[..., 0] * s + p[..., 1] * c
            z = p[..., 2]
            p_rotated = jnp.stack([x, y, z], axis=-1)
        return child_sdf(p_rotated)

    def __call__(self, p: Array) -> Array:
        """Evaluate rotated SDF."""
        return Rotate.sdf(self.sdf, p, self.angle_param.value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Rotate.sdf
