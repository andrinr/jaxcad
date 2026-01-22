"""Affine transformations for SDFs."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Point, Distance, Angle
from jaxcad.sdf import SDF


class Translate(SDF):
    """Translate an SDF by a vector offset.

    Note: For SDFs, we translate by applying the *inverse* transform to the
    query point. This moves the geometry in the opposite direction.

    Args:
        sdf: The SDF to translate
        offset: Translation vector (Array or Point constraint)
    """

    def __init__(self, sdf: SDF, offset: Union[Array, Point]):
        self.sdf = sdf
        # Accept both raw values and constraints
        if isinstance(offset, Point):
            self.offset_param = offset
        else:
            # Wrap raw value in a fixed Point constraint
            self.offset_param = Point(value=jnp.asarray(offset), free=False)

    def __call__(self, p: Array) -> Array:
        """Evaluate translated SDF."""
        # Use static functional method
        return Translate.eval(self.sdf, p, self.offset_param.value)

    @staticmethod
    def eval(sdf_fn, p: Array, offset: Array) -> Array:
        """Functional evaluation of translation.

        Args:
            sdf_fn: SDF function to translate
            p: Query point(s)
            offset: Translation vector

        Returns:
            Translated SDF value
        """
        return sdf_fn(p - offset)


class Scale(SDF):
    """Scale an SDF uniformly or non-uniformly.

    Note: Non-uniform scaling doesn't produce exact SDFs. For uniform scaling,
    we can divide the distance by the scale factor to maintain correctness.

    Args:
        sdf: The SDF to scale
        scale: Uniform scale factor (float/Distance) or per-axis scale (Array/Point)
    """

    def __init__(self, sdf: SDF, scale: Union[float, Array, Distance, Point]):
        self.sdf = sdf
        # Accept both raw values and constraints
        if isinstance(scale, (Distance, Point)):
            self.scale_param = scale
        elif isinstance(scale, (int, float)):
            # Uniform scale - wrap in Distance constraint
            self.scale_param = Distance(value=float(scale), free=False)
        else:
            # Non-uniform scale - wrap in Point constraint
            self.scale_param = Point(value=jnp.asarray(scale), free=False)

        self.is_uniform = isinstance(self.scale_param, Distance)

    def __call__(self, p: Array) -> Array:
        """Evaluate scaled SDF."""
        # Use static functional method
        return Scale.eval(self.sdf, p, self.scale_param.value)

    @staticmethod
    def eval(sdf_fn, p: Array, scale: float | Array) -> Array:
        """Functional evaluation of scaling.

        Args:
            sdf_fn: SDF function to scale
            p: Query point(s)
            scale: Scale factor (uniform) or scale vector (non-uniform)

        Returns:
            Scaled SDF value
        """
        is_uniform = isinstance(scale, (int, float)) or scale.ndim == 0
        if is_uniform:
            # Uniform scaling: divide point by scale, multiply distance by scale
            return sdf_fn(p / scale) * scale
        else:
            # Non-uniform scaling: approximate (not exact SDF)
            return sdf_fn(p / scale)


class Rotate(SDF):
    """Rotate an SDF around an axis.

    Args:
        sdf: The SDF to rotate
        axis: Rotation axis ('x', 'y', 'z') or custom axis vector
        angle: Rotation angle in radians (float or Angle constraint)
    """

    def __init__(self, sdf: SDF, axis: str | Array, angle: Union[float, Angle]):
        self.sdf = sdf
        self.axis = axis

        # Accept both raw values and constraints
        if isinstance(angle, Angle):
            self.angle_param = angle
        else:
            # Wrap raw value in a fixed Angle constraint
            self.angle_param = Angle(value=float(angle), free=False)

        # Compute rotation matrix for class-based API
        if isinstance(axis, str):
            if axis == 'x':
                self.rotation_matrix = self._rotation_matrix_x(self.angle_param.value)
            elif axis == 'y':
                self.rotation_matrix = self._rotation_matrix_y(self.angle_param.value)
            elif axis == 'z':
                self.rotation_matrix = self._rotation_matrix_z(self.angle_param.value)
            else:
                raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', 'z' or provide axis vector")
        else:
            # Custom axis rotation using Rodrigues' formula
            self.rotation_matrix = self._rotation_matrix_axis(jnp.asarray(axis), self.angle_param.value)

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
        # Use static functional method
        return Rotate.eval_z(self.sdf, p, self.angle_param.value)

    @staticmethod
    def eval_z(sdf_fn, p: Array, angle: float) -> Array:
        """Functional evaluation of Z-axis rotation.

        Args:
            sdf_fn: SDF function to rotate
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
        return sdf_fn(p_rotated)


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
