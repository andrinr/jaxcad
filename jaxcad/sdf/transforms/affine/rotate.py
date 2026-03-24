"""Rotate transformation for SDFs."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector, Scalar
from jaxcad.sdf import SDF
from jaxcad.sdf.transforms.base import Transform


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
