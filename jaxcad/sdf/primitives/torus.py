"""Torus primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.sdf.primitives.base import Primitive


class Torus(Primitive):
    """Torus in XY plane centered at origin.

    Args:
        major_radius: Distance from origin to tube center (float or Scalar)
        minor_radius: Tube radius (float or Scalar)
    """

    def __init__(self, major_radius: Union[float, Scalar], minor_radius: Union[float, Scalar]):
        self.params = {'major_radius': major_radius, 'minor_radius': minor_radius}

    @staticmethod
    def sdf(p: Array, major_radius: float, minor_radius: float) -> Array:
        """Pure SDF function for torus.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            major_radius: Distance from origin to tube center
            minor_radius: Tube radius

        Returns:
            Signed distance to torus surface
        """
        q_xy = jnp.linalg.norm(p[..., :2], axis=-1) - major_radius
        q = jnp.stack([q_xy, p[..., 2]], axis=-1)
        return jnp.linalg.norm(q, axis=-1) - minor_radius

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p."""
        return Torus.sdf(p, self.params['major_radius'].value, self.params['minor_radius'].value)

    def to_functional(self):
        """Return pure function for compilation."""
        return Torus.sdf
