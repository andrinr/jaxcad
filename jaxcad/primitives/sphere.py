"""Sphere primitive."""

from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.parameters import Scalar
from jaxcad.primitives.base import Primitive


class Sphere(Primitive):
    """Sphere centered at origin with given radius.

    This is a thin wrapper that provides:
    - Fluent API: sphere.translate().rotate()
    - Operator overloading: sphere | box
    - Parameter storage for compilation

    The actual SDF computation is in the static sdf() method.

    Args:
        radius: Sphere radius (float or Scalar parameter)

    Example:
        # Direct creation
        sphere = Sphere(radius=1.0)

        # With parameter for optimization
        from jaxcad.parameters import Scalar
        radius = Scalar(value=1.0, free=True, name='radius')
        sphere = Sphere(radius=radius)

        # Fluent API
        shape = Sphere(1.0).translate([1, 0, 0]).scale(2.0)

        # Direct pure function call
        distance = Sphere.sdf(point, radius=1.0)
    """

    def __init__(self, radius: Union[float, Scalar]):
        self.params = {'radius': radius}

    @staticmethod
    def sdf(p: Array, radius: float) -> Array:
        """Pure SDF function for sphere.

        This is the source of truth for sphere SDF computation.
        Used directly during JAX compilation and tracing.

        Args:
            p: Point(s) to evaluate, shape (..., 3)
            radius: Sphere radius

        Returns:
            Signed distance: ||p|| - radius
        """
        return jnp.linalg.norm(p, axis=-1) - radius

    def __call__(self, p: Array) -> Array:
        """Evaluate SDF at point(s) p.

        Delegates to the pure function with stored parameters.
        """
        return Sphere.sdf(p, self.params['radius'].value)

    def to_functional(self):
        """Return pure function for compilation.

        Returns:
            Pure function: Sphere.sdf(p, radius) -> sdf_value
        """
        return Sphere.sdf
