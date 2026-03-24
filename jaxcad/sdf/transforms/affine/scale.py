"""Scale transformation for SDFs."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from jaxcad.geometry.parameters import Vector
from jaxcad.sdf import SDF
from jaxcad.sdf.transforms.base import Transform


class Scale(Transform):
    """Scale an SDF component-wise.

    Note: Non-uniform scaling doesn't produce exact SDFs. For uniform scaling,
    we can divide the distance by the scale factor to maintain correctness.

    Args:
        sdf: The SDF to scale
        scale: Per-axis scale as Array [sx, sy, sz], Vector parameter, or float for uniform scaling
    """

    def __init__(self, sdf: SDF, scale: float | Array | Vector):
        self.sdf = sdf
        # Convert scalar to uniform 3D scale vector before auto-cast
        if isinstance(scale, (int, float)):
            scale = jnp.array([scale, scale, scale])
        self.params = {"scale": scale}

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
        return Scale.sdf(self.sdf, p, self.params["scale"].xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Scale.sdf
