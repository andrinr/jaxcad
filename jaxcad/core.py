"""Core data structures for differentiable CAD."""

from typing import NamedTuple

import jax.numpy as jnp


class Solid(NamedTuple):
    """Represents a 3D solid as a mesh.

    Attributes:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (M, 3) containing vertex indices for triangular faces
    """

    vertices: jnp.ndarray  # (N, 3)
    faces: jnp.ndarray  # (M, 3) indices into vertices

    def __repr__(self):
        return f"Solid(vertices={self.vertices.shape}, faces={self.faces.shape})"
