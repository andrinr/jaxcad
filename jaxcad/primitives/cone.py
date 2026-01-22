"""Cone primitive."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Cone(SDF):
    """Cone along Z-axis centered at origin.

    Args:
        radius: Base radius
        height: Cone height (apex at +height/2, base at -height/2)
    """

    def __init__(self, radius: float, height: float):
        self.radius = radius
        self.height = height
        # Cone angle: sin and cos
        h = jnp.sqrt(radius * radius + height * height)
        self.sin_angle = radius / h
        self.cos_angle = height / h

    def __call__(self, p: Array) -> Array:
        """SDF for cone centered at origin."""
        # Shift z so apex is at top, base at bottom
        z = p[..., 2] + self.height / 2.0

        # Distance in XY plane
        r = jnp.linalg.norm(p[..., :2], axis=-1)

        # Create 2D point (r, z)
        q = jnp.stack([r, z], axis=-1)

        # Cone direction vector (pointing from apex to base edge)
        c = jnp.array([self.sin_angle, -self.cos_angle])

        # Project q onto cone direction
        dot_qc = q[..., 0] * c[0] + q[..., 1] * c[1]

        # Clamped projection (only valid between apex and base)
        t = jnp.clip(dot_qc, 0.0, self.height)

        # Closest point on cone spine
        closest = jnp.stack([c[0] * t, c[1] * t], axis=-1)

        # Distance to cone surface
        d = jnp.linalg.norm(q - closest, axis=-1)

        # Inside/outside sign
        inside = (q[..., 0] * c[1] - q[..., 1] * c[0]) < 0

        # Cap the bottom
        d_base = jnp.where(z < 0, jnp.maximum(d, -z), d)

        return jnp.where(inside, -d_base, d_base)
