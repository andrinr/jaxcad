"""Material properties for SDF primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array


@dataclass
class Material:
    """Material properties for physically-inspired shading.

    Args:
        color: RGB surface color, values in [0, 1].
        roughness: Surface roughness, 0 = mirror, 1 = fully diffuse.
        metallic: Metallic factor; 0 = dielectric (white specular), 1 = metallic (tinted specular).
        opacity: Opacity; 0 = fully transparent, 1 = fully opaque.
        ior: Index of refraction; 1.0 = air, 1.33 = water, 1.5 = glass.
    """

    color: list | Array = field(default_factory=lambda: [1.0, 1.0, 1.0])
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    ior: float = 1.0  # index of refraction; 1.0 = air, 1.33 = water, 1.5 = glass

    def __post_init__(self):
        self.color = jnp.asarray(self.color, dtype=float)
        self.roughness = jnp.asarray(self.roughness, dtype=float)
        self.metallic = jnp.asarray(self.metallic, dtype=float)
        self.opacity = jnp.asarray(self.opacity, dtype=float)
        self.ior = jnp.asarray(self.ior, dtype=float)

    def as_dict(self) -> dict:
        """Return material properties as a JAX-pytree-compatible dict."""
        return {
            "color": self.color,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "opacity": self.opacity,
            "ior": self.ior,
        }

    @staticmethod
    def blend(m1: dict, m2: dict, t: Array) -> dict:
        """Linearly interpolate between two material dicts.

        Args:
            m1: First material dict.
            m2: Second material dict.
            t: Blend factor; t=1 returns m1, t=0 returns m2.

        Returns:
            Blended material dict.
        """
        return {k: m2[k] * (1.0 - t) + m1[k] * t for k in m1}
