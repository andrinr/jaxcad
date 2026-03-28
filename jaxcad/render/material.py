"""Material properties for SDF primitives."""

from __future__ import annotations

from jax import Array

from jaxcad.fluent import Fluent
from jaxcad.geometry.parameters import Vector


class Material(Fluent):
    """Material properties for physically-inspired shading.

    Properties are stored as Parameter objects in ``self.params`` (fixed by
    default), consistent with how SDF primitives handle their parameters.
    ``extract_parameters`` can therefore discover and—if marked free—optimise
    material properties alongside geometry.

    Args:
        color: RGB surface color, values in [0, 1].
        roughness: Surface roughness, 0 = mirror, 1 = fully diffuse.
        metallic: Metallic factor; 0 = dielectric, 1 = metallic specular.
        opacity: Opacity; 0 = fully transparent, 1 = fully opaque.
        ior: Index of refraction; 1.0 = air, 1.33 = water, 1.5 = glass.
    """

    def __init__(
        self,
        color=None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        opacity: float = 1.0,
        ior: float = 1.0,
    ):
        self.params = {
            "color": color if color is not None else [1.0, 1.0, 1.0],
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
            "ior": ior,
        }
        self._cast_params()  # convert raw values → Scalar / Vector

    def children(self) -> list:
        return []

    def as_dict(self) -> dict:
        """Return material properties as a JAX-pytree-compatible dict of arrays."""
        return {k: (v.xyz if isinstance(v, Vector) else v.value) for k, v in self.params.items()}

    @staticmethod
    def blend(m1: dict, m2: dict, t: Array) -> dict:
        """Linearly interpolate between two material dicts.

        Args:
            m1: First material dict.
            m2: Second material dict.
            t: Blend factor; t=1 returns m1, t=0 returns m2.
        """
        return {k: m2[k] * (1.0 - t) + m1[k] * t for k in m1}
