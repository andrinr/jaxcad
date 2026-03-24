"""Translate transformation for SDFs."""

from __future__ import annotations

from typing import Union

from jax import Array

from jaxcad.geometry.parameters import Vector
from jaxcad.sdf import SDF
from jaxcad.sdf.transforms.base import Transform


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
        self.params = {'offset': offset}

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
        return Translate.sdf(self.sdf, p, self.params['offset'].xyz)

    def to_functional(self):
        """Return pure function for compilation."""
        return Translate.sdf
