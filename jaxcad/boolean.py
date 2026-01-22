"""Boolean operations (CSG) for SDFs."""

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


def smooth_min(a: Array, b: Array, k: float = 0.1) -> Array:
    """Smooth minimum function for blending SDFs (Inigo Quilez's version).

    Args:
        a: First SDF value(s)
        b: Second SDF value(s)
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended minimum value
    """
    k_scaled = k * 4.0
    h = jnp.maximum(k_scaled - jnp.abs(a - b), 0.0)
    return jnp.minimum(a, b) - h * h * 0.25 / k_scaled


def smooth_max(a: Array, b: Array, k: float = 0.1) -> Array:
    """Smooth maximum function for blending SDFs.

    Args:
        a: First SDF value(s)
        b: Second SDF value(s)
        k: Smoothness parameter (larger = smoother blend)

    Returns:
        Smoothly blended maximum value
    """
    return -smooth_min(-a, -b, k)


class Union(SDF):
    """Union of two SDFs (combines both shapes).

    Uses smooth minimum for differentiable blending at the intersection.

    Args:
        sdf1: First SDF
        sdf2: Second SDF
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, sdf1: SDF, sdf2: SDF, smoothness: float = 0.1):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.smoothness = smoothness

    def __call__(self, p: Array) -> Array:
        """Union: min(d1, d2) with smooth blending"""
        d1 = self.sdf1(p)
        d2 = self.sdf2(p)
        if self.smoothness > 0:
            return smooth_min(d1, d2, self.smoothness)
        return jnp.minimum(d1, d2)


class Intersection(SDF):
    """Intersection of two SDFs (only overlapping region).

    Uses smooth maximum for differentiable blending.

    Args:
        sdf1: First SDF
        sdf2: Second SDF
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, sdf1: SDF, sdf2: SDF, smoothness: float = 0.1):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.smoothness = smoothness

    def __call__(self, p: Array) -> Array:
        """Intersection: max(d1, d2) with smooth blending"""
        d1 = self.sdf1(p)
        d2 = self.sdf2(p)
        if self.smoothness > 0:
            return smooth_max(d1, d2, self.smoothness)
        return jnp.maximum(d1, d2)


class Difference(SDF):
    """Difference of two SDFs (subtract second from first).

    Uses smooth maximum for differentiable blending.

    Args:
        sdf1: Base SDF
        sdf2: SDF to subtract
        smoothness: Blend radius (0 = sharp, >0 = smooth)
    """

    def __init__(self, sdf1: SDF, sdf2: SDF, smoothness: float = 0.1):
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.smoothness = smoothness

    def __call__(self, p: Array) -> Array:
        """Difference: max(d1, -d2) with smooth blending"""
        d1 = self.sdf1(p)
        d2 = self.sdf2(p)
        if self.smoothness > 0:
            return smooth_max(d1, -d2, self.smoothness)
        return jnp.maximum(d1, -d2)


# Convenience functions for functional API
def union(sdf1: SDF, sdf2: SDF, smoothness: float = 0.1) -> Union:
    """Create union of two SDFs.

    Args:
        sdf1: First SDF
        sdf2: Second SDF
        smoothness: Blend radius

    Returns:
        Union SDF
    """
    return Union(sdf1, sdf2, smoothness)


def intersection(sdf1: SDF, sdf2: SDF, smoothness: float = 0.1) -> Intersection:
    """Create intersection of two SDFs.

    Args:
        sdf1: First SDF
        sdf2: Second SDF
        smoothness: Blend radius

    Returns:
        Intersection SDF
    """
    return Intersection(sdf1, sdf2, smoothness)


def difference(sdf1: SDF, sdf2: SDF, smoothness: float = 0.1) -> Difference:
    """Create difference of two SDFs.

    Args:
        sdf1: Base SDF
        sdf2: SDF to subtract
        smoothness: Blend radius

    Returns:
        Difference SDF
    """
    return Difference(sdf1, sdf2, smoothness)


class Xor(SDF):
    """XOR of two SDFs (symmetric difference).

    Args:
        sdf1: First SDF
        sdf2: Second SDF
    """

    def __init__(self, sdf1: SDF, sdf2: SDF):
        self.sdf1 = sdf1
        self.sdf2 = sdf2

    def __call__(self, p: Array) -> Array:
        """XOR: max(min(d1, d2), -max(d1, d2))"""
        d1 = self.sdf1(p)
        d2 = self.sdf2(p)
        return jnp.maximum(jnp.minimum(d1, d2), -jnp.maximum(d1, d2))


def xor(sdf1: SDF, sdf2: SDF) -> Xor:
    """Create XOR (symmetric difference) of two SDFs.

    Args:
        sdf1: First SDF
        sdf2: Second SDF

    Returns:
        Xor SDF
    """
    return Xor(sdf1, sdf2)
