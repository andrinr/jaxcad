"""Boolean operations (CSG) for SDFs."""

from jaxcad.sdf.boolean.base import BooleanOp
from jaxcad.sdf.boolean.difference import Difference
from jaxcad.sdf.boolean.intersection import Intersection
from jaxcad.sdf.boolean.smooth import smooth_max, smooth_min
from jaxcad.sdf.boolean.union import Union
from jaxcad.sdf.boolean.xor import Xor
from jaxcad.sdf.base import SDF

__all__ = [
    "BooleanOp",
    "Union",
    "Intersection",
    "Difference",
    "Xor",
    "smooth_min",
    "smooth_max",
    "union",
    "intersection",
    "difference",
    "xor",
]


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


def xor(sdf1: SDF, sdf2: SDF) -> Xor:
    """Create XOR (symmetric difference) of two SDFs.

    Args:
        sdf1: First SDF
        sdf2: Second SDF

    Returns:
        Xor SDF
    """
    return Xor(sdf1, sdf2)
