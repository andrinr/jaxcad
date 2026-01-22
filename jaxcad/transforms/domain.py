"""Domain manipulation operations (symmetry, repetition, etc)."""

from typing import Optional, Union

import jax.numpy as jnp
from jax import Array

from jaxcad.sdf import SDF


class Symmetry(SDF):
    """Apply symmetry to an SDF across specified axes.

    Args:
        sdf: The SDF to make symmetric
        axis: Axis/axes to mirror across ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz')
    """

    def __init__(self, sdf: SDF, axis: str = "x"):
        self.sdf = sdf
        self.axis = axis.lower()

    def __call__(self, p: Array) -> Array:
        """Apply symmetry by taking absolute value of coordinates"""
        p_sym = p.copy() if isinstance(p, jnp.ndarray) else jnp.array(p)

        if "x" in self.axis:
            p_sym = p_sym.at[..., 0].set(jnp.abs(p[..., 0]))
        if "y" in self.axis:
            p_sym = p_sym.at[..., 1].set(jnp.abs(p[..., 1]))
        if "z" in self.axis and p.shape[-1] > 2:
            p_sym = p_sym.at[..., 2].set(jnp.abs(p[..., 2]))

        return self.sdf(p_sym)


class InfiniteRepetition(SDF):
    """Repeat an SDF infinitely in space.

    Args:
        sdf: The SDF to repeat
        spacing: Spacing between repetitions (scalar or array)
    """

    def __init__(self, sdf: SDF, spacing: Union[float, Array]):
        self.sdf = sdf
        self.spacing = jnp.asarray(spacing)

    def __call__(self, p: Array) -> Array:
        """Apply infinite repetition using modulo"""
        # Center each cell at origin using mod
        q = jnp.mod(p + 0.5 * self.spacing, self.spacing) - 0.5 * self.spacing
        return self.sdf(q)


class FiniteRepetition(SDF):
    """Repeat an SDF a finite number of times.

    Args:
        sdf: The SDF to repeat
        spacing: Spacing between repetitions (scalar or array)
        count: Number of repetitions in each dimension (scalar or array)
    """

    def __init__(self, sdf: SDF, spacing: Union[float, Array], count: Union[int, Array]):
        self.sdf = sdf
        self.spacing = jnp.asarray(spacing)
        self.count = jnp.asarray(count)

    def __call__(self, p: Array) -> Array:
        """Apply finite repetition with clamping"""
        # Clamp to valid repetition range
        q = p - self.spacing * jnp.clip(
            jnp.round(p / self.spacing),
            -self.count,
            self.count
        )
        return self.sdf(q)


class Elongation(SDF):
    """Elongate/stretch an SDF along specified axes.

    Args:
        sdf: The SDF to elongate
        h: Elongation amount per axis
    """

    def __init__(self, sdf: SDF, h: Union[float, Array]):
        self.sdf = sdf
        self.h = jnp.asarray(h)

    def __call__(self, p: Array) -> Array:
        """Apply elongation by clamping coordinates"""
        q = p - jnp.clip(p, -self.h, self.h)
        return self.sdf(q)


class Rounding(SDF):
    """Round/fillet an SDF by a specified radius.

    Args:
        sdf: The SDF to round
        radius: Rounding radius
    """

    def __init__(self, sdf: SDF, radius: float):
        self.sdf = sdf
        self.radius = radius

    def __call__(self, p: Array) -> Array:
        """Apply rounding by subtracting radius"""
        return self.sdf(p) - self.radius


class Onion(SDF):
    """Create a shell/thickness from an SDF (onion operation).

    Args:
        sdf: The SDF to convert to shell
        thickness: Wall thickness
    """

    def __init__(self, sdf: SDF, thickness: float):
        self.sdf = sdf
        self.thickness = thickness

    def __call__(self, p: Array) -> Array:
        """Apply onion by taking absolute distance minus half thickness"""
        return jnp.abs(self.sdf(p)) - self.thickness


# Convenience functions
def symmetry(sdf: SDF, axis: str = "x") -> Symmetry:
    """Apply symmetry to an SDF.

    Args:
        sdf: The SDF to make symmetric
        axis: Axis/axes to mirror across

    Returns:
        Symmetric SDF
    """
    return Symmetry(sdf, axis)


def repeat_infinite(sdf: SDF, spacing: Union[float, Array]) -> InfiniteRepetition:
    """Repeat an SDF infinitely.

    Args:
        sdf: The SDF to repeat
        spacing: Spacing between repetitions

    Returns:
        Infinitely repeated SDF
    """
    return InfiniteRepetition(sdf, spacing)


def repeat_finite(
    sdf: SDF, spacing: Union[float, Array], count: Union[int, Array]
) -> FiniteRepetition:
    """Repeat an SDF a finite number of times.

    Args:
        sdf: The SDF to repeat
        spacing: Spacing between repetitions
        count: Number of repetitions

    Returns:
        Finitely repeated SDF
    """
    return FiniteRepetition(sdf, spacing, count)


def elongate(sdf: SDF, h: Union[float, Array]) -> Elongation:
    """Elongate/stretch an SDF.

    Args:
        sdf: The SDF to elongate
        h: Elongation amount

    Returns:
        Elongated SDF
    """
    return Elongation(sdf, h)


def round_sdf(sdf: SDF, radius: float) -> Rounding:
    """Round/fillet an SDF.

    Args:
        sdf: The SDF to round
        radius: Rounding radius

    Returns:
        Rounded SDF
    """
    return Rounding(sdf, radius)


def onion(sdf: SDF, thickness: float) -> Onion:
    """Create a shell from an SDF.

    Args:
        sdf: The SDF to convert to shell
        thickness: Wall thickness

    Returns:
        Shell SDF
    """
    return Onion(sdf, thickness)
