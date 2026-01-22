"""Non-linear deformation transformations for SDFs."""

from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxcad.constraints import Distance
from jaxcad.sdf import SDF


class Twist(SDF):
    """Twist an SDF around an axis.

    Args:
        sdf: The SDF to twist
        axis: Twist axis ('x', 'y', or 'z')
        strength: Twist strength (radians per unit length along axis, float or Distance)
    """

    def __init__(self, sdf: SDF, axis: str = 'z', strength: Union[float, Distance] = 1.0):
        self.sdf = sdf
        self.axis = axis.lower()

        # Accept both raw values and constraints
        if isinstance(strength, Distance):
            self.strength_param = strength
        else:
            # Wrap raw value in a fixed Distance constraint
            self.strength_param = Distance(value=float(strength), free=False)

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")

    def __call__(self, p: Array) -> Array:
        """Evaluate twisted SDF."""
        # Use static functional method (default to Z-axis for now)
        return Twist.eval(self.sdf, p, self.strength_param.value)

    @staticmethod
    def eval(sdf_fn, p: Array, strength: float) -> Array:
        """Functional evaluation of Z-axis twist.

        Args:
            sdf_fn: SDF function to twist
            p: Query point(s)
            strength: Twist strength (radians per unit length along Z)

        Returns:
            Twisted SDF value
        """
        # Twist around Z axis
        z = p[..., 2]
        angle = z * strength
        c, s = jnp.cos(angle), jnp.sin(angle)
        x = p[..., 0] * c + p[..., 1] * s
        y = -p[..., 0] * s + p[..., 1] * c
        p_twisted = jnp.stack([x, y, z], axis=-1)
        return sdf_fn(p_twisted)


class Bend(SDF):
    """Bend an SDF along an axis using polar transformation.

    Args:
        sdf: The SDF to bend
        axis: Bend axis ('x', 'y', or 'z') - the axis to bend along
        strength: Bend strength (curvature factor, float or Distance)
    """

    def __init__(self, sdf: SDF, axis: str = 'z', strength: Union[float, Distance] = 1.0):
        self.sdf = sdf
        self.axis = axis.lower()

        # Accept both raw values and constraints
        if isinstance(strength, Distance):
            self.strength_param = strength
        else:
            # Wrap raw value in a fixed Distance constraint
            self.strength_param = Distance(value=float(strength), free=False)

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")

    def __call__(self, p: Array) -> Array:
        """Evaluate bent SDF."""
        k = self.strength_param.value

        if self.axis == 'z':
            # Bend along Z axis
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            # Convert to polar-like coordinates
            c = jnp.cos(k * z)
            s = jnp.sin(k * z)
            # Inverse bend transformation
            x_bent = c * x - s / k
            z_bent = s * x + c / k - 1.0 / k
            p_bent = jnp.stack([x_bent, y, z_bent], axis=-1)
        elif self.axis == 'y':
            # Bend along Y axis
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            c = jnp.cos(k * y)
            s = jnp.sin(k * y)
            x_bent = c * x - s / k
            y_bent = s * x + c / k - 1.0 / k
            p_bent = jnp.stack([x_bent, y_bent, z], axis=-1)
        else:  # 'x'
            # Bend along X axis
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            c = jnp.cos(k * x)
            s = jnp.sin(k * x)
            y_bent = c * y - s / k
            x_bent = s * y + c / k - 1.0 / k
            p_bent = jnp.stack([x_bent, y_bent, z], axis=-1)

        return self.sdf(p_bent)


class Taper(SDF):
    """Taper an SDF along an axis (linear scale variation).

    Args:
        sdf: The SDF to taper
        axis: Taper axis ('x', 'y', or 'z')
        strength: Taper strength (scale change per unit length, float or Distance)
    """

    def __init__(self, sdf: SDF, axis: str = 'z', strength: Union[float, Distance] = 0.5):
        self.sdf = sdf
        self.axis = axis.lower()

        # Accept both raw values and constraints
        if isinstance(strength, Distance):
            self.strength_param = strength
        else:
            # Wrap raw value in a fixed Distance constraint
            self.strength_param = Distance(value=float(strength), free=False)

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")

    def __call__(self, p: Array) -> Array:
        """Evaluate tapered SDF."""
        # Use static functional method (default to Z-axis)
        return Taper.eval(self.sdf, p, self.strength_param.value)

    @staticmethod
    def eval(sdf_fn, p: Array, strength: float) -> Array:
        """Functional evaluation of Z-axis taper.

        Args:
            sdf_fn: SDF function to taper
            p: Query point(s)
            strength: Taper strength (scale change per unit length along Z)

        Returns:
            Tapered SDF value
        """
        z = p[..., 2]
        scale = 1.0 + z * strength
        x = p[..., 0] / scale
        y = p[..., 1] / scale
        p_tapered = jnp.stack([x, y, z], axis=-1)
        return sdf_fn(p_tapered)


class RepeatInfinite(SDF):
    """Infinitely repeat an SDF in space.

    Args:
        sdf: The SDF to repeat
        spacing: Spacing between repetitions (per axis)
    """

    def __init__(self, sdf: SDF, spacing: Array):
        self.sdf = sdf
        self.spacing = jnp.asarray(spacing)

    def __call__(self, p: Array) -> Array:
        """Evaluate repeated SDF."""
        # Map to infinite grid using modulo
        p_repeated = jnp.mod(p + self.spacing * 0.5, self.spacing) - self.spacing * 0.5
        return self.sdf(p_repeated)


class RepeatFinite(SDF):
    """Repeat an SDF a finite number of times.

    Args:
        sdf: The SDF to repeat
        spacing: Spacing between repetitions (per axis)
        count: Number of repetitions in each direction (x, y, z)
    """

    def __init__(self, sdf: SDF, spacing: Array, count: Array):
        self.sdf = sdf
        self.spacing = jnp.asarray(spacing)
        self.count = jnp.asarray(count, dtype=int)

    def __call__(self, p: Array) -> Array:
        """Evaluate finite repeated SDF."""
        # Clamp repetition indices to finite range
        indices = jnp.round(p / self.spacing)
        clamped_indices = jnp.clip(indices, -self.count / 2, self.count / 2 - 1)
        p_repeated = p - clamped_indices * self.spacing
        return self.sdf(p_repeated)


class Mirror(SDF):
    """Mirror an SDF across a plane.

    Args:
        sdf: The SDF to mirror
        axis: Axis perpendicular to mirror plane ('x', 'y', 'z')
        offset: Offset of mirror plane from origin
    """

    def __init__(self, sdf: SDF, axis: str = 'x', offset: float = 0.0):
        self.sdf = sdf
        self.axis = axis.lower()
        self.offset = offset

        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'")

        self.axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.axis]

    def __call__(self, p: Array) -> Array:
        """Evaluate mirrored SDF."""
        # Mirror by taking absolute value of coordinate
        p_mirrored = p.at[..., self.axis_idx].set(
            jnp.abs(p[..., self.axis_idx] - self.offset) + self.offset
        )
        return self.sdf(p_mirrored)


# Convenience functions
def twist(sdf: SDF, axis: str = 'z', strength: float = 1.0) -> Twist:
    """Twist an SDF around an axis."""
    return Twist(sdf, axis, strength)


def bend(sdf: SDF, axis: str = 'z', strength: float = 1.0) -> Bend:
    """Bend an SDF along an axis."""
    return Bend(sdf, axis, strength)


def taper(sdf: SDF, axis: str = 'z', strength: float = 0.5) -> Taper:
    """Taper an SDF along an axis."""
    return Taper(sdf, axis, strength)


def repeat_infinite(sdf: SDF, spacing: Array) -> RepeatInfinite:
    """Infinitely repeat an SDF in space."""
    return RepeatInfinite(sdf, spacing)


def repeat_finite(sdf: SDF, spacing: Array, count: Array) -> RepeatFinite:
    """Repeat an SDF a finite number of times."""
    return RepeatFinite(sdf, spacing, count)


def mirror(sdf: SDF, axis: str = 'x', offset: float = 0.0) -> Mirror:
    """Mirror an SDF across a plane."""
    return Mirror(sdf, axis, offset)
