"""Optimizable parameters for gradient-based shape optimization.

This module provides parameter types that can be marked as free (optimizable)
or fixed during gradient descent. Parameters are JAX pytrees and can be
automatically extracted and optimized.

Types:
- Parameter: Base class for all parameters
- Scalar: Single values (radius, distance, angle, scale, etc.)
- Vector: 3D/4D vectors (positions, offsets, directions) with homogeneous coordinates

For constraint systems (distances, angles, parallel, etc.), see constraints.py (future).
"""

from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class Parameter:
    """Base class for optimizable parameters.

    Parameters are FIXED by default. Mark as free=True to make them optimizable.

    Args:
        value: Initial/current value
        free: Whether this parameter is free (optimizable). Default: False
        name: Optional name for debugging
        bounds: Optional (min, max) bounds for optimization
    """
    value: Union[float, Array]
    free: bool = False  # FIXED by default
    name: Optional[str] = None
    bounds: Optional[tuple] = None

    def __post_init__(self):
        self.value = jnp.asarray(self.value)

    @property
    def fixed(self) -> bool:
        """Whether this parameter is fixed (non-optimizable)."""
        return not self.free

    def __repr__(self):
        status = "FIXED" if self.fixed else "FREE"
        if self.name:
            return f"{self.__class__.__name__}({self.name}={self.value}, {status})"
        return f"{self.__class__.__name__}({self.value}, {status})"

    def tree_flatten(self):
        """Flatten for JAX pytree."""
        if self.free:
            children = (self.value,)
            aux_data = {'free': True, 'name': self.name, 'bounds': self.bounds}
        else:
            children = ()
            aux_data = {'free': False, 'name': self.name, 'bounds': self.bounds, 'value': self.value}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree."""
        if aux_data['free']:
            value = children[0]
        else:
            value = aux_data['value']
        return cls(value=value, free=aux_data['free'], name=aux_data['name'], bounds=aux_data['bounds'])


@dataclass
class Scalar(Parameter):
    """Scalar parameter (radius, distance, angle, scale, strength, etc.).

    Example:
        radius = Scalar(value=1.0, free=True, name='radius')
        angle = Scalar(value=jnp.pi/4, free=False, name='rotation')
    """

    def __post_init__(self):
        super().__post_init__()
        # Ensure scalar shape
        if self.value.shape != ():
            self.value = jnp.asarray(float(self.value))


@dataclass
class Vector(Parameter):
    """3D or 4D vector parameter (position, offset, direction, size).

    Supports both 3D vectors [x, y, z] and 4D homogeneous coordinates [x, y, z, w].
    When initialized with 3 components, automatically extends to 4D with w=1.0 for points.

    In homogeneous coordinates:
    - Points (w=1): affected by translation
    - Directions (w=0): not affected by translation

    Example:
        position = Vector(value=[1.0, 2.0, 3.0], free=True, name='pos')  # becomes [1,2,3,1]
        direction = Vector(value=[0, 1, 0, 0], free=False)  # pure direction
        offset = Vector(value=[0, 0, 0], free=False)
    """

    def __post_init__(self):
        super().__post_init__()
        # Accept both 3D and 4D vectors
        if self.value.shape == (3,):
            # Extend 3D to homogeneous coordinates with w=1 (point)
            self.value = jnp.append(self.value, 1.0)
        elif self.value.shape == (4,):
            # Already in homogeneous coordinates
            pass
        else:
            raise ValueError(f"Vector must be 3D or 4D, got shape {self.value.shape}")

    @property
    def xyz(self) -> Array:
        """Get 3D cartesian coordinates (x, y, z)."""
        return self.value[:3]

    @property
    def w(self) -> float:
        """Get homogeneous coordinate w."""
        return self.value[3]

    @property
    def is_point(self) -> bool:
        """True if this is a point (w=1), False if direction (w=0)."""
        return jnp.abs(self.w - 1.0) < 1e-6

    @property
    def is_direction(self) -> bool:
        """True if this is a direction vector (w=0)."""
        return jnp.abs(self.w) < 1e-6


# Backwards compatibility aliases
Distance = Scalar  # Distance is just a Scalar with semantic meaning
Angle = Scalar     # Angle is just a Scalar with semantic meaning
Point = Vector     # Point is an alias for Vector for backwards compatibility


# Register as JAX pytrees
jax.tree_util.register_pytree_node(
    Parameter,
    Parameter.tree_flatten,
    Parameter.tree_unflatten
)

jax.tree_util.register_pytree_node(
    Scalar,
    Scalar.tree_flatten,
    Scalar.tree_unflatten
)

jax.tree_util.register_pytree_node(
    Vector,
    Vector.tree_flatten,
    Vector.tree_unflatten
)
