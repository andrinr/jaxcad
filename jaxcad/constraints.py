"""Parameter system for optimization control.

Two types of parameters:
- Scalar: single values (radius, distance, angle, scale, etc.)
- Point: 3D vectors (positions, offsets, directions)

Parameters can be marked as fixed or free to control which values
can be optimized during gradient descent.
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
class Point(Parameter):
    """3D point/vector parameter (position, offset, direction, size).

    Example:
        position = Point(value=[1.0, 2.0, 3.0], free=True, name='pos')
        offset = Point(value=[0, 0, 0], free=False)
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.value.shape == (3,), f"Point must be 3D, got shape {self.value.shape}"


# Backwards compatibility aliases
Distance = Scalar  # Distance is just a Scalar with semantic meaning
Angle = Scalar     # Angle is just a Scalar with semantic meaning


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
    Point,
    Point.tree_flatten,
    Point.tree_unflatten
)
