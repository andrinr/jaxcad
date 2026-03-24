"""Optimizable parameters for gradient-based shape optimization.

This module provides parameter types that can be marked as free (optimizable)
or fixed during gradient descent. Parameters are JAX pytrees and can be
automatically extracted and optimized.

Types:
- Parameter: Base class for all parameters
- Scalar: Single values (radius, distance, angle, scale, etc.)
- Vector: 3D vectors (positions, offsets, directions)

Helpers:
- as_parameter: Auto-convert raw values to Parameter objects

For constraint systems (distances, angles, parallel, etc.), see constraints.py (future).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union, overload

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from jaxcad.constraints.base import Constraint


@dataclass
class Parameter:
    """Base class for optimizable parameters.

    Parameters are FIXED by default. Mark as free=True to make them optimizable.

    Args:
        value: Initial/current value
        free: Whether this parameter is free (optimizable). Default: False
        name: Optional name for debugging
        bounds: Optional (min, max) bounds for optimization
        _constraints: Internal list of constraints that reference this parameter
    """

    value: Union[float, Array]
    free: bool = False  # FIXED by default
    name: Optional[str] = None
    bounds: Optional[tuple] = None
    _constraints: list["Constraint"] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self):
        self.value = jnp.asarray(self.value)

    def add_constraint(self, constraint: "Constraint") -> None:
        """Register a constraint that references this parameter.

        Args:
            constraint: Constraint to register
        """
        # Use object identity to avoid JAX array comparison issues
        if not any(c is constraint for c in self._constraints):
            self._constraints.append(constraint)

    def get_constraints(self) -> list["Constraint"]:
        """Get all constraints that reference this parameter.

        Returns:
            List of Constraint objects
        """
        return self._constraints.copy()

    @property
    def fixed(self) -> bool:
        """Whether this parameter is fixed (non-optimizable)."""
        return not self.free

    def extract_value(self) -> Union[float, Array]:
        """Extract raw numeric value for computation.

        Base implementation returns .value.
        Subclasses like Vector override to return .xyz.

        Returns:
            Raw numeric value (float or Array)
        """
        return self.value

    def __repr__(self):
        status = "FIXED" if self.fixed else "FREE"
        if self.name:
            return f"{self.__class__.__name__}({self.name}={self.value}, {status})"
        return f"{self.__class__.__name__}({self.value}, {status})"

    def tree_flatten(self):
        """Flatten for JAX pytree."""
        if self.free:
            children = (self.value,)
            aux_data = {"free": True, "name": self.name, "bounds": self.bounds}
        else:
            children = ()
            aux_data = {
                "free": False,
                "name": self.name,
                "bounds": self.bounds,
                "value": self.value,
            }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree."""
        value = children[0] if aux_data["free"] else aux_data["value"]
        return cls(
            value=value, free=aux_data["free"], name=aux_data["name"], bounds=aux_data["bounds"]
        )


@dataclass
class Scalar(Parameter):
    """Scalar parameter (radius, distance, angle, scale, strength, etc.).

    Example:
        ```python
        radius = Scalar(value=1.0, free=True, name='radius')
        angle = Scalar(value=jnp.pi / 4, free=False, name='rotation')
        ```
    """

    def __post_init__(self):
        super().__post_init__()
        # Ensure scalar shape
        if self.value.shape != ():
            self.value = jnp.asarray(float(self.value))


@dataclass
class Vector(Parameter):
    """3D vector parameter (position, offset, direction, size).

    Example:
        ```python
        position = Vector(value=[1.0, 2.0, 3.0], free=True, name='pos')
        offset = Vector(value=[0, 0, 0], free=False)
        ```
    """

    def __post_init__(self):
        super().__post_init__()
        if self.value.shape != (3,):
            raise ValueError(f"Vector must be 3D, got shape {self.value.shape}")
        if not jnp.issubdtype(self.value.dtype, jnp.floating):
            self.value = self.value.astype(jnp.float32)

    def norm(self) -> float:
        """Compute Euclidean norm."""
        return jnp.linalg.norm(self.value)

    def normalize(self) -> "Vector":
        """Return a normalized Vector (unit length)."""
        norm = self.norm()
        if norm < 1e-8:
            raise ValueError("Cannot normalize zero-length vector.")
        return Vector(value=self.value / norm, free=self.free, name=self.name, bounds=self.bounds)

    @property
    def xyz(self) -> Array:
        """Alias for .value (3D coordinates)."""
        return self.value


# Register as JAX pytrees
jax.tree_util.register_pytree_node(Parameter, Parameter.tree_flatten, Parameter.tree_unflatten)

jax.tree_util.register_pytree_node(Scalar, Scalar.tree_flatten, Scalar.tree_unflatten)

jax.tree_util.register_pytree_node(Vector, Vector.tree_flatten, Vector.tree_unflatten)


@overload
def as_parameter(value: Union[float, Scalar], name: Optional[str] = None) -> Scalar: ...


@overload
def as_parameter(value: Union[Array, Vector], name: Optional[str] = None) -> Vector: ...


def as_parameter(
    value: Union[float, Array, Scalar, Vector], name: Optional[str] = None
) -> Union[Scalar, Vector]:
    """Convert a raw value to a Parameter object automatically.

    If the value is already a Parameter, return it as-is.
    Otherwise, wrap it in the appropriate Parameter type with free=False.

    Args:
        value: Value to convert
        name: Optional name for the parameter

    Returns:
        Parameter object

    Examples:
        >>> as_parameter(1.5)
        Scalar(value=1.5, free=False)

        >>> as_parameter([1, 2, 3])
        Vector(value=[1, 2, 3], free=False)

        >>> existing = Scalar(value=2.0, free=True, name='radius')
        >>> as_parameter(existing)
        Scalar(value=2.0, free=True, name='radius')  # unchanged
    """
    # If already a Parameter, return as-is
    if isinstance(value, Parameter):
        return value

    # Convert to jax array first
    arr = jnp.asarray(value)

    # Scalar (0-d array or single value)
    if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] == 1):
        return Scalar(value=arr if arr.ndim == 0 else arr[0], free=False, name=name)

    # Vector (3D array)
    elif arr.shape == (3,):
        return Vector(value=arr, free=False, name=name)

    else:
        raise ValueError(
            f"Cannot auto-convert value with shape {arr.shape} to Parameter. "
            f"Use Scalar for scalars or Vector for 3D vectors."
        )
