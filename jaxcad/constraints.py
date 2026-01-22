"""Constraint-based parametric modeling for JaxCAD.

Constraints allow you to define geometric relationships that can be
optimized via automatic differentiation.
"""

from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp
from jax import Array


@dataclass
class Parameter:
    """A parameter that can be free (optimizable) or fixed (static).

    Parameters are FREE by default (optimizable). Mark as fixed=True to
    make them static constants.

    Args:
        value: Initial/current value
        fixed: Whether this parameter is fixed (non-optimizable). Default: False
        name: Optional name for debugging
        bounds: Optional (min, max) bounds for optimization
    """
    value: Union[float, Array]
    fixed: bool = False  # Changed: now defaults to FREE
    name: Optional[str] = None
    bounds: Optional[tuple] = None

    def __post_init__(self):
        self.value = jnp.asarray(self.value)

    @property
    def free(self) -> bool:
        """Whether this parameter is free (optimizable)."""
        return not self.fixed

    def __repr__(self):
        status = "FIXED" if self.fixed else "FREE"
        if self.name:
            return f"Parameter({self.name}={self.value}, {status})"
        return f"Parameter({self.value}, {status})"


@dataclass
class Point(Parameter):
    """A 3D point constraint (FREE by default).

    Args:
        value: 3D coordinates [x, y, z]
        fixed: Whether this point is fixed (default: False = free)
        name: Optional name for the point
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.value.shape == (3,), f"Point must be 3D, got shape {self.value.shape}"


@dataclass
class Distance(Parameter):
    """A distance constraint (FREE by default).

    Args:
        value: Distance value
        fixed: Whether this distance is fixed (default: False = free)
        name: Optional name
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.value.shape == (), f"Distance must be scalar, got shape {self.value.shape}"


@dataclass
class Angle(Parameter):
    """An angle constraint (in radians, FREE by default).

    Args:
        value: Angle in radians
        fixed: Whether this angle is fixed (default: False = free)
        name: Optional name
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.value.shape == (), f"Angle must be scalar, got shape {self.value.shape}"


class ConstraintSystem:
    """A system of constraints for parametric modeling.

    Manages free and fixed parameters, and provides methods for
    extracting gradients with respect to free parameters only.
    """

    def __init__(self):
        self.parameters: list[Parameter] = []

    def add(self, param: Parameter) -> Parameter:
        """Add a parameter to the system."""
        self.parameters.append(param)
        return param

    def point(self, value: Array, fixed: bool = False, name: Optional[str] = None) -> Point:
        """Create and register a point constraint (free by default)."""
        p = Point(value=value, fixed=fixed, name=name)
        self.add(p)
        return p

    def distance(self, value: float, fixed: bool = False, name: Optional[str] = None) -> Distance:
        """Create and register a distance constraint (free by default)."""
        d = Distance(value=value, fixed=fixed, name=name)
        self.add(d)
        return d

    def angle(self, value: float, fixed: bool = False, name: Optional[str] = None) -> Angle:
        """Create and register an angle constraint (free by default)."""
        a = Angle(value=value, fixed=fixed, name=name)
        self.add(a)
        return a

    def get_free_params(self) -> list[Parameter]:
        """Get all free (optimizable) parameters."""
        return [p for p in self.parameters if p.free]

    def get_fixed_params(self) -> list[Parameter]:
        """Get all fixed (static) parameters."""
        return [p for p in self.parameters if not p.free]

    def to_vector(self, params: Optional[list[Parameter]] = None) -> Array:
        """Convert parameters to a flat vector.

        Args:
            params: Parameters to convert (defaults to all free parameters)

        Returns:
            Flat array of parameter values
        """
        if params is None:
            params = self.get_free_params()

        values = []
        for p in params:
            values.append(p.value.ravel())

        if not values:
            return jnp.array([])

        return jnp.concatenate(values)

    def from_vector(self, vec: Array, params: Optional[list[Parameter]] = None) -> None:
        """Update parameters from a flat vector.

        Args:
            vec: Flat array of values
            params: Parameters to update (defaults to all free parameters)
        """
        if params is None:
            params = self.get_free_params()

        offset = 0
        for p in params:
            size = p.value.size
            p.value = vec[offset:offset + size].reshape(p.value.shape)
            offset += size

    def summary(self) -> str:
        """Generate a summary of the constraint system."""
        lines = ["Constraint System:"]
        lines.append(f"  Total parameters: {len(self.parameters)}")
        lines.append(f"  Free parameters: {len(self.get_free_params())}")
        lines.append(f"  Fixed parameters: {len(self.get_fixed_params())}")

        if self.parameters:
            lines.append("\nParameters:")
            for p in self.parameters:
                status = "FREE" if p.free else "FIXED"
                name = f" ({p.name})" if p.name else ""
                lines.append(f"  [{status}]{name}: {p.value}")

        return "\n".join(lines)


def make_differentiable_sdf(sdf_builder, constraint_system: ConstraintSystem):
    """Convert an SDF builder function to a differentiable function.

    This creates a function that can be differentiated with respect to
    free parameters only, treating fixed parameters as static.

    Args:
        sdf_builder: Function that takes parameter values and returns an SDF
        constraint_system: The constraint system containing parameters

    Returns:
        Function (params_vector, query_point) -> sdf_value

    Example:
        ```python
        cs = ConstraintSystem()
        offset = cs.point([1.0, 0.0, 0.0], free=True, name="offset")
        radius = cs.distance(1.0, free=False, name="radius")

        def build_sdf(offset_val, radius_val):
            sphere = Sphere(radius=radius_val)
            return sphere.translate(offset_val)

        # Create differentiable version
        diff_sdf = make_differentiable_sdf(
            lambda: build_sdf(offset.value, radius.value),
            cs
        )

        # Optimize with respect to free parameters (offset only)
        def loss(params):
            cs.from_vector(params)
            sdf = diff_sdf()
            return sdf(target_point) ** 2

        grad = jax.grad(loss)
        ```
    """
    def differentiable_fn(params_vector: Optional[Array] = None):
        """Evaluate SDF with current parameter values.

        Args:
            params_vector: Optional flat vector of free parameter values.
                          If None, uses current values in constraint system.
        """
        if params_vector is not None:
            constraint_system.from_vector(params_vector)

        return sdf_builder()

    return differentiable_fn
