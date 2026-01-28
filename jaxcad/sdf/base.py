"""Base SDF (Signed Distance Function) class.

Architecture:
- Pure functions are the source of truth for SDF evaluation
- SDF classes are thin wrappers providing fluent API (method chaining, operators)
- Compilation unwraps classes to pure functions for JAX tracing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

from jax import Array

if TYPE_CHECKING:
    from jaxcad.parameters import Parameter


class SDF(ABC):
    """Abstract base class for Signed Distance Functions.

    An SDF represents geometry implicitly as a function f(p) -> distance,
    where:
    - f(p) < 0: point p is inside the shape
    - f(p) = 0: point p is on the surface
    - f(p) > 0: point p is outside the shape

    Attributes:
        params: Dictionary of Parameter objects for this SDF operation

    SDF classes are thin wrappers around pure functions:
    - Provide fluent API: .translate().rotate() chaining
    - Enable operators: sphere | box, sphere & box
    - Store parameters for later compilation
    - Actual computation happens in pure functions

    Each SDF instance should store its parameters in self.params dictionary:
        self.params = {
            'radius': Scalar(value=1.0, free=True, name='radius'),
            'offset': Vector(value=[0, 0, 0], free=False),
        }

    Subclasses must implement:
    - @staticmethod def sdf(...): Pure function for computation (CONVENTION)
    - __call__(p): Evaluate SDF (delegates to sdf())
    - to_functional(): Return the static sdf method

    Pattern for primitives:
        @staticmethod
        def sdf(p: Array, param1: float, param2: float) -> Array:
            # Pure computation here

        def __call__(self, p: Array) -> Array:
            return ClassName.sdf(p, self.params['param1'].value, self.params['param2'].value)

        def to_functional(self):
            return ClassName.sdf

    Pattern for transforms:
        @staticmethod
        def sdf(child_sdf: Callable, p: Array, param: float) -> Array:
            # Transform computation here

        def __call__(self, p: Array) -> Array:
            return ClassName.sdf(self.child_sdf, p, self.params['param'].value)

        def to_functional(self):
            return ClassName.sdf
    """

    params: dict[str, 'Parameter']

    def __init_subclass__(cls, **kwargs):
        """Automatically wrap __init__ to call _cast_params after initialization."""
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Auto-cast params after initialization
            if hasattr(self, 'params'):
                self._cast_params()

        cls.__init__ = new_init

    def _cast_params(self) -> None:
        """Convert all values in self.params to Parameter objects.

        This is automatically called after __init__ via __init_subclass__.
        Converts raw values (floats, ints, arrays) to Parameter objects,
        while leaving existing Parameter objects unchanged.
        """
        from jaxcad.parameters import as_parameter

        # Convert each value in the params dict
        for key, value in self.params.items():
            # already handles case where value is already a Parameter
            self.params[key] = as_parameter(value)

    def _extract_param_values(self) -> dict:
        """Extract raw numeric values from Parameter objects for graph compilation.

        Returns:
            Dictionary mapping parameter names to their raw values (arrays/floats).
            Vector parameters are converted to 3D arrays (xyz), Scalars to floats.
        """
        from jaxcad.parameters import Vector

        params = {}
        for param_name, param in self.params.items():
            if isinstance(param, Vector):
                params[param_name] = param.xyz
            else:
                params[param_name] = param.value
        return params


    @abstractmethod
    def __call__(self, p: Array) -> Array:
        """Evaluate the signed distance at point(s) p.

        This is for direct use only. During compilation, this is replaced
        by the pure function from to_functional().

        Args:
            p: Point(s) to evaluate, shape (..., 3) for 3D or (..., 2) for 2D

        Returns:
            Signed distance value(s), shape (...)
        """
        pass

    @abstractmethod
    def to_functional(self) -> Callable:
        """Return pure function for JAX tracing and compilation.

        Returns:
            Pure function with signature: (p: Array, **params) -> Array
            where params are the primitive/transform parameters.

        Example:
            sphere = Sphere(radius=1.0)
            func = sphere.to_functional()
            # func(p, radius=1.0) -> Array
        """
        pass

    def __or__(self, other: SDF) -> SDF:
        """Union operator: self | other"""
        from jaxcad.sdf.boolean import Union
        return Union(self, other)

    def __and__(self, other: SDF) -> SDF:
        """Intersection operator: self & other"""
        from jaxcad.sdf.boolean import Intersection
        return Intersection(self, other)

    def __sub__(self, other: SDF) -> SDF:
        """Difference operator: self - other"""
        from jaxcad.sdf.boolean import Difference
        return Difference(self, other)

    @classmethod
    def register(cls, name: str, sdf_class):
        """Register an SDF class as a fluent API method.

        This enables method chaining for both transforms and operations.

        Args:
            name: Method name to add to SDF class
            sdf_class: SDF class to instantiate

        Example:
            SDF.register('translate', Translate)
            SDF.register('smooth_union', SmoothUnion)
            # Now you can do:
            # sphere.translate([1, 0, 0])
            # sphere.smooth_union(box, k=0.5)
        """
        def method(self, *args, **kwargs):
            return sdf_class(self, *args, **kwargs)

        method.__name__ = name
        method.__doc__ = sdf_class.__doc__
        setattr(cls, name, method)
