"""Backward compatibility shim for parameters module.

This module re-exports from jaxcad.geometry.parameters for backward compatibility.
New code should import directly from jaxcad.geometry.parameters.
"""

# Re-export everything from the new location
from jaxcad.geometry.parameters import (
    Parameter,
    Scalar,
    Vector,
    as_parameter,
)

__all__ = [
    'Parameter',
    'Scalar',
    'Vector',
    'as_parameter',
]
