"""Compiler: Parameter extraction and compilation to pure JAX functions.

This module provides tools for:
1. Extracting parameters from SDF trees
2. Compiling SDFs to pure JAX functions
3. Constraint-aware parameter extraction with DOF reduction

The compiler layer bridges the parametric SDF layer with JAX optimization.
"""

from jaxcad.compiler.extraction import extract_parameters
from jaxcad.compiler.compilation import to_function
from jaxcad.compiler.constrained import extract_parameters_with_constraints

__all__ = [
    'extract_parameters',
    'to_function',
    'extract_parameters_with_constraints',
]
