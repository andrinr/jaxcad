"""Compiler: Parameter extraction and compilation to pure JAX functions.

This module provides tools for:
1. Extracting parameters from SDF trees
2. Compiling SDFs to pure JAX functions
3. Constraint-aware parameter extraction with DOF reduction

The compiler layer bridges the parametric SDF layer with JAX optimization.
"""

from jaxcad.compiler.extraction import extract_parameters
from jaxcad.compiler.compilation import functionalize
from jaxcad.compiler.constrained import extract_parameters_with_constraints
from jaxcad.compiler.solve import solve_constraints

__all__ = [
    'extract_parameters',
    'functionalize',
    'extract_parameters_with_constraints',
    'solve_constraints',
]
