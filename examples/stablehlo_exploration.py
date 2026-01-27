"""Explore StableHLO IR from JAX SDF code.

This example shows what StableHLO IR looks like for simple SDF operations,
which helps us understand what we need to compile to GLSL.
"""

import jax
import jax.numpy as jnp
from jax import make_jaxpr

# ============================================================================
# Example 1: Simple Sphere SDF
# ============================================================================

def sphere_sdf(p, radius):
    """Simple sphere SDF: length(p) - radius"""
    return jnp.linalg.norm(p) - radius

# Get JAX program representation
jaxpr = make_jaxpr(sphere_sdf)(jnp.array([1.0, 0.0, 0.0]), 1.0)
print("=" * 70)
print("SPHERE SDF - JAXPR")
print("=" * 70)
print(jaxpr)
print()

# To get StableHLO, we'd use:
# compiled = jax.jit(sphere_sdf).lower(jnp.array([1.0, 0.0, 0.0]), 1.0)
# stablehlo = compiled.compiler_ir(dialect='stablehlo')
# print(stablehlo)

# ============================================================================
# Example 2: Translated Sphere
# ============================================================================

def translated_sphere_sdf(p, offset, radius):
    """Translated sphere: length(p - offset) - radius"""
    return jnp.linalg.norm(p - offset) - radius

jaxpr = make_jaxpr(translated_sphere_sdf)(
    jnp.array([1.0, 0.0, 0.0]),
    jnp.array([0.5, 0.0, 0.0]),
    1.0
)
print("=" * 70)
print("TRANSLATED SPHERE - JAXPR")
print("=" * 70)
print(jaxpr)
print()

# ============================================================================
# Example 3: Union of Two Spheres
# ============================================================================

def union_sdf(p):
    """Union of two spheres"""
    # Sphere 1 at (-1, 0, 0) with radius 0.8
    d1 = jnp.linalg.norm(p - jnp.array([-1.0, 0.0, 0.0])) - 0.8

    # Sphere 2 at (1, 0, 0) with radius 0.8
    d2 = jnp.linalg.norm(p - jnp.array([1.0, 0.0, 0.0])) - 0.8

    # Union: min
    return jnp.minimum(d1, d2)

jaxpr = make_jaxpr(union_sdf)(jnp.array([0.0, 0.0, 0.0]))
print("=" * 70)
print("UNION OF SPHERES - JAXPR")
print("=" * 70)
print(jaxpr)
print()

# ============================================================================
# Example 4: Get Actual StableHLO
# ============================================================================

print("=" * 70)
print("SPHERE SDF - STABLEHLO")
print("=" * 70)
try:
    # JIT compile to get StableHLO
    compiled = jax.jit(sphere_sdf).lower(
        jnp.array([1.0, 0.0, 0.0]),
        1.0
    )
    stablehlo = compiled.compiler_ir(dialect='stablehlo')
    print(stablehlo)
except Exception as e:
    print(f"Note: Full StableHLO requires specific JAX version")
    print(f"Error: {e}")

print()

# ============================================================================
# Analysis
# ============================================================================

print("=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print("""
From the JAXPR, we can see the operations we need to support:

1. VECTOR OPERATIONS:
   - sub (vector subtraction: p - offset)
   - dot_general (for computing length)
   - sqrt (for length calculation)
   - reduce (for sum in dot product)

2. ARITHMETIC:
   - sub (scalar subtraction: length - radius)
   - add, mul, div (general arithmetic)

3. COMPARISONS:
   - min, max (for union/intersection)
   - select (for smooth blending)

4. MATH FUNCTIONS:
   - sin, cos, atan2 (for rotations)
   - abs (for various operations)

5. CONTROL FLOW:
   - select (ternary: cond ? a : b)
   - reduce (loops/aggregation)

The good news: Most of these map directly to GLSL built-ins!

Vector subtraction:    p - offset     →  p - offset
Length:                sqrt(dot(p,p)) →  length(p)
Min/Max:               minimum(a, b)  →  min(a, b)
Select:                select(c,a,b)  →  c ? a : b
Trig:                  sin(x)         →  sin(x)

Next steps:
1. Parse StableHLO text format (or use protobuf)
2. Build AST from operations
3. Map each op to GLSL equivalent
4. Generate shader code
""")
