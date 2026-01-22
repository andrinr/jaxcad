"""Detailed walkthrough: How fluent API translates to functional layer.

This example shows the exact transformation pipeline from fluent API
to differentiable functional form.
"""

import jax
import jax.numpy as jnp

from jaxcad.compiler.parametric import compile_parametric
from jaxcad.compiler.graph import extract_graph
from jaxcad.constraints import ConstraintSystem
from jaxcad.primitives import Sphere


def show_step_by_step():
    """Show each transformation step in detail."""

    print("=" * 80)
    print("FLUENT API → FUNCTIONAL LAYER TRANSFORMATION")
    print("=" * 80)

    # STEP 1: User writes fluent API
    print("\n" + "─" * 80)
    print("STEP 1: User writes fluent API code")
    print("─" * 80)

    fluent_code = """
    sphere = Sphere(radius=1.0)
    sdf = sphere.translate([2.0, 0.0, 0.0])
    """
    print(fluent_code)

    sphere = Sphere(radius=1.0)
    sdf = sphere.translate(jnp.array([2.0, 0.0, 0.0]))

    print("✓ Creates SDF object tree with nested class instances")
    print(f"  Type: {type(sdf).__name__}")
    print(f"  Wraps: {type(sdf.sdf).__name__}")

    # STEP 2: Extract computation graph
    print("\n" + "─" * 80)
    print("STEP 2: Compiler extracts computation graph")
    print("─" * 80)

    graph = extract_graph(sdf)

    print(f"Extracted graph with {len(graph.nodes)} nodes:")
    print(graph.visualize())

    print("\nGraph structure:")
    for i, node in enumerate(graph.nodes):
        print(f"  Node {i}: {node.op_type.value}")
        if node.params:
            for key, val in node.params.items():
                print(f"    - {key}: {val}")

    # STEP 3: Extract constraints from graph
    print("\n" + "─" * 80)
    print("STEP 3: Extract parameters as constraints")
    print("─" * 80)

    cs = ConstraintSystem()
    eval_fn = compile_parametric(sdf, cs)

    print("Extracted constraints:")
    print(cs.summary())

    print("\nConstraint details:")
    for param in cs.parameters:
        print(f"  {param.name}:")
        print(f"    Value: {param.value}")
        print(f"    Free: {param.free}")
        print(f"    Type: {type(param).__name__}")

    # STEP 4: Show the functional evaluation
    print("\n" + "─" * 80)
    print("STEP 4: Functional evaluation with traced parameters")
    print("─" * 80)

    print("\nThe compiled eval_fn signature:")
    print("  eval_fn(params_vector: Array, query_point: Array) -> Array")

    print("\nHow it works internally:")
    print("  1. params_vector contains all FREE parameter values")
    print("  2. eval_fn unpacks params into constraint system")
    print("  3. Rebuilds SDF using functional transforms")
    print("  4. Evaluates at query_point")

    # Show functional rebuild
    print("\n  Functional rebuild process:")
    print("  - Sphere(radius) → uses constraint value (traceable!)")
    print("  - translate(offset) → translate_eval(sdf, p, offset)")
    print("    where offset comes from constraint (traceable!)")

    # STEP 5: Demonstrate traceability
    print("\n" + "─" * 80)
    print("STEP 5: Verify JAX traceability (gradients work!)")
    print("─" * 80)

    def loss_fn(params_vec):
        """Loss function for optimization."""
        query_point = jnp.array([3.0, 0.0, 0.0])
        return eval_fn(params_vec, query_point) ** 2

    # Get gradient function
    grad_fn = jax.grad(loss_fn)

    # Current parameters
    params_vec = cs.to_vector()
    print(f"\nCurrent parameters: {params_vec}")
    print(f"  (Shape: {params_vec.shape})")

    # Compute gradient
    gradient = grad_fn(params_vec)
    print(f"\nGradient w.r.t. parameters: {gradient}")
    print(f"  (Shape: {gradient.shape})")

    print("\n✓ Gradients computed successfully!")
    print("  This proves the entire chain is differentiable:")
    print("  params → rebuild → evaluate → loss")

    # STEP 6: Show the transformation visually
    print("\n" + "─" * 80)
    print("STEP 6: Visual summary of transformation")
    print("─" * 80)

    print("""
    FLUENT API (Object-Oriented)
         Sphere(radius=1.0)
              ↓ instance attribute
         .translate([2, 0, 0])
              ↓ wraps in Translate class
         Translate(Sphere(...), offset=[2,0,0])

    ──────────────────────────────────────

    GRAPH EXTRACTION (Compiler)
         Node 0: PRIMITIVE
           └─ sdf_fn: Sphere instance
           └─ extract: radius=1.0

         Node 1: TRANSLATE
           └─ children: [Node 0]
           └─ extract: offset=[2,0,0]

    ──────────────────────────────────────

    CONSTRAINT SYSTEM
         sphere_radius_0: 1.0 (FREE)
         translate_1: [2,0,0] (FREE)

         params_vector = [1.0, 2.0, 0.0, 0.0]

    ──────────────────────────────────────

    FUNCTIONAL LAYER (Differentiable)
         def eval_fn(params_vec, query_point):
             # Unpack params
             radius = params_vec[0]
             offset = params_vec[1:4]

             # Rebuild with functional ops
             sphere_fn = Sphere(radius)  # ← radius is traced!
             result = translate_eval(
                 sphere_fn,
                 query_point,
                 offset  # ← offset is traced!
             )
             return result

    ──────────────────────────────────────

    JAX AUTODIFF
         grad_fn = jax.grad(loss_fn)
         ∂loss/∂params = grad_fn(params_vec)

         ✓ Gradients flow through entire chain!
    """)


def show_translate_eval_internals():
    """Show what happens inside translate_eval."""

    print("\n" + "=" * 80)
    print("DEEP DIVE: translate_eval() INTERNALS")
    print("=" * 80)

    from jaxcad.transforms.functional import translate_eval

    print("\nSource code of translate_eval:")
    print("""
    def translate_eval(sdf_fn, p: Array, offset: Array) -> Array:
        '''Evaluate translated SDF - fully differentiable.'''
        return sdf_fn(p - offset)
    """)

    print("\nKey insight:")
    print("  - Takes offset as an ARGUMENT (not instance attribute)")
    print("  - offset can be a JAX tracer (gradients flow through it)")
    print("  - Pure function: no side effects, fully traceable")

    print("\nCompare to class-based Translate:")
    print("""
    class Translate(SDF):
        def __init__(self, sdf, offset):
            self.offset = offset  # ← Stored as attribute

        def __call__(self, p):
            return self.sdf(p - self.offset)  # ← Hard to trace
    """)

    print("\nProblem with class-based:")
    print("  - offset is an instance attribute (captured at __init__)")
    print("  - JAX can't easily trace through instance attributes")
    print("  - Gradients w.r.t. offset would require special handling")

    print("\nSolution with functional:")
    print("  - offset is a function parameter")
    print("  - JAX automatically traces function arguments")
    print("  - Gradients work out of the box!")


def show_parameter_unpacking():
    """Show how params_vector gets unpacked to constraints."""

    print("\n" + "=" * 80)
    print("PARAMETER UNPACKING DETAILS")
    print("=" * 80)

    cs = ConstraintSystem()

    # Add some parameters
    cs.distance(1.0, name="radius")
    cs.point([2.0, 0.0, 0.0], name="offset")
    cs.angle(0.5, name="angle")

    print("\nConstraints in system:")
    print(cs.summary())

    # Convert to vector
    vec = cs.to_vector()
    print(f"\nFlattened to vector: {vec}")
    print(f"  Shape: {vec.shape}")

    # Modify vector
    new_vec = jnp.array([1.5, 3.0, 1.0, 0.5, 0.8])
    print(f"\nNew vector: {new_vec}")

    # Unpack back
    cs.from_vector(new_vec)

    print("\nUnpacked back to constraints:")
    for param in cs.parameters:
        print(f"  {param.name}: {param.value}")

    print("\nThe magic:")
    print("  - to_vector(): Flattens all FREE params to 1D array")
    print("  - from_vector(): Reconstructs param values from array")
    print("  - This enables: params_vec → optimize → params_vec → constraints")


def main():
    """Run all demonstrations."""
    show_step_by_step()
    show_translate_eval_internals()
    show_parameter_unpacking()

    print("\n" + "=" * 80)
    print("SUMMARY: Fluent → Functional Transformation")
    print("=" * 80)
    print("""
The transformation pipeline:

1. FLUENT API: User writes OOP-style code
   → Creates nested class instances

2. GRAPH EXTRACTION: Compiler analyzes object tree
   → Builds explicit computation graph

3. CONSTRAINT EXTRACTION: Parameters become constraints
   → Both primitives AND transforms
   → All FREE by default

4. VECTOR PACKING: Constraints → flat array
   → Enables optimization via gradient descent

5. FUNCTIONAL REBUILD: Graph → pure functions
   → translate_eval, rotate_z_eval, etc.
   → Parameters passed as arguments (traceable!)

6. JAX AUTODIFF: Compute gradients
   → ∂loss/∂params flows through entire chain
   → Optimize parameters to minimize loss

The key insight: By converting to functional form with parameters
as arguments (not attributes), we get full JAX traceability!
    """)


if __name__ == "__main__":
    main()
