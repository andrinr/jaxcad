"""Layered Construction System Demonstration

This example demonstrates the complete layered architecture of JaxCAD:

1. **Geometry Layer** - Parametric geometric primitives (Line, Rectangle, Circle)
2. **Constraints Layer** - Geometric constraints (Distance, Angle, etc.)
3. **Construction Layer** - Bridge from geometry to SDFs (extrude, from_line, etc.)
4. **Compiler Layer** - Extract and optimize constrained parameters
5. **SDF Layer** - Boolean operations and rendering

This shows how all layers work together for constraint-driven CAD.

Author: JaxCAD Team
Date: 2026-02-02
"""

import jax
import jax.numpy as jnp

# Layer 1: Geometry
from jaxcad.geometry import Vector, Scalar
from jaxcad.geometry.primitives import Line, Rectangle, Circle

# Layer 2: Constraints
from jaxcad.constraints import (
    ConstraintGraph,
    DistanceConstraint as Distance,
    PerpendicularConstraint as Perpendicular,
)

# Layer 3: Construction
from jaxcad.construction import extrude, from_line, from_circle, from_point

# Layer 4: Compiler
from jaxcad.compiler import extract_parameters_with_constraints

# Layer 5: SDF (for boolean operations)
# (imported dynamically in construction functions)


def example_1_constrained_capsule():
    """Example 1: Constrained capsule from line."""
    print("=" * 80)
    print("EXAMPLE 1: Constrained Capsule from Line")
    print("=" * 80)
    print()

    # Layer 1: Define parametric geometry
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([3, 0, 0], free=True, name='p2')

    line = Line(start=p1, end=p2)

    print(f"Line: {line.start.xyz} → {line.end.xyz}")
    print(f"Initial length: {line.length():.3f}")
    print()

    # Layer 2: Add distance constraint
    graph = ConstraintGraph()
    constraint = Distance(p1, p2, distance=2.0)
    graph.add_constraint(constraint)

    print(f"Constraint: Fix distance to 2.0")
    print(f"DOF: 6 (two points) → {6 - graph.get_total_dof_reduction()} (with constraint)")
    print()

    # Layer 3: Construct SDF from geometry
    capsule = from_line(line, radius=0.5)

    print(f"Constructed: Capsule with radius 0.5")
    print(f"  Capsule parameters: {list(capsule.params.keys())}")
    print()

    # Layer 4: Extract constrained parameters for optimization
    reduced, null_space, base, params = extract_parameters_with_constraints(
        capsule,
        graph
    )

    print(f"Compiler output:")
    print(f"  Reduced DOF: {reduced.shape[0]}")
    print(f"  Null space: {null_space.shape}")
    print(f"  Base point: {base.shape}")
    print()

    print("✓ Capsule is now ready for constrained optimization")
    print()


def example_2_extruded_rectangle():
    """Example 2: Extruded box from rectangle."""
    print("=" * 80)
    print("EXAMPLE 2: Extruded Box from Rectangle")
    print("=" * 80)
    print()

    # Layer 1: Define parametric rectangle
    center = Vector([0, 0, 0], free=True, name='center')
    width = Scalar(2.0, free=True, name='width')
    height = Scalar(1.0, free=False, name='height')  # Fixed

    rect = Rectangle(center=center, width=width, height=height)

    print(f"Rectangle: center={rect.center.xyz}, width={rect.width.value}, height={rect.height.value}")
    print()

    # Layer 2: No constraints in this example
    graph = ConstraintGraph()

    # Layer 3: Extrude to create box
    depth = Scalar(3.0, free=False, name='depth')
    box = extrude(rect, depth=depth)

    print(f"Extruded box with depth={depth.value}")
    print(f"  Box size: {box.params['size'].xyz}")
    print()

    # Layer 4: Extract parameters
    reduced, null_space, base, params = extract_parameters_with_constraints(
        box,
        graph
    )

    print(f"Free parameters: {[p.name for p in params if p.free]}")
    print(f"DOF: {reduced.shape[0]}")
    print()

    print("✓ Box can be optimized by varying center and width")
    print()


def example_3_cylinder_from_circle():
    """Example 3: Cylinder from constrained circle."""
    print("=" * 80)
    print("EXAMPLE 3: Cylinder from Constrained Circle")
    print("=" * 80)
    print()

    # Layer 1: Define parametric circle
    center = Vector([0, 0, 0], free=True, name='center')
    radius = Scalar(1.5, free=True, name='radius')

    circle = Circle(center=center, radius=radius)

    print(f"Circle: center={circle.center.xyz}, radius={circle.radius.value}")
    print()

    # Layer 2: No constraints
    graph = ConstraintGraph()

    # Layer 3: Create cylinder from circle
    height = Scalar(5.0, free=False, name='height')
    cylinder = from_circle(circle, height=height)

    print(f"Cylinder: radius={cylinder.params['radius'].value}, height={cylinder.params['height'].value}")
    print()

    # Layer 4: Extract parameters
    reduced, null_space, base, params = extract_parameters_with_constraints(
        cylinder,
        graph
    )

    print(f"DOF: {reduced.shape[0]} (center.xyz + radius)")
    print()

    print("✓ Cylinder shares parameters with circle")
    print()


def example_4_multi_geometry_assembly():
    """Example 4: Multiple geometries with constraints."""
    print("=" * 80)
    print("EXAMPLE 4: Multi-Geometry Assembly with Constraints")
    print("=" * 80)
    print()

    # Layer 1: Define multiple geometric entities
    p1 = Vector([0, 0, 0], free=True, name='p1')
    p2 = Vector([2, 0, 0], free=True, name='p2')
    p3 = Vector([1, 2, 0], free=True, name='p3')

    line1 = Line(start=p1, end=p2)
    line2 = Line(start=p2, end=p3)

    print(f"Line 1: {line1.start.xyz} → {line1.end.xyz}")
    print(f"Line 2: {line2.start.xyz} → {line2.end.xyz}")
    print()

    # Layer 2: Add constraints
    graph = ConstraintGraph()
    graph.add_constraint(Distance(p1, p2, 2.0))  # Fix line1 length
    graph.add_constraint(Distance(p2, p3, 2.0))  # Fix line2 length
    graph.add_constraint(Perpendicular(
        Vector(line1.direction(), free=False, name='dir1'),
        Vector(line2.direction(), free=False, name='dir2')
    ))

    print(f"Constraints:")
    for c in graph.constraints:
        print(f"  {c}")
    print()
    print(f"Total DOF reduction: {graph.get_total_dof_reduction()}")
    print(f"DOF: 9 (3 points × 3) → {9 - graph.get_total_dof_reduction()}")
    print()

    # Layer 3: Construct SDFs
    capsule1 = from_line(line1, radius=0.3)
    capsule2 = from_line(line2, radius=0.3)

    # Layer 5: Boolean union
    assembly = capsule1 | capsule2

    print(f"Assembly: capsule1 | capsule2")
    print()

    # Layer 4: Extract constrained parameters
    reduced, null_space, base, params = extract_parameters_with_constraints(
        assembly,
        graph
    )

    print(f"Reduced DOF: {reduced.shape[0]}")
    print(f"The assembly forms an 'L' shape that can rotate/translate while")
    print(f"maintaining perpendicular arms of fixed length.")
    print()

    print("✓ Complex assemblies can be constrained and optimized")
    print()


def example_5_optimization_workflow():
    """Example 5: Complete optimization workflow."""
    print("=" * 80)
    print("EXAMPLE 5: Complete Optimization Workflow")
    print("=" * 80)
    print()

    # Layer 1: Geometry
    p1 = Vector([0, 0, 1], free=True, name='p1')
    p2 = Vector([2, 0, 1], free=True, name='p2')

    line = Line(start=p1, end=p2)

    # Layer 2: Constraints
    graph = ConstraintGraph()
    graph.add_constraint(Distance(p1, p2, 2.0))

    # Layer 3: Construction
    capsule = from_line(line, radius=0.5)

    # Layer 4: Compile
    reduced, null_space, base, params = extract_parameters_with_constraints(
        capsule,
        graph
    )

    print(f"Initial configuration:")
    print(f"  p1: {p1.xyz}")
    print(f"  p2: {p2.xyz}")
    print(f"  Distance: {jnp.linalg.norm(p2.xyz - p1.xyz):.3f}")
    print()

    # Define optimization goal: minimize z-coordinates
    def loss_fn(reduced_params):
        # Project to full space
        full = base + null_space @ reduced_params

        # Sum of z-coordinates (indices 2 and 5)
        return full[2] + full[5]

    # Optimize
    print("Optimizing to minimize z-coordinates...")
    grad_fn = jax.grad(loss_fn)
    current_reduced = reduced

    for i in range(50):
        grad = grad_fn(current_reduced)
        current_reduced = current_reduced - 0.1 * grad

    # Results
    final_full = base + null_space @ current_reduced
    final_p1 = final_full[:3]
    final_p2 = final_full[3:6]
    final_distance = jnp.linalg.norm(final_p2 - final_p1)

    print()
    print(f"Final configuration:")
    print(f"  p1: {final_p1}")
    print(f"  p2: {final_p2}")
    print(f"  Distance: {final_distance:.3f} (target: 2.000)")
    print(f"  Constraint error: {jnp.abs(final_distance - 2.0):.6f}")
    print()

    print("✓ Optimization completed while preserving constraints")
    print()


def main():
    """Run all examples."""
    example_1_constrained_capsule()
    example_2_extruded_rectangle()
    example_3_cylinder_from_circle()
    example_4_multi_geometry_assembly()
    example_5_optimization_workflow()

    print("=" * 80)
    print("SUMMARY: Layered Architecture")
    print("=" * 80)
    print("""
JaxCAD's layered architecture enables flexible, constraint-driven CAD:

1. **Geometry Layer** (jaxcad/geometry/)
   - Parametric primitives: Line, Rectangle, Circle
   - Parameters: Vector, Scalar (free or fixed)
   - Pure geometric definitions, no SDF dependencies

2. **Constraints Layer** (jaxcad/constraints/)
   - Constraint types: Distance, Angle, Parallel, Perpendicular
   - ConstraintGraph manages DOF reduction
   - Null space projection for optimization

3. **Construction Layer** (jaxcad/construction/)
   - Bridges geometry → SDF
   - Functions: extrude(), from_line(), from_circle(), from_point()
   - Preserves parameter references

4. **Compiler Layer** (jaxcad/compiler/)
   - extract_parameters() - walks SDF tree
   - extract_parameters_with_constraints() - integrates constraints
   - compile_to_function() - pure JAX functions

5. **SDF Layer** (jaxcad/sdf/)
   - Primitives: Box, Sphere, Cylinder, Capsule, etc.
   - Boolean operations: Union, Intersection, Difference
   - Transforms: Translate, Rotate, Scale

**Workflow:**
```python
# 1. Define geometry
p1 = Vector([0, 0, 0], free=True, name='p1')
p2 = Vector([2, 0, 0], free=True, name='p2')
line = Line(start=p1, end=p2)

# 2. Add constraints
graph = ConstraintGraph()
graph.add_constraint(Distance(p1, p2, 2.0))

# 3. Construct SDF
capsule = from_line(line, radius=0.5)

# 4. Extract constrained parameters
reduced, null_space, base, params = extract_parameters_with_constraints(
    capsule, graph
)

# 5. Optimize
def loss_fn(reduced_params):
    full = base + null_space @ reduced_params
    # ... compute loss ...

grad_fn = jax.grad(loss_fn)
# ... gradient descent ...
```

Each layer is independent and can be used standalone!
    """)


if __name__ == "__main__":
    main()
