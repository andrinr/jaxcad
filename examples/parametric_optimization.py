"""Example: Parametric CAD with constraint-based optimization.

Demonstrates how to:
1. Build CAD models with free/fixed parameters
2. Automatically extract constraints from transform operations
3. Optimize free parameters to satisfy design goals
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.compiler.parametric import compile_parametric, optimize_parameters
from jaxcad.constraints import ConstraintSystem
from jaxcad.primitives import Box, Sphere


def example_basic_constraints():
    """Basic constraint system usage."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Constraint System")
    print("=" * 70)

    cs = ConstraintSystem()

    # Create parameters with different properties
    center = cs.point([0.0, 0.0, 0.0], name="center")  # Free by default
    radius = cs.distance(1.0, fixed=True, name="radius")  # Explicitly fixed
    angle = cs.angle(jnp.pi/4, name="rotation_angle")  # Free by default

    print("\n" + cs.summary())

    # Convert free parameters to vector
    param_vec = cs.to_vector()
    print(f"\nFree parameters as vector: {param_vec}")

    # Modify and restore
    new_params = jnp.array([1.0, 2.0, 3.0, jnp.pi/2])
    cs.from_vector(new_params)

    print(f"\nAfter updating from vector:")
    print(f"  Center: {center.value}")
    print(f"  Radius: {radius.value} (unchanged - fixed)")
    print(f"  Angle: {angle.value}")


def example_automatic_extraction():
    """Automatically extract constraints from SDF expressions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Automatic Constraint Extraction")
    print("=" * 70)

    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Build a complex expression - constraints will be auto-extracted
    sdf = (
        sphere
        .translate(jnp.array([1.0, 0.0, 0.0]))
        .rotate('z', jnp.pi/6)
        .twist('z', 0.5)
    )

    print("\nBuilt SDF with transforms:")
    print("  - translate([1, 0, 0])")
    print("  - rotate(z, π/6)")
    print("  - twist(z, 0.5)")

    # Compile - this extracts all transform parameters
    eval_fn = compile_parametric(sdf, cs)

    print("\nAutomatically extracted constraints:")
    print(cs.summary())

    print(f"\nTotal free parameters: {len(cs.to_vector())}")


def example_optimize_position():
    """Optimize shape position to pass through target points."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Optimize Position to Match Target Points")
    print("=" * 70)

    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Start with sphere at origin (unknown position)
    sdf = sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Goal: Find where the sphere should be centered so that
    # its surface passes through these target points
    target_points = jnp.array([
        [2.0, 0.0, 0.0],   # Point on +X axis
        [1.0, 1.0, 0.0],   # Point at 45°
        [1.0, 0.0, 1.0],   # Point elevated in Z
    ])

    # We want distance = 0 at all target points (on surface)
    target_values = jnp.zeros(len(target_points))

    print("\nTarget: Sphere surface should pass through:")
    for i, p in enumerate(target_points):
        print(f"  Point {i+1}: {p}")

    print(f"\nInitial sphere offset: [0, 0, 0]")
    print("Optimizing...")

    # Optimize!
    optimized_params, loss_history = optimize_parameters(
        sdf,
        target_points,
        target_values,
        cs,
        num_iterations=50,
        learning_rate=0.1
    )

    # Get the optimized offset
    free_params = cs.get_free_params()
    optimized_offset = free_params[0].value if free_params else None

    print(f"\n✓ Optimization complete!")
    print(f"  Optimized offset: {optimized_offset}")
    print(f"  Final loss: {loss_history[-1]:.6f}")
    print(f"  Loss improvement: {loss_history[0]:.6f} → {loss_history[-1]:.6f}")

    # Plot loss history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Optimization Convergence')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE, log scale)')
    plt.title('Optimization Convergence (Log Scale)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('assets/parametric_optimization.png', dpi=150, bbox_inches='tight')
    print("\nSaved: assets/parametric_optimization.png")


def example_mixed_free_fixed():
    """Demonstrate mixing free and fixed parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Mixed Free and Fixed Parameters")
    print("=" * 70)

    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([0.8, 0.8, 0.8]))

    # Build complex geometry with multiple transforms
    sdf = (
        sphere.translate(jnp.array([1.0, 0.0, 0.0]))  # Will be free
        | box.rotate('z', jnp.pi/4)                    # Will be free
    ).twist('z', 1.5)                                  # Will be free

    # Compile to extract all parameters
    eval_fn = compile_parametric(sdf, cs)

    print("\nInitial state (all free):")
    print(cs.summary())

    # Now mark some as fixed
    params = cs.get_free_params()
    if len(params) >= 2:
        params[1].fixed = True  # Fix the rotation angle
        print(f"\nMarked '{params[1].name}' as FIXED")

    print("\nAfter marking rotation as fixed:")
    print(cs.summary())

    # Now optimization will only affect free parameters
    target_points = jnp.array([[2.5, 0.0, 0.0]])
    target_values = jnp.array([0.0])

    print("\nOptimizing with mixed free/fixed parameters...")
    optimized_params, loss_history = optimize_parameters(
        sdf,
        target_points,
        target_values,
        cs,
        num_iterations=30,
        learning_rate=0.1
    )

    print(f"\n✓ Optimization complete (only free params changed)!")
    print(f"  Final loss: {loss_history[-1]:.6f}")

    print("\nFinal parameter values:")
    for p in cs.parameters:
        status = "FREE" if p.free else "FIXED"
        print(f"  [{status}] {p.name}: {p.value}")


def main():
    """Run all examples."""
    example_basic_constraints()
    example_automatic_extraction()
    example_optimize_position()
    example_mixed_free_fixed()

    print("\n" + "=" * 70)
    print("PARAMETRIC OPTIMIZATION EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. ALL parameters are FREE (optimizable) by default")
    print("  2. Both primitives AND transforms become constraints automatically")
    print("  3. Mark parameters as fixed=True to exclude from optimization")
    print("  4. JAX AD only differentiates through free parameters")
    print("  5. Optimize to match target geometric conditions")


if __name__ == "__main__":
    main()
