"""Example: New @parametric decorator API with PyTrees.

Demonstrates the clean, decorator-based API for parametric CAD with
automatic differentiation. No manual constraint management needed!
"""

import jax
import jax.numpy as jnp

from jaxcad.parametric import parametric
from jaxcad.primitives import Sphere, Box


def example_basic_usage():
    """Basic @parametric decorator usage."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Decorator Usage")
    print("=" * 70)

    # Define a parametric SDF using decorator
    @parametric
    def my_shape():
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([2.0, 0.0, 0.0]))

    # Get initial parameters (PyTree structure)
    params = my_shape.init_params()
    print("\nInitial parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Evaluate at a point
    point = jnp.array([3.0, 0.0, 0.0])
    distance = my_shape(params, point)
    print(f"\nDistance at {point}: {distance:.4f}")

    # Modify parameters
    params['sphere_0']['radius'] = jnp.array(2.0)
    params['translate_1']['offset'] = jnp.array([3.0, 0.0, 0.0])

    distance_new = my_shape(params, point)
    print(f"\nAfter modifying params:")
    print(f"  New radius: {params['sphere_0']['radius']}")
    print(f"  New offset: {params['translate_1']['offset']}")
    print(f"  Distance at {point}: {distance_new:.4f}")


def example_gradient_descent():
    """Optimize parameters using gradient descent."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Gradient-Based Optimization")
    print("=" * 70)

    @parametric
    def my_shape():
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Goal: Find parameters so sphere surface passes through target point
    target_point = jnp.array([3.0, 0.0, 0.0])
    print(f"\nTarget: Make sphere surface pass through {target_point}")

    # Define loss function
    def loss_fn(params):
        """Distance squared at target point."""
        dist = my_shape(params, target_point)
        return dist ** 2

    # Initial parameters
    params = my_shape.init_params()
    print(f"\nInitial parameters:")
    print(f"  Radius: {params['sphere_0']['radius']}")
    print(f"  Offset: {params['translate_1']['offset']}")
    print(f"  Loss: {loss_fn(params):.6f}")

    # Optimize with JAX
    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.1

    print("\nOptimizing...")
    for i in range(50):
        grad = grad_fn(params)
        params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            params, grad
        )
        if i % 10 == 0:
            loss = loss_fn(params)
            print(f"  Iteration {i}: loss = {loss:.6f}")

    # Final result
    final_loss = loss_fn(params)
    print(f"\n✓ Optimization complete!")
    print(f"  Final radius: {params['sphere_0']['radius']}")
    print(f"  Final offset: {params['translate_1']['offset']}")
    print(f"  Final loss: {final_loss:.6f}")


def example_complex_shape():
    """Optimize a complex shape with multiple primitives."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Complex Shape Optimization")
    print("=" * 70)

    @parametric
    def complex_shape():
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([1.0, 1.0, 1.0]))

        return (
            sphere.translate(jnp.array([0.0, 0.0, 0.0]))
            | box.translate(jnp.array([2.0, 0.0, 0.0]))
        ).rotate('z', 0.0)

    params = complex_shape.init_params()
    print("\nInitial parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Goal: Fit shape to multiple target points
    target_points = jnp.array([
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
    ])

    def loss_fn(params):
        """Sum of squared distances at target points."""
        distances = jax.vmap(lambda p: complex_shape(params, p))(target_points)
        return jnp.sum(distances ** 2)

    print(f"\nInitial loss: {loss_fn(params):.6f}")

    # Optimize
    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.05

    for i in range(30):
        grad = grad_fn(params)
        params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            params, grad
        )

    final_loss = loss_fn(params)
    print(f"Final loss: {final_loss:.6f}")
    print(f"\nOptimized parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")


def example_direct_compilation():
    """Compile SDF directly without decorator."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Direct Compilation (No Decorator)")
    print("=" * 70)

    # Build SDF fluently
    sphere = Sphere(radius=1.5)
    sdf = sphere.translate(jnp.array([1.0, 2.0, 3.0])).scale(2.0)

    # Compile it
    compiled_sdf = parametric(sdf)

    # Use it
    params = compiled_sdf.init_params()
    point = jnp.array([2.0, 2.0, 3.0])
    distance = compiled_sdf(params, point)

    print(f"\nCompiled SDF parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print(f"\nDistance at {point}: {distance:.4f}")

    # Test gradient
    def loss_fn(p):
        return compiled_sdf(p, point) ** 2

    grad = jax.grad(loss_fn)(params)
    print(f"\nGradient computed successfully!")
    print(f"  Gradient keys: {list(grad.keys())}")


def example_pytree_benefits():
    """Show benefits of PyTree structure vs flat vectors."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: PyTree Benefits")
    print("=" * 70)

    @parametric
    def my_shape():
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([2.0, 0.0, 0.0])).scale(1.5)

    params = my_shape.init_params()

    print("\nPyTree structure:")
    print(f"  Type: {type(params)}")
    print(f"  Keys: {list(params.keys())}")
    print(f"\nHierarchical access:")
    print(f"  params['sphere_0']['radius'] = {params['sphere_0']['radius']}")
    print(f"  params['translate_1']['offset'] = {params['translate_1']['offset']}")
    print(f"  params['scale_2']['scale'] = {params['scale_2']['scale']}")

    print("\n✓ Benefits:")
    print("  1. Natural nested structure (not flat vector)")
    print("  2. Easy to access/modify individual parameters")
    print("  3. JAX handles gradients automatically")
    print("  4. No manual pack/unpack needed")
    print("  5. Works seamlessly with jax.tree_util operations")

    # Demonstrate tree operations
    print("\n✓ PyTree operations:")

    # Scale all parameters by 2
    scaled_params = jax.tree_util.tree_map(lambda x: x * 2, params)
    print(f"  Scaled all params by 2:")
    print(f"    Original radius: {params['sphere_0']['radius']}")
    print(f"    Scaled radius: {scaled_params['sphere_0']['radius']}")

    # Compute gradient and update
    point = jnp.array([4.0, 0.0, 0.0])
    grad = jax.grad(lambda p: my_shape(p, point) ** 2)(params)
    updated_params = jax.tree_util.tree_map(
        lambda p, g: p - 0.1 * g,
        params, grad
    )
    print(f"  After gradient update:")
    print(f"    Original offset: {params['translate_1']['offset']}")
    print(f"    Updated offset: {updated_params['translate_1']['offset']}")


def main():
    """Run all examples."""
    example_basic_usage()
    example_gradient_descent()
    example_complex_shape()
    example_direct_compilation()
    example_pytree_benefits()

    print("\n" + "=" * 70)
    print("SUMMARY: @parametric Decorator API")
    print("=" * 70)
    print("""
Key features:

1. CLEAN API:
   @parametric decorator converts SDF to differentiable function
   No manual constraint management needed

2. PYTREE PARAMETERS:
   Natural nested structure: {'sphere_0': {'radius': 1.0}, ...}
   Not flat vectors - easier to understand and modify

3. FULLY DIFFERENTIABLE:
   jax.grad works out of the box
   All parameters (primitives + transforms) are differentiable

4. FUNCTIONAL EVALUATION:
   Pure function: fn(params, point) -> distance
   No mutable state - safe for JAX transformations

5. FLEXIBLE:
   Use as decorator: @parametric def my_shape(): ...
   Or directly: compiled = parametric(sdf)

6. JAX-NATIVE:
   Works seamlessly with jax.jit, jax.vmap, jax.grad
   Tree operations via jax.tree_util

The decorator approach combines the convenience of the fluent API
with the power of functional, fully-differentiable evaluation!
    """)


if __name__ == "__main__":
    main()
