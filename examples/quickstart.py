"""Quickstart: Get started with JaxCAD in 5 minutes.

This example shows the basics:
1. Creating primitives
2. Boolean operations (CSG)
3. Transforms
4. Parametric optimization with gradients
"""

import jax
import jax.numpy as jnp

from jaxcad.constraints import Distance, Vector
from jaxcad.parametric import parametric
from jaxcad.primitives import Sphere, Box


def example_basic_shapes():
    """Create and evaluate basic shapes."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Shapes")
    print("=" * 70)

    # Create a sphere
    sphere = Sphere(radius=1.0)

    # Evaluate at a point
    point = jnp.array([2.0, 0.0, 0.0])
    distance = sphere(point)
    print(f"\nSphere with radius 1.0")
    print(f"Distance at [2,0,0]: {distance:.2f} (outside)")

    # Point inside sphere
    inside_point = jnp.array([0.5, 0.0, 0.0])
    distance_inside = sphere(inside_point)
    print(f"Distance at [0.5,0,0]: {distance_inside:.2f} (inside)")


def example_boolean_operations():
    """Combine shapes with CSG operations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Boolean Operations (CSG)")
    print("=" * 70)

    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([0.8, 0.8, 0.8]))

    # Union (|)
    union = sphere | box
    print("\nUnion: sphere | box")

    # Intersection (&)
    intersection = sphere & box
    print("Intersection: sphere & box")

    # Difference (-)
    difference = sphere - box
    print("Difference: sphere - box")

    # Evaluate
    point = jnp.array([0.5, 0.5, 0.5])
    print(f"\nAt point [0.5, 0.5, 0.5]:")
    print(f"  Union distance: {union(point):.3f}")
    print(f"  Intersection distance: {intersection(point):.3f}")
    print(f"  Difference distance: {difference(point):.3f}")


def example_transforms():
    """Apply transformations to shapes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Transforms")
    print("=" * 70)

    sphere = Sphere(radius=1.0)

    # Translate
    translated = sphere.translate(jnp.array([2.0, 0.0, 0.0]))
    print("\nTranslated sphere by [2, 0, 0]")

    # Rotate
    rotated = sphere.rotate('z', jnp.pi/4)
    print("Rotated sphere around Z axis by π/4")

    # Scale
    scaled = sphere.scale(2.0)
    print("Scaled sphere by factor of 2")

    # Chain transforms
    transformed = sphere.translate([1, 0, 0]).rotate('z', jnp.pi/6).scale(1.5)
    print("Chained: translate → rotate → scale")

    # Evaluate
    point = jnp.array([2.0, 0.0, 0.0])
    print(f"\nOriginal sphere at [2,0,0]: {sphere(point):.3f}")
    print(f"Translated sphere at [2,0,0]: {translated(point):.3f}")


def example_parametric():
    """Use @parametric decorator for gradient-based optimization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Parametric Optimization")
    print("=" * 70)

    # Define a parametric shape
    @parametric
    def my_shape():
        # All parameters are automatically differentiable
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Get initial parameters
    params = my_shape.init_params()
    print("\nInitial parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Goal: Optimize so sphere surface passes through target
    target_point = jnp.array([2.5, 0.0, 0.0])
    print(f"\nTarget: Make surface pass through {target_point}")

    # Define loss function
    def loss_fn(params):
        distance = my_shape(params, target_point)
        return distance ** 2  # Minimize squared distance

    # Optimize using JAX
    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.1

    initial_loss = loss_fn(params)
    print(f"Initial loss: {initial_loss:.6f}")

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


def example_free_parameters():
    """Use explicit constraints for fine control."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Explicit Constraints (Advanced)")
    print("=" * 70)

    # Create constraints explicitly
    radius = Distance(value=1.0, free=False, name='radius')  # Fixed
    offset = Vector(value=jnp.array([0.0, 0.0, 0.0]), free=True, name='offset')  # Free

    @parametric
    def constrained_shape():
        sphere = Sphere(radius=radius)
        return sphere.translate(offset)

    params = constrained_shape.init_params()
    print("\nParameters:")
    print(f"  Radius: fixed at {radius.value}")
    print(f"  Offset: free, starts at {offset.value}")

    # Optimization will only affect free parameters (offset)
    target = jnp.array([2.0, 0.0, 0.0])

    def loss_fn(p):
        return constrained_shape(p, target) ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.1

    for i in range(30):
        grad = grad_fn(params)
        params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            params, grad
        )

    print(f"\nAfter optimization (radius stays fixed):")
    print(f"  Radius: {radius.value}")
    print(f"  Offset: {params['translate_1']['offset']}")


def main():
    """Run all examples."""
    example_basic_shapes()
    example_boolean_operations()
    example_transforms()
    example_parametric()
    example_free_parameters()

    print("\n" + "=" * 70)
    print("QUICKSTART COMPLETE")
    print("=" * 70)
    print("""
Key takeaways:

1. PRIMITIVES: Sphere, Box, Cylinder, Cone, Torus, etc.
   sphere = Sphere(radius=1.0)

2. BOOLEAN OPS: Union (|), Intersection (&), Difference (-)
   combined = sphere | box

3. TRANSFORMS: .translate(), .rotate(), .scale(), .twist(), etc.
   transformed = sphere.translate([1,0,0]).rotate('z', π/4)

4. PARAMETRIC: @parametric decorator makes everything differentiable
   @parametric
   def my_shape():
       return Sphere(radius=1.0).translate([2,0,0])

5. OPTIMIZATION: Use JAX gradients to optimize parameters
   grad_fn = jax.grad(loss_fn)
   params = params - learning_rate * grad_fn(params)

6. CONSTRAINTS: Mark parameters as free=True for optimization
   radius = Distance(value=1.0, free=True)

Next steps:
- Check examples/boolean_operations.py for visualization
- Check examples/decorator_api.py for advanced parametric usage
- Check examples/primitives.py to see all available shapes
    """)


if __name__ == "__main__":
    main()
