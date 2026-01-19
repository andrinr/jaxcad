"""Example demonstrating differentiability of JaxCAD operations.

This example shows how to:
1. Create parametric shapes
2. Compute gradients of vertex positions with respect to parameters
3. Use JAX's automatic differentiation
"""

import jax
import jax.numpy as jnp

from jaxcad import box, rotate, sphere, translate
from jaxcad.viz import plot_solids


def parametric_box_volume(size):
    """Create a box and compute its approximate volume from vertices.

    This function is fully differentiable with respect to size.
    """
    center = jnp.array([0.0, 0.0, 0.0])
    solid = box(center, size)

    # Compute bounding box volume as proxy
    min_coords = jnp.min(solid.vertices, axis=0)
    max_coords = jnp.max(solid.vertices, axis=0)
    dims = max_coords - min_coords
    volume = jnp.prod(dims)

    return volume


def parametric_shape_position(center_x, radius):
    """Create a sphere and return the position of its first vertex.

    This function is fully differentiable with respect to center_x and radius.
    """
    center = jnp.array([center_x, 0.0, 0.0])
    solid = sphere(center, radius, resolution=16)

    # Return the position of the first vertex
    return solid.vertices[0]


def transformed_shape_position(translation_x, rotation_angle):
    """Create and transform a box, returning vertex position.

    This function is fully differentiable with respect to transformation parameters.
    """
    center = jnp.array([0.0, 0.0, 0.0])
    size = jnp.array([1.0, 1.0, 1.0])

    # Create box
    solid = box(center, size)

    # Apply transformations
    solid = translate(solid, jnp.array([translation_x, 0.0, 0.0]))
    solid = rotate(solid, jnp.array([0.0, 0.0, 1.0]), rotation_angle)

    # Return position of a specific vertex
    return solid.vertices[0]


def main():
    print("=" * 60)
    print("JaxCAD: Differentiable CAD Example")
    print("=" * 60)

    # Example 1: Gradient of volume with respect to size
    print("\n1. Gradient of box volume w.r.t. size parameters")
    print("-" * 60)

    size = jnp.array([2.0, 3.0, 4.0])
    volume_grad_fn = jax.grad(parametric_box_volume)
    gradient = volume_grad_fn(size)

    print(f"Box size: {size}")
    print(f"Volume: {parametric_box_volume(size):.2f}")
    print(f"∂(volume)/∂(size): {gradient}")
    print("Expected: [12, 8, 6] (product of other two dimensions)")

    # Example 2: Gradient of vertex position with respect to shape parameters
    print("\n2. Gradient of vertex position w.r.t. shape parameters")
    print("-" * 60)

    center_x = 5.0
    radius = 2.0

    # Compute gradients with respect to both parameters
    grad_fn = jax.grad(lambda cx, r: jnp.sum(parametric_shape_position(cx, r)), argnums=(0, 1))
    grad_center, grad_radius = grad_fn(center_x, radius)

    vertex_pos = parametric_shape_position(center_x, radius)
    print(f"Sphere center_x: {center_x}, radius: {radius}")
    print(f"First vertex position: {vertex_pos}")
    print(f"∂(vertex)/∂(center_x): {grad_center:.4f}")
    print(f"∂(vertex)/∂(radius): {grad_radius:.4f}")

    # Example 3: Gradient through transformations
    print("\n3. Gradient of vertex position w.r.t. transformation parameters")
    print("-" * 60)

    translation_x = 1.0
    rotation_angle = jnp.pi / 4  # 45 degrees

    # Compute Jacobian (gradients of all output components w.r.t. inputs)
    jacobian_fn = jax.jacfwd(transformed_shape_position, argnums=(0, 1))
    jac_translation, jac_rotation = jacobian_fn(translation_x, rotation_angle)

    vertex_pos = transformed_shape_position(translation_x, rotation_angle)
    print(f"Translation: {translation_x}, Rotation: {rotation_angle:.4f} rad")
    print(f"Transformed vertex position: {vertex_pos}")
    print(f"∂(vertex)/∂(translation_x):\n{jac_translation}")
    print(f"∂(vertex)/∂(rotation_angle):\n{jac_rotation}")

    # Example 4: Vectorized gradient computation with vmap
    print("\n4. Vectorized gradient computation (multiple parameter values)")
    print("-" * 60)

    # Compute gradients for multiple box sizes at once
    sizes = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )

    # Vectorize the gradient function
    batched_grad_fn = jax.vmap(volume_grad_fn)
    batched_gradients = batched_grad_fn(sizes)

    print("Box sizes and their volume gradients:")
    for _i, (size, grad) in enumerate(zip(sizes, batched_gradients)):
        vol = parametric_box_volume(size)
        print(f"  Size {size} -> Volume: {vol:.2f}, Gradient: {grad}")

    # Example 5: Second-order derivatives (Hessian)
    print("\n5. Second-order derivatives (curvature of volume w.r.t. size)")
    print("-" * 60)

    size = jnp.array([2.0, 3.0, 4.0])
    hessian_fn = jax.hessian(parametric_box_volume)
    hessian = hessian_fn(size)

    print(f"Box size: {size}")
    print(f"Hessian matrix (∂²V/∂size²):\n{hessian}")

    print("\n" + "=" * 60)
    print("All gradients computed successfully!")
    print("=" * 60)

    # Visualize examples
    print("\nVisualizing example shapes...")

    # Example 1: Different sized boxes showing gradient effects
    sizes = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 3.0, 4.0]])
    boxes = [box(jnp.array([i * 3.0, 0.0, 0.0]), size) for i, size in enumerate(sizes)]
    plot_solids(
        boxes,
        colors=['lightblue', 'lightcoral', 'lightgreen'],
        labels=[f'Size {size}' for size in sizes],
        title='Example 1: Box Volumes (Different Sizes)'
    )

    # Example 2: Sphere at different positions
    center_x_values = [0.0, 3.0, 5.0]
    radius = 0.8
    spheres = [sphere(jnp.array([cx, 0.0, 0.0]), radius) for cx in center_x_values]
    plot_solids(
        spheres,
        colors=['lightblue', 'lightcoral', 'lightgreen'],
        labels=[f'Center X={cx}' for cx in center_x_values],
        title='Example 2: Sphere Positions'
    )

    # Example 3: Transformed shapes
    center = jnp.array([0.0, 0.0, 0.0])
    size = jnp.array([1.0, 1.0, 1.0])

    original = box(center, size)
    translated = translate(original, jnp.array([2.0, 0.0, 0.0]))
    rotated = rotate(translated, jnp.array([0.0, 0.0, 1.0]), jnp.pi / 4)

    plot_solids(
        [original, translated, rotated],
        colors=['lightblue', 'lightcoral', 'lightgreen'],
        labels=['Original', 'Translated', 'Translated + Rotated'],
        title='Example 3: Transformations'
    )


if __name__ == "__main__":
    main()
