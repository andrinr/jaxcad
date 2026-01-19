"""Example of using gradients for shape optimization.

This example demonstrates using JAX's autodiff to optimize shape parameters
to meet certain objectives (e.g., target volume, target vertex positions).
"""

import jax
import jax.numpy as jnp

from jaxcad import box, sphere
from jaxcad.viz import plot_solids


def loss_target_volume(size, target_volume):
    """Loss function: squared difference between box volume and target."""
    center = jnp.array([0.0, 0.0, 0.0])
    solid = box(center, size)

    # Compute volume from bounding box
    min_coords = jnp.min(solid.vertices, axis=0)
    max_coords = jnp.max(solid.vertices, axis=0)
    dims = max_coords - min_coords
    volume = jnp.prod(dims)

    return (volume - target_volume) ** 2


def loss_target_vertex_position(params, target_position, vertex_idx=0):
    """Loss function: squared distance from target position for a specific vertex."""
    center, radius = params

    center_3d = jnp.array([center, 0.0, 0.0])
    solid = sphere(center_3d, radius, resolution=16)

    vertex_position = solid.vertices[vertex_idx]
    distance = jnp.sum((vertex_position - target_position) ** 2)

    return distance


def optimize_box_size():
    """Optimize box size to match a target volume."""
    print("\n" + "=" * 60)
    print("Optimization Example 1: Box Size -> Target Volume")
    print("=" * 60)

    target_volume = 100.0
    initial_size = jnp.array([1.0, 1.0, 1.0])
    learning_rate = 0.01  # Reduced learning rate for stability

    size = initial_size
    print(f"\nTarget volume: {target_volume}")
    print(f"Initial size: {size}")

    # Gradient descent with clipping
    for i in range(100):
        loss = loss_target_volume(size, target_volume)
        grad = jax.grad(loss_target_volume)(size, target_volume)

        # Clip gradients to prevent explosion
        grad = jnp.clip(grad, -10.0, 10.0)
        size = size - learning_rate * grad
        # Ensure size stays positive
        size = jnp.maximum(size, 0.1)

        if i % 20 == 0:
            current_volume = jnp.prod(size)
            print(f"Iteration {i}: size={size}, volume={current_volume:.2f}, loss={loss:.4f}")

    final_volume = jnp.prod(size)
    print(f"\nFinal size: {size}")
    print(f"Final volume: {final_volume:.2f}")
    print(f"Target volume: {target_volume}")
    print(f"Error: {abs(final_volume - target_volume):.4f}")

    # Return optimized box for visualization
    center = jnp.array([0.0, 0.0, 0.0])
    return box(center, size)


def optimize_sphere_position():
    """Optimize sphere parameters to place a vertex at target position."""
    print("\n" + "=" * 60)
    print("Optimization Example 2: Sphere Parameters -> Target Vertex Position")
    print("=" * 60)

    target_position = jnp.array([5.0, 2.0, 0.0])
    initial_params = (0.0, 1.0)  # (center_x, radius)
    learning_rate = 0.05

    params = initial_params
    print(f"\nTarget vertex position: {target_position}")
    print(f"Initial params (center_x, radius): {params}")

    # Gradient descent with JAX optimizer
    for i in range(100):
        loss = loss_target_vertex_position(params, target_position)
        grad = jax.grad(loss_target_vertex_position)(params, target_position)

        # Update parameters
        center, radius = params
        grad_center, grad_radius = grad
        center = center - learning_rate * grad_center
        radius = radius - learning_rate * grad_radius
        params = (center, radius)

        if i % 20 == 0:
            center_3d = jnp.array([center, 0.0, 0.0])
            solid = sphere(center_3d, radius, resolution=16)
            current_position = solid.vertices[0]
            print(
                f"Iteration {i}: center_x={center:.2f}, radius={radius:.2f}, "
                f"vertex={current_position}, loss={loss:.4f}"
            )

    # Final result
    center, radius = params
    center_3d = jnp.array([center, 0.0, 0.0])
    solid = sphere(center_3d, radius, resolution=16)
    final_position = solid.vertices[0]

    print(f"\nFinal params: center_x={center:.2f}, radius={radius:.2f}")
    print(f"Final vertex position: {final_position}")
    print(f"Target position: {target_position}")
    print(f"Distance error: {jnp.linalg.norm(final_position - target_position):.4f}")

    # Return optimized sphere for visualization
    return solid


def main():
    print("=" * 60)
    print("JaxCAD: Shape Optimization using Gradients")
    print("=" * 60)

    optimized_box = optimize_box_size()
    optimized_sphere = optimize_sphere_position()

    print("\n" + "=" * 60)
    print("Optimization completed successfully!")
    print("=" * 60)

    # Visualize optimization results
    print("\nVisualizing optimization results...")

    # Show initial vs optimized box
    initial_box = box(jnp.array([-3.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0]))
    # Reconstruct optimized box at offset position
    offset = jnp.array([3.0, 0.0, 0.0])
    optimized_box_offset = box(offset, optimized_box.vertices.max(axis=0) - optimized_box.vertices.min(axis=0))

    plot_solids(
        [initial_box, optimized_box_offset],
        colors=['lightcoral', 'lightgreen'],
        labels=['Initial (Volume=1)', 'Optimized (Volume≈100)'],
        title='Optimization 1: Box Size → Target Volume'
    )

    # Show initial vs optimized sphere
    initial_sphere = sphere(jnp.array([0.0, 0.0, 0.0]), 1.0, resolution=16)
    plot_solids(
        [initial_sphere, optimized_sphere],
        colors=['lightcoral', 'lightgreen'],
        labels=['Initial (center=0, r=1)', 'Optimized (center≈5, r≈0)'],
        title='Optimization 2: Sphere Position → Target Vertex'
    )


if __name__ == "__main__":
    main()
