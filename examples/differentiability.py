"""Example: Differentiability and gradient-based optimization."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import Cylinder, Sphere


def volume_estimate(sdf, bounds=2.0, resolution=50):
    """Estimate volume by sampling SDF on a grid."""
    points = jnp.linspace(-bounds, bounds, resolution)
    grid = jnp.stack(jnp.meshgrid(points, points, points), axis=-1)
    grid = grid.reshape(-1, 3)

    distances = sdf(grid)
    inside = distances < 0

    cell_volume = (2 * bounds / resolution) ** 3
    return jnp.sum(inside) * cell_volume


def example_gradient():
    """Compute gradient of sphere volume w.r.t. radius."""
    print("=" * 60)
    print("Example 1: Gradient of sphere volume w.r.t. radius")
    print("=" * 60)

    def sphere_volume(radius):
        sphere = Sphere(radius=radius)
        return volume_estimate(sphere)

    radius = 1.0
    vol = sphere_volume(radius)
    grad = jax.grad(sphere_volume)(radius)

    analytical_volume = (4 / 3) * jnp.pi * radius**3
    analytical_grad = 4 * jnp.pi * radius**2

    print(f"\nRadius: {radius}")
    print(f"Estimated volume: {vol:.4f}")
    print(f"Analytical volume: {analytical_volume:.4f}")
    print(f"\nEstimated ∂V/∂r: {grad:.4f}")
    print(f"Analytical ∂V/∂r (surface area): {analytical_grad:.4f}")
    print(f"Error: {abs(grad - analytical_grad):.4f}")


def example_optimization():
    """Optimize to match target volume."""
    print("\n" + "=" * 60)
    print("Example 2: Optimize sphere radius to match target volume")
    print("=" * 60)

    target_volume = 20.0

    def loss(radius):
        sphere = Sphere(radius=radius)
        vol = volume_estimate(sphere, resolution=40)
        return (vol - target_volume) ** 2

    # Gradient descent
    radius = 1.0
    learning_rate = 0.01
    history = []

    print(f"\nTarget volume: {target_volume:.2f}")
    print(f"Initial radius: {radius:.4f}\n")

    for i in range(20):
        loss_val = loss(radius)
        grad = jax.grad(loss)(radius)

        history.append((radius, loss_val))

        if i % 5 == 0:
            vol = volume_estimate(Sphere(radius=radius), resolution=40)
            print(f"Iteration {i:2d}: radius={radius:.4f}, volume={vol:.2f}, loss={loss_val:.4f}")

        radius = radius - learning_rate * grad

    final_vol = volume_estimate(Sphere(radius=radius), resolution=40)
    print(f"\nFinal radius: {radius:.4f}")
    print(f"Final volume: {final_vol:.2f}")
    print(f"Target volume: {target_volume:.2f}")

    return history


def example_composite_gradient():
    """Gradient through composite boolean operations."""
    print("\n" + "=" * 60)
    print("Example 3: Gradient through boolean operations")
    print("=" * 60)

    def drilled_sphere_volume(hole_radius):
        sphere = Sphere(radius=2.0)
        drill = Cylinder(radius=hole_radius, height=3.0)
        drilled = sphere - drill
        return volume_estimate(drilled, resolution=40)

    hole_radius = 0.5
    vol = drilled_sphere_volume(hole_radius)
    grad = jax.grad(drilled_sphere_volume)(hole_radius)

    print(f"\nSphere radius: 2.0")
    print(f"Hole radius: {hole_radius}")
    print(f"Drilled volume: {vol:.4f}")
    print(f"∂V/∂(hole_radius): {grad:.4f}")
    print(f"(Negative gradient confirms larger hole → smaller volume)")


def plot_optimization(history):
    """Plot optimization history."""
    radii = [h[0] for h in history]
    losses = [h[1] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(radii, "o-", linewidth=2, markersize=6)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Radius")
    ax1.set_title("Radius over optimization")
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(losses, "o-", linewidth=2, markersize=6, color="red")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("Loss over optimization")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/optimization.png", dpi=150, bbox_inches="tight")
    print("\nSaved: assets/optimization.png")
    plt.show()


def main():
    """Run all differentiability examples."""
    example_gradient()
    history = example_optimization()
    example_composite_gradient()
    plot_optimization(history)


if __name__ == "__main__":
    main()
