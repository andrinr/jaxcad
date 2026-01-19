"""Example: Boolean operations (CSG) on SDFs."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import Box, Cylinder, Sphere


def plot_sdf_slice(sdf, title, ax, z_slice=0.0, extent=2.5, resolution=200):
    """Plot a 2D slice of an SDF."""
    x = jnp.linspace(-extent, extent, resolution)
    y = jnp.linspace(-extent, extent, resolution)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.full_like(X, z_slice)

    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    distances = sdf(points).reshape(X.shape)

    # Plot with contours
    levels = jnp.linspace(-2, 2, 21)
    contour = ax.contourf(X, Y, distances, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(X, Y, distances, levels=[0], colors="black", linewidths=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return contour


def main():
    """Demonstrate boolean operations."""
    sphere = Sphere(radius=1.5)
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))
    cylinder = Cylinder(radius=0.5, height=2.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Basic shapes
    plot_sdf_slice(sphere, "Sphere", axes[0])
    plot_sdf_slice(box, "Box", axes[1])
    plot_sdf_slice(cylinder, "Cylinder", axes[2])

    # Boolean operations
    union = sphere | box
    intersection = sphere & box
    difference = sphere - cylinder

    contour = plot_sdf_slice(union, "Union: Sphere | Box", axes[3])
    plot_sdf_slice(intersection, "Intersection: Sphere & Box", axes[4])
    plot_sdf_slice(difference, "Difference: Sphere - Cylinder", axes[5])

    # Add colorbar
    fig.colorbar(contour, ax=axes, label="Signed Distance", shrink=0.8)

    plt.tight_layout()
    plt.savefig("assets/boolean_operations.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/boolean_operations.png")
    plt.show()


if __name__ == "__main__":
    main()
