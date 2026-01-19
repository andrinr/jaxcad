"""Example: Visualizing SDF primitives."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm

from jaxcad.primitives import Box, Capsule, Cone, Cylinder, Sphere, Torus


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
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return contour


def main():
    """Visualize primitive shapes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    primitives = [
        (Sphere(radius=1.5), "Sphere"),
        (Box(size=jnp.array([1.2, 1.2, 1.2])), "Box"),
        (Cylinder(radius=1.0, height=1.5), "Cylinder"),
        (Cone(radius=1.2, height=2.0), "Cone"),
        (Torus(major_radius=1.5, minor_radius=0.5), "Torus"),
        (Capsule(radius=0.6, height=1.0), "Capsule"),
    ]

    for idx, (primitive, name) in enumerate(primitives):
        contour = plot_sdf_slice(primitive, name, axes[idx])

    # Add colorbar
    fig.colorbar(contour, ax=axes, label="Signed Distance", shrink=0.8)

    plt.tight_layout()
    plt.savefig("assets/primitives.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/primitives.png")
    plt.show()


if __name__ == "__main__":
    main()
