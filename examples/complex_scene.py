"""Example: Complex mechanical part using boolean operations."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import Box, Cylinder, Sphere


def plot_sdf_slice(sdf, title, ax, z_slice=0.0, extent=3.5, resolution=300):
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
    ax.contour(X, Y, distances, levels=[0], colors="black", linewidths=2.5)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.2)

    return contour


def create_bracket():
    """Create a mechanical bracket using CSG operations."""
    # Base plate
    base = Box(size=jnp.array([2.5, 2.0, 0.3]))

    # Mounting boss
    boss = Cylinder(radius=0.8, height=0.5)

    # Mounting holes (4 corners)
    hole_offset = 1.8
    hole1 = Cylinder(radius=0.2, height=1.0)
    # Note: would need transforms to position holes at corners
    # For now, just drill center hole
    center_hole = Cylinder(radius=0.3, height=1.0)

    # Combine
    bracket = (base | boss) - center_hole

    return bracket


def create_gear_blank():
    """Create a simple gear blank."""
    # Outer cylinder
    outer = Cylinder(radius=2.0, height=0.5)

    # Inner bore
    bore = Cylinder(radius=0.6, height=1.0)

    # Keyway (simplified as small box)
    # Note: would need transforms to position properly
    keyway = Box(size=jnp.array([0.8, 0.2, 0.6]))

    gear = outer - bore - keyway

    return gear


def create_drilled_sphere():
    """Create a sphere with multiple holes."""
    # Main sphere
    sphere = Sphere(radius=2.0)

    # Drill holes through X, Y, Z axes
    hole_x = Cylinder(radius=0.4, height=3.0)
    # Note: would need transforms to rotate for Y and Z holes
    # For now just show X-axis hole (cylinder along Z)

    drilled = sphere - hole_x

    return drilled


def main():
    """Visualize complex mechanical parts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Create parts
    bracket = create_bracket()
    gear = create_gear_blank()
    drilled = create_drilled_sphere()

    # Plot XY slices
    plot_sdf_slice(bracket, "Mounting Bracket", axes[0], z_slice=0.0)
    plot_sdf_slice(gear, "Gear Blank", axes[1], z_slice=0.0)
    contour = plot_sdf_slice(drilled, "Drilled Sphere", axes[2], z_slice=0.0)

    # Add colorbar
    fig.colorbar(contour, ax=axes, label="Signed Distance", shrink=0.8)

    plt.tight_layout()
    plt.savefig("assets/complex_parts.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/complex_parts.png")
    plt.show()


if __name__ == "__main__":
    main()
