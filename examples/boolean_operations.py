"""Example: Boolean operations (CSG)."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import Box, Cylinder, Sphere
from jaxcad.render import render_marching_cubes


def main():
    """Demonstrate CSG boolean operations."""
    fig = plt.figure(figsize=(18, 12))

    # Define primitives
    sphere = Sphere(radius=1.2)
    box = Box(size=jnp.array([0.7, 1.5, 1.3]))
    cylinder = Cylinder(radius=0.6, height=3.0)

    operations = [
        (sphere, "Sphere"),
        (box, "Box"),
        (cylinder, "Cylinder"),
        (sphere | box, "Union: Sphere | Box"),
        (sphere & box, "Intersection: Sphere & Box"),
        (sphere - cylinder, "Difference: Sphere - Cylinder"),
    ]

    extent = 2.5
    for idx, (shape, name) in enumerate(operations):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        print(f"Generating mesh for: {name}...")

        # Use new vectorized rendering
        render_marching_cubes(
            shape,
            bounds=(-extent, -extent, -extent),
            size=(2*extent, 2*extent, 2*extent),
            resolution=80,
            ax=ax,
            color='cyan' if idx >= 3 else 'lightblue',
            alpha=0.8,
            title=name
        )

        # Set view limits
        ax.set_xlim([-extent, extent])
        ax.set_ylim([-extent, extent])
        ax.set_zlim([-extent, extent])
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig("assets/boolean_operations.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/boolean_operations.png")
    plt.show()


if __name__ == "__main__":
    main()
