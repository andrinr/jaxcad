"""Showcase all available operations."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import Sphere, Box
from jaxcad.boolean import union, intersection, difference, xor
from jaxcad.render import render_marching_cubes


def main():
    """Visualize all operations."""
    fig = plt.figure(figsize=(24, 18))

    # Base shapes for boolean ops
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([0.8, 0.8, 0.8])).translate([0.5, 0.0, 0.0])

    operations = [
        # Boolean operations
        (sphere | box, "Union (|)"),
        (sphere & box, "Intersection (&)"),
        (sphere - box, "Difference (-)"),
        (xor(sphere, box), "XOR"),
        (union(sphere, box, smoothness=0.3), "Smooth Union"),
        (intersection(sphere, box, smoothness=0.3), "Smooth Intersection"),
        (difference(sphere, box, smoothness=0.3), "Smooth Difference"),

        # Transform operations
        (sphere.translate([1.0, 0.5, 0.0]), "Translate"),
        (sphere.rotate('z', jnp.pi/4), "Rotate"),
        (sphere.scale(1.5), "Scale"),

        # Deformation operations
        (Sphere(radius=0.8).twist(strength=2.0), "Twist"),
        (Sphere(radius=0.8).taper(strength=0.5), "Taper"),
        (Box(size=jnp.array([0.6, 0.6, 0.6])).bend(strength=0.8), "Bend"),

        # Domain operations
        (sphere.symmetry(axis="x"), "Symmetry (X)"),
        (Sphere(radius=0.4).elongate(jnp.array([0.5, 0.0, 0.5])), "Elongation"),
        (box.round(0.2), "Rounding"),
        (sphere.onion(0.2), "Onion (Shell)"),
    ]

    extent = 2.5
    for idx, (shape, name) in enumerate(operations):
        if idx >= 16:  # Only show 4x4 grid
            break

        ax = fig.add_subplot(4, 4, idx + 1, projection='3d')
        print(f"Rendering {idx+1}/{len(operations)}: {name}...")

        try:
            render_marching_cubes(
                shape,
                bounds=(-extent, -extent, -extent),
                size=(2*extent, 2*extent, 2*extent),
                resolution=60,
                ax=ax,
                color='coral',
                alpha=0.85,
                title=name
            )

            ax.set_xlim([-extent, extent])
            ax.set_ylim([-extent, extent])
            ax.set_zlim([-extent, extent])
            ax.set_box_aspect([1, 1, 1])
        except Exception as e:
            print(f"  Warning: Failed to render {name}: {e}")
            ax.text(0, 0, 0, f"{name}\n(render failed)", ha='center')

    plt.tight_layout()
    plt.savefig("assets/all_operations.png", dpi=150, bbox_inches="tight")
    print("\nSaved: assets/all_operations.png")
    plt.show()


if __name__ == "__main__":
    main()
