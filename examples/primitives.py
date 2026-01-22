"""Example: Visualizing SDF primitives."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import Box, Capsule, Cone, Cylinder, Sphere, Torus
from jaxcad.render import render_marching_cubes


def main():
    """Visualize primitive shapes with 3D mesh visualization."""
    fig = plt.figure(figsize=(18, 12))

    primitives = [
        (Sphere(radius=1.5), "Sphere"),
        (Box(size=jnp.array([1.2, 1.2, 1.2])), "Box"),
        (Cylinder(radius=1.0, height=1.5), "Cylinder"),
        (Cone(radius=1.2, height=2.0), "Cone"),
        (Torus(major_radius=1.5, minor_radius=0.5), "Torus"),
        (Capsule(radius=0.6, height=1.0), "Capsule"),
    ]

    extent = 2.5
    for idx, (primitive, name) in enumerate(primitives):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        print(f"Generating mesh for: {name}...")

        # Use new vectorized rendering
        render_marching_cubes(
            primitive,
            bounds=(-extent, -extent, -extent),
            size=(2*extent, 2*extent, 2*extent),
            resolution=80,
            ax=ax,
            color='lightblue',
            alpha=0.8,
            title=name
        )

        # Set view limits
        ax.set_xlim([-extent, extent])
        ax.set_ylim([-extent, extent])
        ax.set_zlim([-extent, extent])
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig("assets/primitives.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/primitives.png")
    plt.show()


if __name__ == "__main__":
    main()
