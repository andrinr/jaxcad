"""Showcase all available primitives."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.primitives import (
    Box, CappedCone, Capsule, Cone, Cylinder, Ellipsoid,
    HexagonalPrism, Octahedron, Plane, Pyramid, RoundBox,
    RoundedCylinder, Sphere, Torus, TriangularPrism
)
from jaxcad.render import render_marching_cubes


def main():
    """Visualize all primitive shapes."""
    fig = plt.figure(figsize=(24, 18))

    primitives = [
        (Sphere(radius=1.2), "Sphere"),
        (Box(size=jnp.array([1.0, 1.0, 1.0])), "Box"),
        (RoundBox(size=jnp.array([1.0, 1.0, 1.0]), radius=0.2), "RoundBox"),
        (Cylinder(radius=0.8, height=1.2), "Cylinder"),
        (RoundedCylinder(radius=0.8, height=1.0, rounding=0.2), "RoundedCylinder"),
        (Cone(radius=1.0, height=1.5), "Cone"),
        (CappedCone(height=1.5, r1=1.0, r2=0.3), "CappedCone"),
        (Torus(major_radius=1.2, minor_radius=0.4), "Torus"),
        (Capsule(radius=0.5, height=1.0), "Capsule"),
        (Ellipsoid(radii=jnp.array([1.2, 0.8, 0.6])), "Ellipsoid"),
        (Octahedron(size=1.2), "Octahedron"),
        (Pyramid(height=1.5), "Pyramid"),
        (HexagonalPrism(h=jnp.array([1.0, 1.2])), "HexagonalPrism"),
        (TriangularPrism(h=jnp.array([1.0, 1.2, 0.6])), "TriangularPrism"),
        (Plane(normal=jnp.array([0.0, 1.0, 0.0]), distance=0.0), "Plane"),
    ]

    extent = 2.0
    for idx, (primitive, name) in enumerate(primitives):
        ax = fig.add_subplot(4, 4, idx + 1, projection='3d')
        print(f"Rendering {idx+1}/{len(primitives)}: {name}...")

        try:
            # Adjust bounds for plane
            if name == "Plane":
                bounds = (-extent, -extent, -2.0)
                size = (2*extent, 2*extent, 2.0)
            else:
                bounds = (-extent, -extent, -extent)
                size = (2*extent, 2*extent, 2*extent)

            render_marching_cubes(
                primitive,
                bounds=bounds,
                size=size,
                resolution=60,
                ax=ax,
                color='steelblue',
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
    plt.savefig("assets/all_primitives.png", dpi=150, bbox_inches="tight")
    print("\nSaved: assets/all_primitives.png")
    plt.show()


if __name__ == "__main__":
    main()
