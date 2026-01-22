"""Example: Transformations on SDFs."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from jaxcad.primitives import Box, Cylinder, Sphere


def sdf_to_mesh(sdf, extent=3.0, resolution=80):
    """Convert SDF to mesh using marching cubes."""
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    z = np.linspace(-extent, extent, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    volume = np.array(sdf(jnp.array(points))).reshape(X.shape)

    verts, faces, _, _ = measure.marching_cubes(
        volume, level=0.0,
        spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
    )

    verts = verts + np.array([-extent, -extent, -extent])
    return verts, faces


def plot_mesh(verts, faces, title, ax):
    """Plot mesh using matplotlib 3D."""
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color='lightblue', edgecolor='darkblue', linewidth=0.1, alpha=0.8
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    extent = 3.0
    ax.set_xlim([-extent, extent])
    ax.set_ylim([-extent, extent])
    ax.set_zlim([-extent, extent])


def main():
    """Demonstrate transformations with 3D visualization."""
    # Create base shapes
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))
    sphere = Sphere(radius=0.8)
    cylinder = Cylinder(radius=0.5, height=2.0)

    fig = plt.figure(figsize=(18, 12))

    shapes = [
        (box, "Box (Original)"),
        (box.translate(jnp.array([1.5, 0.0, 0.0])), "Translated"),
        (box.rotate('z', jnp.pi / 4), "Rotated 45°"),
        (sphere.scale(1.5), "Scaled 1.5x"),
        (
            cylinder
            .rotate('y', jnp.pi / 4)
            .translate(jnp.array([0.0, 0.0, 1.0])),
            "Rotated + Translated"
        ),
        (
            (sphere.scale(0.8) | box.rotate('z', jnp.pi / 4))
            .translate(jnp.array([1.0, 1.0, 0.0])),
            "Combined + Transformed"
        ),
    ]

    for idx, (shape, title) in enumerate(shapes):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        print(f"Generating mesh for: {title}...")
        verts, faces = sdf_to_mesh(shape, extent=3.0, resolution=70)
        plot_mesh(verts, faces, title, ax)

    plt.tight_layout()
    plt.savefig("assets/transforms.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/transforms.png")
    plt.show()


if __name__ == "__main__":
    main()
