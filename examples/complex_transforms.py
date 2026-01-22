"""Example: Complex transformations and deformations."""

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


def plot_mesh(verts, faces, title, ax, extent=3.0):
    """Plot mesh using matplotlib 3D."""
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        color='lightblue', edgecolor='darkblue', linewidth=0.1, alpha=0.8
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlim([-extent, extent])
    ax.set_ylim([-extent, extent])
    ax.set_zlim([-extent, extent])


def main():
    """Demonstrate complex transformations."""
    # Base shapes
    box = Box(size=jnp.array([0.5, 0.5, 1.5]))
    cylinder = Cylinder(radius=0.5, height=2.0)
    sphere = Sphere(radius=0.8)

    fig = plt.figure(figsize=(18, 12))

    shapes = [
        (box, "Box", 3.0),
        (box.twist('z', strength=2.0), "Twisted Box", 3.0),
        (cylinder.bend('z', strength=0.8), "Bent Cylinder", 4.0),
        (box.taper('z', strength=0.4), "Tapered Box", 3.0),
        (sphere.scale(0.6).repeat_finite(
            spacing=jnp.array([1.8, 1.8, 1.8]),
            count=jnp.array([3, 3, 1])
        ), "Repeated Spheres", 4.5),
        (
            box.scale(0.8).translate(jnp.array([1.0, 0.0, 0.0])).mirror('x'),
            "Translated + Mirrored",
            3.5
        ),
    ]

    for idx, (shape, title, extent) in enumerate(shapes):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        print(f"Generating mesh for: {title}...")
        try:
            verts, faces = sdf_to_mesh(shape, extent=extent, resolution=70)
            plot_mesh(verts, faces, title, ax, extent=extent)
        except Exception as e:
            print(f"  Warning: Failed to generate mesh - {e}")
            ax.text(0, 0, 0, f"Error:\n{str(e)[:50]}", ha='center')

    plt.tight_layout()
    plt.savefig("assets/complex_transforms.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/complex_transforms.png")
    plt.show()


if __name__ == "__main__":
    main()
