"""Example: Visualizing SDF primitives."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from jaxcad.primitives import Box, Capsule, Cone, Cylinder, Sphere, Torus


def sdf_to_mesh(sdf, extent=2.5, resolution=100):
    """Convert SDF to mesh using marching cubes.

    Args:
        sdf: SDF function to evaluate
        extent: Spatial extent in each dimension
        resolution: Grid resolution

    Returns:
        verts: Vertex positions (N, 3)
        faces: Triangle faces (M, 3)
    """
    # Create 3D grid
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    z = np.linspace(-extent, extent, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Evaluate SDF on grid
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    volume = np.array(sdf(jnp.array(points))).reshape(X.shape)

    # Extract mesh at level=0 (surface)
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=(
        x[1] - x[0], y[1] - y[0], z[1] - z[0]
    ))

    # Offset vertices to match grid coordinates
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

    # Set equal aspect ratio
    extent = 2.5
    ax.set_xlim([-extent, extent])
    ax.set_ylim([-extent, extent])
    ax.set_zlim([-extent, extent])


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

    for idx, (primitive, name) in enumerate(primitives):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        print(f"Generating mesh for: {name}...")
        verts, faces = sdf_to_mesh(primitive, extent=2.5, resolution=80)
        plot_mesh(verts, faces, name, ax)

    plt.tight_layout()
    plt.savefig("assets/primitives.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/primitives.png")
    plt.show()


if __name__ == "__main__":
    main()
