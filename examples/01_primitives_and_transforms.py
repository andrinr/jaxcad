"""Example 1: Primitives, Transforms, and 3D Rendering

This example demonstrates:
- Creating basic SDF primitives (Sphere, Box, Cylinder)
- Applying transforms (translate, scale)
- Boolean operations (union, difference)
- 3D rendering using marching cubes
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.sdf.primitives import Sphere, Box, Cylinder
from jaxcad.sdf.boolean import Union, Difference
from jaxcad.render import render_marching_cubes


def create_scene():
    """Create a scene with multiple primitives and transforms."""
    print("Creating scene with primitives and transforms...")

    # Base platform - large flat box
    platform = Box(size=jnp.array([3.0, 3.0, 0.2]))
    platform = platform.translate(offset=jnp.array([0.0, 0.0, -0.5]))

    # Central sphere
    sphere = Sphere(radius=0.8)
    sphere = sphere.translate(offset=jnp.array([0.0, 0.0, 0.3]))

    # Four pillars at corners (cylinders)
    pillar = Cylinder(radius=0.15, height=0.6)

    pillar1 = pillar.translate(offset=jnp.array([1.5, 1.5, -0.3]))
    pillar2 = pillar.translate(offset=jnp.array([-1.5, 1.5, -0.3]))
    pillar3 = pillar.translate(offset=jnp.array([1.5, -1.5, -0.3]))
    pillar4 = pillar.translate(offset=jnp.array([-1.5, -1.5, -0.3]))

    # Small box on top, scaled and translated
    top_box = Box(size=jnp.array([0.3, 0.3, 0.3]))
    top_box = top_box.scale(scale=1.5)
    top_box = top_box.translate(offset=jnp.array([0.0, 0.0, 1.2]))

    # Create hole in sphere using difference
    hole = Cylinder(radius=0.4, height=1.0)
    sphere_with_hole = Difference(sphere, hole)

    # Union everything together
    scene = Union(platform, pillar1)
    scene = Union(scene, pillar2)
    scene = Union(scene, pillar3)
    scene = Union(scene, pillar4)
    scene = Union(scene, sphere_with_hole)
    scene = Union(scene, top_box)

    print("✓ Scene created with 8 objects")
    return scene


def render_scene(scene):
    """Render the scene from multiple angles using marching cubes."""
    print("Rendering scene with marching cubes...")

    fig = plt.figure(figsize=(15, 5))

    # View 1: Front view
    ax1 = fig.add_subplot(131, projection='3d')
    render_marching_cubes(
        scene,
        bounds=(-3, -3, -1),
        size=(6, 6, 3),
        resolution=60,
        ax=ax1,
        color='#06A77D',
        alpha=0.8
    )
    ax1.set_title('Front View', fontsize=14, fontweight='bold')
    ax1.view_init(elev=15, azim=45)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # View 2: Top view
    ax2 = fig.add_subplot(132, projection='3d')
    render_marching_cubes(
        scene,
        bounds=(-3, -3, -1),
        size=(6, 6, 3),
        resolution=60,
        ax=ax2,
        color='#2E86AB',
        alpha=0.8
    )
    ax2.set_title('Top View', fontsize=14, fontweight='bold')
    ax2.view_init(elev=90, azim=0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # View 3: Isometric view
    ax3 = fig.add_subplot(133, projection='3d')
    render_marching_cubes(
        scene,
        bounds=(-3, -3, -1),
        size=(6, 6, 3),
        resolution=60,
        ax=ax3,
        color='#F18F01',
        alpha=0.8
    )
    ax3.set_title('Isometric View', fontsize=14, fontweight='bold')
    ax3.view_init(elev=30, azim=135)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig('examples/output/primitives_and_transforms.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: examples/output/primitives_and_transforms.png")
    plt.close()


def main():
    """Main execution."""
    print("=" * 80)
    print("JAXcad Example 1: Primitives, Transforms, and 3D Rendering")
    print("=" * 80)
    print()

    scene = create_scene()
    print()
    render_scene(scene)

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
