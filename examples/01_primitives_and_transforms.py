"""Example 1: Primitives, Transforms, and 3D Rendering

This example demonstrates:
- Creating basic SDF primitives (Sphere, Box, Cylinder)
- Applying transforms (translate, scale)
- Boolean operations (union, difference, intersection)
- 3D rendering using marching cubes
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.sdf.primitives import Sphere, Box, Cylinder
from jaxcad.sdf.boolean import Union, Difference, Intersection
from jaxcad.sdf.transforms import Translate
from jaxcad.render import render_marching_cubes


def create_scene1():
    """Scene 1: Union - Platform with spheres."""
    print("Creating scene 1: Union operations...")

    # Platform
    platform = Translate(Box(size=jnp.array([3.0, 3.0, 0.3])), offset=jnp.array([0.0, 0.0, -0.5]))

    # Three spheres at different positions
    sphere1 = Translate(Sphere(radius=0.6), offset=jnp.array([-1.0, 0.0, 0.0]))
    sphere2 = Translate(Sphere(radius=0.6), offset=jnp.array([1.0, 0.0, 0.0]))
    sphere3 = Translate(Sphere(radius=0.5), offset=jnp.array([0.0, 0.0, 0.8]))

    # Combine with union
    scene = Union(platform, sphere1)
    scene = Union(scene, sphere2)
    scene = Union(scene, sphere3)

    print("✓ Scene 1 created")
    return scene


def create_scene2():
    """Scene 2: Difference - Sphere with cylindrical holes."""
    print("Creating scene 2: Difference operations...")

    # Large sphere
    sphere = Sphere(radius=1.2)

    # Three cylinders as holes
    hole1 = Translate(Cylinder(radius=0.4, height=3.0), offset=jnp.array([0.0, 0.0, -1.5]))
    hole2 = Translate(Cylinder(radius=0.4, height=3.0), offset=jnp.array([0.0, -1.5, 0.0]))
    hole3 = Translate(Cylinder(radius=0.4, height=3.0), offset=jnp.array([-1.5, 0.0, 0.0]))

    # Subtract holes from sphere
    scene = Difference(sphere, hole1)
    scene = Difference(scene, hole2)
    scene = Difference(scene, hole3)

    print("✓ Scene 2 created")
    return scene


def create_scene3():
    """Scene 3: Intersection and scaling - Rounded cube."""
    print("Creating scene 3: Intersection and scaling...")

    # Box
    box = Box(size=jnp.array([1.5, 1.5, 1.5]))

    # Sphere to round the corners
    sphere = Sphere(radius=1.3)

    # Intersect to create rounded cube
    rounded_cube = Intersection(box, sphere)

    # Add a cylinder on top
    cylinder = Translate(Cylinder(radius=0.3, height=0.8), offset=jnp.array([0.0, 0.0, 0.75]))

    # Union with cylinder
    scene = Union(rounded_cube, cylinder)

    print("✓ Scene 3 created")
    return scene


def render_scenes():
    """Render three different scenes side by side."""
    print("Rendering scenes with marching cubes...")

    scene1 = create_scene1()
    scene2 = create_scene2()
    scene3 = create_scene3()

    fig = plt.figure(figsize=(16, 5))

    # Scene 1: Union
    ax1 = fig.add_subplot(131, projection='3d')
    render_marching_cubes(
        scene1,
        bounds=(-2, -2, -1),
        size=(4, 4, 2.5),
        resolution=50,
        ax=ax1,
        color='#06A77D',
        alpha=0.8
    )
    ax1.set_title('Union Operations', fontsize=14, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Scene 2: Difference
    ax2 = fig.add_subplot(132, projection='3d')
    render_marching_cubes(
        scene2,
        bounds=(-1.5, -1.5, -1.5),
        size=(3, 3, 3),
        resolution=50,
        ax=ax2,
        color='#2E86AB',
        alpha=0.8
    )
    ax2.set_title('Difference Operations', fontsize=14, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Scene 3: Intersection
    ax3 = fig.add_subplot(133, projection='3d')
    render_marching_cubes(
        scene3,
        bounds=(-1.5, -1.5, -1),
        size=(3, 3, 2.5),
        resolution=50,
        ax=ax3,
        color='#F18F01',
        alpha=0.8
    )
    ax3.set_title('Intersection & Transforms', fontsize=14, fontweight='bold')
    ax3.view_init(elev=20, azim=45)
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

    render_scenes()

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
