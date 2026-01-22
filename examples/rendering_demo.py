"""Demonstrate rendering capabilities."""

import matplotlib.pyplot as plt

from jaxcad.primitives import Sphere, Box
from jaxcad.render import render_marching_cubes


def demo_rendering():
    """Show marching cubes rendering."""

    # Create a fun shape
    sphere = Sphere(radius=1.2).translate([0.5, 0.5, 0])
    box = Box(size=[1.5, 1.5, 1.5])
    shape = (sphere | box).twist(strength=0.8)

    fig = plt.figure(figsize=(8, 8))

    # Marching Cubes (3D)
    ax = fig.add_subplot(111, projection='3d')
    print("Rendering marching cubes mesh...")
    render_marching_cubes(
        shape,
        bounds=(-2, -2, -2),
        size=(4, 4, 4),
        resolution=50,
        ax=ax,
        color='cyan',
        alpha=0.8,
        title='Twisted Union - Marching Cubes'
    )
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('assets/rendering_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Generated: assets/rendering_demo.png")
    plt.show()


if __name__ == "__main__":
    demo_rendering()
