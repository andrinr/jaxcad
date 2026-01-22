"""Showcase example: combining multiple primitives with deformations."""

from jaxcad.primitives import Sphere, Box, Cylinder, Torus
from jaxcad.render import render_marching_cubes
import matplotlib.pyplot as plt

# Example 1: Twisted union with subtracted spheres
shape1_base = (Sphere(radius=1.5).translate([0.5, 0, 0]) | Box(size=[2, 2, 2])).twist(strength=0.6)
hole1 = Sphere(radius=0.5).translate([1.5, 0, 0.5])
hole2 = Sphere(radius=0.4).translate([-0.8, 0.8, -0.5])
hole3 = Sphere(radius=0.3).translate([0, -1, 0.8])
shape1 = shape1_base - hole1 - hole2 - hole3

# Example 2: Torus with cylinders and twist
torus = Torus(major_radius=1.2, minor_radius=0.4)
cyl1 = Cylinder(radius=0.3, height=3).rotate('x', 1.57)
cyl2 = Cylinder(radius=0.3, height=3).rotate('y', 1.57)
shape2 = ((torus | cyl1 | cyl2) - Sphere(radius=0.6)).twist(strength=0.4)

# Example 3: Layered structure with bend
base = Box(size=[2.5, 2.5, 0.5]).translate([0, 0, -0.8])
pillars = (Cylinder(radius=0.3, height=2).translate([0.8, 0.8, 0]) |
           Cylinder(radius=0.3, height=2).translate([-0.8, 0.8, 0]) |
           Cylinder(radius=0.3, height=2).translate([0.8, -0.8, 0]) |
           Cylinder(radius=0.3, height=2).translate([-0.8, -0.8, 0]))
top = Sphere(radius=1.0).translate([0, 0, 1.2])
shape3 = (base | pillars | top).bend(strength=0.3)

# Create figure with three 3D subplots
fig = plt.figure(figsize=(18, 6))

print("Rendering shape 1...")
ax1 = fig.add_subplot(131, projection='3d')
render_marching_cubes(shape1, bounds=(-2.5, -2.5, -2.5), size=(5, 5, 5), resolution=50, ax=ax1, color='cyan', alpha=0.85)
ax1.set_title('Twisted Union with Holes', fontsize=14, fontweight='bold')
ax1.view_init(elev=20, azim=45)

print("Rendering shape 2...")
ax2 = fig.add_subplot(132, projection='3d')
render_marching_cubes(shape2, bounds=(-2.5, -2.5, -2.5), size=(5, 5, 5), resolution=50, ax=ax2, color='magenta', alpha=0.85)
ax2.set_title('Twisted Torus with Cylinders', fontsize=14, fontweight='bold')
ax2.view_init(elev=20, azim=45)

print("Rendering shape 3...")
ax3 = fig.add_subplot(133, projection='3d')
render_marching_cubes(shape3, bounds=(-2.5, -2.5, -2.5), size=(5, 5, 5), resolution=50, ax=ax3, color='orange', alpha=0.85)
ax3.set_title('Bent Layered Structure', fontsize=14, fontweight='bold')
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('assets/showcase_geometry.png', dpi=150, bbox_inches='tight')
print("Saved to assets/showcase_geometry.png")
plt.close()
