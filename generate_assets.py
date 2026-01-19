"""Generate visual assets for README."""

import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from jaxcad import box, sphere, translate
from jaxcad.boolean import merge
from jaxcad.modifications import taper, twist
from jaxcad.operations import array_circular, extrude, revolve, sweep
from jaxcad.sketch import circle, polygon, rectangle
from jaxcad.viz import plot_solid, plot_solids


def save_plot(filename):
    """Save current plot to assets directory."""
    plt.savefig(f'assets/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved assets/{filename}")


def generate_basic_primitives():
    """Generate basic primitives visualization."""
    fig = plt.figure(figsize=(12, 4))

    # Box
    ax1 = fig.add_subplot(131, projection='3d')
    solid = box(jnp.zeros(3), jnp.array([2., 2., 2.]))
    plot_solid(solid, ax=ax1, show=False, title='Box', color='lightblue')

    # Sphere
    ax2 = fig.add_subplot(132, projection='3d')
    solid = sphere(jnp.zeros(3), 1.5, resolution=32)
    plot_solid(solid, ax=ax2, show=False, title='Sphere', color='lightcoral')

    # Transformed
    ax3 = fig.add_subplot(133, projection='3d')
    solid = box(jnp.zeros(3), jnp.array([1.5, 1.5, 1.5]))
    solid = translate(solid, jnp.array([0., 0., 1.]))
    plot_solid(solid, ax=ax3, show=False, title='Translated Box', color='lightgreen')

    save_plot('primitives.png')


def generate_extrude():
    """Generate extrude example."""
    profile = rectangle(jnp.zeros(2), width=2.0, height=1.0)
    solid = extrude(profile, height=3.0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_solid(solid, ax=ax, show=False, title='Extruded Rectangle', color='lightblue')
    save_plot('extrude.png')


def generate_revolve():
    """Generate revolve example (vase)."""
    profile_points = jnp.array([
        [0.5, 0.0],
        [1.0, 0.5],
        [0.8, 1.5],
        [1.2, 2.5],
        [0.5, 3.0],
    ])
    profile = polygon(profile_points, closed=False)
    vase = revolve(profile, angle=2 * jnp.pi, resolution=32)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_solid(vase, ax=ax, show=False, title='Revolved Vase', color='lightcoral')
    save_plot('revolve.png')


def generate_sweep():
    """Generate sweep example (spring)."""
    profile = circle(jnp.zeros(2), 0.3, resolution=8)
    t = jnp.linspace(0, 4 * jnp.pi, 32)
    helix_path = jnp.stack([jnp.cos(t), jnp.sin(t), t / 2], axis=1)
    spring = sweep(profile, helix_path)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_solid(spring, ax=ax, show=False, title='Swept Spring', color='lightyellow')
    save_plot('sweep.png')


def generate_array():
    """Generate circular array example (wheel)."""
    spoke = box(jnp.array([2.0, 0.0, 0.0]), jnp.array([1.0, 0.2, 0.2]))
    wheel = array_circular(spoke, jnp.array([0., 0., 1.]), jnp.zeros(3), count=8)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_solid(wheel, ax=ax, show=False, title='Circular Array (Wheel)', color='lightpink')
    save_plot('array.png')


def generate_twist():
    """Generate twist modification example."""
    base = box(jnp.zeros(3), jnp.array([1.0, 1.0, 4.0]))
    twisted = twist(base, jnp.array([0., 0., 1.]), jnp.pi / 2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_solid(twisted, ax=ax, show=False, title='Twisted Box', color='lightcyan')
    save_plot('twist.png')


def generate_complex():
    """Generate complex combined example."""
    profile = circle(jnp.zeros(2), 2.0, resolution=32)
    tower = extrude(profile, height=10.0)
    tower = taper(tower, jnp.array([0., 0., 1.]), scale_top=0.4)
    tower = twist(tower, jnp.array([0., 0., 1.]), jnp.pi)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_solid(tower, ax=ax, show=False, title='Twisted Tapered Tower', color='plum')
    save_plot('complex.png')


def main():
    print("Generating visual assets for README...")
    print("-" * 60)

    generate_basic_primitives()
    generate_extrude()
    generate_revolve()
    generate_sweep()
    generate_array()
    generate_twist()
    generate_complex()

    print("-" * 60)
    print("All assets generated successfully!")


if __name__ == "__main__":
    main()
