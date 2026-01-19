"""Examples demonstrating complex geometry creation with JaxCAD."""

import jax
import jax.numpy as jnp

from jaxcad.boolean import merge
from jaxcad.modifications import taper, thicken, twist
from jaxcad.operations import array_circular, array_linear, extrude, loft, revolve, sweep
from jaxcad.primitives import box, cylinder
from jaxcad.sketch import circle, rectangle, regular_polygon
from jaxcad.viz import plot_solid


def example_extrude():
    """Demonstrate extruding 2D profiles."""
    print("\n" + "=" * 60)
    print("Example 1: Extrude Operations")
    print("=" * 60)

    # Extrude a rectangle
    profile = rectangle(jnp.zeros(2), width=2.0, height=1.0)
    solid = extrude(profile, height=3.0)
    print(f"Extruded rectangle: {solid.vertices.shape[0]} vertices, {solid.faces.shape[0]} faces")

    # Extrude a circle
    profile = circle(jnp.zeros(2), radius=1.0, resolution=32)
    cylinder_solid = extrude(profile, height=2.0)
    print(f"Extruded circle (cylinder): {cylinder_solid.vertices.shape[0]} vertices")

    # Extrude a hexagon
    profile = regular_polygon(jnp.zeros(2), radius=1.5, n_sides=6)
    hex_prism = extrude(profile, height=2.5)
    print(f"Extruded hexagon: {hex_prism.vertices.shape[0]} vertices")

    return solid


def example_revolve():
    """Demonstrate revolving profiles."""
    print("\n" + "=" * 60)
    print("Example 2: Revolve Operations")
    print("=" * 60)

    # Create a vase-like shape
    # Profile in (r, z) coordinates
    profile_points = jnp.array(
        [
            [0.5, 0.0],
            [1.0, 0.5],
            [0.8, 1.5],
            [1.2, 2.5],
            [0.5, 3.0],
        ]
    )

    from jaxcad.sketch import polygon

    profile = polygon(profile_points, closed=False)
    vase = revolve(profile, angle=2 * jnp.pi, resolution=32)
    print(f"Revolved vase: {vase.vertices.shape[0]} vertices, {vase.faces.shape[0]} faces")

    # Partial revolution (270 degrees)
    partial = revolve(profile, angle=3 * jnp.pi / 2, resolution=24)
    print(f"Partial revolution: {partial.vertices.shape[0]} vertices")

    return vase


def example_loft():
    """Demonstrate lofting between profiles."""
    print("\n" + "=" * 60)
    print("Example 3: Loft Operations")
    print("=" * 60)

    # Loft from square to circle
    rectangle(jnp.zeros(2), 2.0, 2.0)
    circ = circle(jnp.zeros(2), 1.0, resolution=32)

    # Need to match point counts - use circle resolution for square too
    square_circle = circle(jnp.zeros(2), 1.414, resolution=32)  # Approximate square

    heights = jnp.array([0.0, 3.0])
    morphed = loft([square_circle, circ], heights)
    print(f"Lofted square->circle: {morphed.vertices.shape[0]} vertices")

    # Three-level loft
    profile1 = circle(jnp.zeros(2), 1.5, resolution=16)
    profile2 = circle(jnp.zeros(2), 1.0, resolution=16)
    profile3 = circle(jnp.zeros(2), 0.5, resolution=16)

    heights = jnp.array([0.0, 1.5, 3.0])
    tapered_shape = loft([profile1, profile2, profile3], heights)
    print(f"Three-level loft: {tapered_shape.vertices.shape[0]} vertices")

    return morphed


def example_sweep():
    """Demonstrate sweeping along a path."""
    print("\n" + "=" * 60)
    print("Example 4: Sweep Operations")
    print("=" * 60)

    # Sweep a circle along a helix
    profile = circle(jnp.zeros(2), 0.3, resolution=8)

    # Create helix path
    t = jnp.linspace(0, 4 * jnp.pi, 32)
    helix_path = jnp.stack([jnp.cos(t), jnp.sin(t), t / 2], axis=1)

    spring = sweep(profile, helix_path)
    print(f"Swept spring: {spring.vertices.shape[0]} vertices, {spring.faces.shape[0]} faces")

    # Sweep along a wavy path
    t = jnp.linspace(0, 4 * jnp.pi, 32)
    wave_path = jnp.stack([t / 2, jnp.sin(t) * 0.5, jnp.cos(t * 2) * 0.3], axis=1)

    wave_tube = sweep(profile, wave_path)
    print(f"Swept wave tube: {wave_tube.vertices.shape[0]} vertices")

    return spring


def example_arrays():
    """Demonstrate array operations."""
    print("\n" + "=" * 60)
    print("Example 5: Array Operations")
    print("=" * 60)

    # Linear array
    base_box = box(jnp.zeros(3), jnp.array([0.5, 0.5, 0.5]))
    linear = array_linear(base_box, jnp.array([1.0, 0.0, 0.0]), count=5, spacing=1.2)
    print(f"Linear array: {linear.vertices.shape[0]} vertices (5 copies)")

    # Circular array
    spoke = box(jnp.array([2.0, 0.0, 0.0]), jnp.array([1.0, 0.2, 0.2]))
    wheel = array_circular(spoke, jnp.array([0.0, 0.0, 1.0]), jnp.zeros(3), count=8)
    print(f"Circular array: {wheel.vertices.shape[0]} vertices (8 spokes)")

    return wheel


def example_modifications():
    """Demonstrate modification operations."""
    print("\n" + "=" * 60)
    print("Example 6: Modification Operations")
    print("=" * 60)

    # Twist
    base = box(jnp.zeros(3), jnp.array([1.0, 1.0, 4.0]))
    twisted = twist(base, jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2)
    print(f"Twisted box: {twisted.vertices.shape[0]} vertices")

    # Taper
    cyl = cylinder(jnp.zeros(3), 1.0, 4.0, resolution=32)
    tapered = taper(cyl, jnp.array([0.0, 0.0, 1.0]), scale_top=0.3)
    print(f"Tapered cylinder: {tapered.vertices.shape[0]} vertices")

    # Thicken
    thin_profile = rectangle(jnp.zeros(2), 3.0, 2.0)
    thin_solid = extrude(thin_profile, 0.05)
    thickened = thicken(thin_solid, thickness=0.3)
    print(f"Thickened solid: {thickened.vertices.shape[0]} vertices")

    return twisted


def example_combine():
    """Demonstrate combining operations."""
    print("\n" + "=" * 60)
    print("Example 7: Combining Multiple Operations")
    print("=" * 60)

    # Create a complex shape: twisted tapered tower with circular base
    profile = circle(jnp.zeros(2), 2.0, resolution=32)
    tower = extrude(profile, height=10.0)

    # Apply taper
    tower = taper(tower, jnp.array([0.0, 0.0, 1.0]), scale_top=0.4)

    # Apply twist
    tower = twist(tower, jnp.array([0.0, 0.0, 1.0]), jnp.pi)

    print(f"Complex tower: {tower.vertices.shape[0]} vertices")

    # Create base platform
    base_profile = regular_polygon(jnp.zeros(2), radius=3.0, n_sides=8)
    base = extrude(base_profile, height=0.5)

    # Merge tower and base
    complete = merge(tower, base)
    print(f"Complete structure: {complete.vertices.shape[0]} vertices")

    return complete


def example_differentiability():
    """Demonstrate gradients through complex operations."""
    print("\n" + "=" * 60)
    print("Example 8: Gradients Through Complex Operations")
    print("=" * 60)

    def complex_volume(params):
        """Create complex shape and compute volume proxy."""
        width, height, twist_angle = params

        # Create and extrude profile
        profile = rectangle(jnp.zeros(2), width, width)
        solid = extrude(profile, height)

        # Apply twist
        solid = twist(solid, jnp.array([0.0, 0.0, 1.0]), twist_angle)

        # Compute bounding box volume as proxy
        min_coords = jnp.min(solid.vertices, axis=0)
        max_coords = jnp.max(solid.vertices, axis=0)
        dims = max_coords - min_coords
        return jnp.prod(dims)

    params = jnp.array([2.0, 3.0, jnp.pi / 4])

    # Compute gradient
    grad = jax.grad(complex_volume)(params)
    volume = complex_volume(params)

    print(f"Parameters: width={params[0]:.2f}, height={params[1]:.2f}, twist={params[2]:.2f}")
    print(f"Volume: {volume:.2f}")
    print(f"Gradient: {grad}")
    print("✓ Gradients computed successfully through extrude + twist operations!")


def main():
    print("=" * 60)
    print("JaxCAD: Complex Geometry Examples")
    print("=" * 60)

    extruded = example_extrude()
    vase = example_revolve()
    example_loft()
    spring = example_sweep()
    wheel = example_arrays()
    twisted = example_modifications()
    tower = example_combine()
    example_differentiability()

    print("\n" + "=" * 60)
    print("All complex geometry examples completed successfully!")
    print("=" * 60)

    # Visualize selected examples
    print("\nVisualizing examples...")
    plot_solid(extruded, color="lightblue", title="Example 1: Extruded Rectangle")
    plot_solid(vase, color="lightcoral", title="Example 2: Revolved Vase")
    plot_solid(spring, color="lightyellow", title="Example 4: Swept Spring")
    plot_solid(wheel, color="lightpink", title="Example 5: Circular Array (Wheel)")
    plot_solid(twisted, color="lightcyan", title="Example 6: Twisted Box")
    plot_solid(tower, color="plum", title="Example 7: Complex Tower")


if __name__ == "__main__":
    main()
