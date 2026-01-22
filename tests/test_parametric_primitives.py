"""Test that primitive parameters are extracted as constraints."""

import jax
import jax.numpy as jnp

from jaxcad.compiler.parametric import compile_parametric, optimize_parameters
from jaxcad.constraints import ConstraintSystem
from jaxcad.primitives import Sphere, Box, Cylinder


def test_extract_sphere_radius():
    """Test extracting sphere radius as a constraint."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Compile - should extract radius
    eval_fn = compile_parametric(sphere, cs)

    # Should have radius as a parameter
    assert len(cs.parameters) > 0

    # Check that radius was extracted
    radius_params = [p for p in cs.parameters if 'radius' in p.name]
    assert len(radius_params) == 1

    radius_param = radius_params[0]
    assert radius_param.value == 1.0
    assert radius_param.free  # Should be free by default

    print(f"\n{cs.summary()}")


def test_extract_box_size():
    """Test extracting box size as a constraint."""
    cs = ConstraintSystem()
    box = Box(size=jnp.array([1.0, 2.0, 3.0]))

    # Compile
    eval_fn = compile_parametric(box, cs)

    # Should have size parameter
    size_params = [p for p in cs.parameters if 'size' in p.name]
    assert len(size_params) == 1

    size_param = size_params[0]
    assert jnp.allclose(size_param.value, jnp.array([1.0, 2.0, 3.0]))
    assert size_param.free

    print(f"\n{cs.summary()}")


def test_extract_cylinder_params():
    """Test extracting cylinder radius and height."""
    cs = ConstraintSystem()
    cylinder = Cylinder(radius=0.5, height=2.0)

    # Compile
    eval_fn = compile_parametric(cylinder, cs)

    # Should have both radius and height
    radius_params = [p for p in cs.parameters if 'radius' in p.name]
    height_params = [p for p in cs.parameters if 'height' in p.name]

    assert len(radius_params) == 1
    assert len(height_params) == 1

    assert radius_params[0].value == 0.5
    assert height_params[0].value == 2.0

    print(f"\n{cs.summary()}")


def test_optimize_sphere_radius():
    """Test optimizing sphere radius to match target."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Target: sphere should have distance 0 at radius 2.0
    target_points = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    target_values = jnp.zeros(3)  # On surface

    print("\nOptimizing sphere radius to match target points at r=2.0...")

    optimized_params, loss_history = optimize_parameters(
        sphere,
        target_points,
        target_values,
        cs,
        num_iterations=50,
        learning_rate=0.1
    )

    # Get optimized radius
    radius_params = [p for p in cs.parameters if 'radius' in p.name]
    optimized_radius = float(radius_params[0].value)

    print(f"\nOptimized radius: {optimized_radius}")
    print(f"Expected: 2.0")
    print(f"Final loss: {loss_history[-1]:.6f}")

    # Should converge to radius ≈ 2.0
    assert abs(optimized_radius - 2.0) < 0.1


def test_all_parameters_free_by_default():
    """Test that all extracted parameters are free by default."""
    cs = ConstraintSystem()

    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))

    complex_sdf = (
        sphere.translate(jnp.array([1.0, 0.0, 0.0]))
        | box.rotate('z', jnp.pi/4)
    )

    eval_fn = compile_parametric(complex_sdf, cs)

    print(f"\n{cs.summary()}")

    # All parameters should be free by default
    assert len(cs.get_free_params()) == len(cs.parameters)
    assert len(cs.get_fixed_params()) == 0

    print(f"\n✓ All {len(cs.parameters)} parameters are FREE by default")


def test_mark_primitive_param_fixed():
    """Test marking a primitive parameter as fixed."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    eval_fn = compile_parametric(sphere, cs)

    # Initially free
    assert len(cs.get_free_params()) == 1

    # Mark radius as fixed
    radius_param = [p for p in cs.parameters if 'radius' in p.name][0]
    radius_param.fixed = True

    # Now should be fixed
    assert len(cs.get_free_params()) == 0
    assert len(cs.get_fixed_params()) == 1

    print(f"\n{cs.summary()}")
    print("\n✓ Successfully marked radius as FIXED")


def test_optimize_with_fixed_radius():
    """Test optimization with fixed radius and free position."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Translate sphere
    sdf = sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Compile - extracts both radius and offset
    eval_fn = compile_parametric(sdf, cs)

    print(f"\n{cs.summary()}")

    # Mark radius as fixed
    radius_param = [p for p in cs.parameters if 'radius' in p.name][0]
    radius_param.fixed = True

    print("\nMarked radius as FIXED")
    print(f"Free parameters: {len(cs.get_free_params())}")
    print(f"Fixed parameters: {len(cs.get_fixed_params())}")

    # Now optimize - should only optimize position, not radius
    target_points = jnp.array([[2.0, 0.0, 0.0]])
    target_values = jnp.array([0.0])

    initial_radius = float(radius_param.value)

    optimized_params, loss_history = optimize_parameters(
        sdf,
        target_points,
        target_values,
        cs,
        num_iterations=30,
        learning_rate=0.1
    )

    final_radius = float(radius_param.value)

    print(f"\nRadius before: {initial_radius}")
    print(f"Radius after: {final_radius}")
    print(f"✓ Radius unchanged (was FIXED)")

    # Radius should not have changed
    assert abs(final_radius - initial_radius) < 1e-6

    # Position should have optimized
    offset_param = [p for p in cs.parameters if 'translate' in p.name][0]
    print(f"Optimized offset: {offset_param.value}")
