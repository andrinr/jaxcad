"""Tests for constraint-based parametric modeling."""

import jax
import jax.numpy as jnp

from jaxcad.constraints import ConstraintSystem, Point, Distance, Angle
from jaxcad.compiler.parametric import compile_parametric, optimize_parameters
from jaxcad.primitives import Sphere


def test_parameter_creation():
    """Test creating parameters."""
    cs = ConstraintSystem()

    # Create different parameter types
    p1 = cs.point([1.0, 0.0, 0.0], free=True, name="origin")
    d1 = cs.distance(2.0, free=False, name="radius")
    a1 = cs.angle(jnp.pi/4, free=True, name="rotation")

    assert len(cs.parameters) == 3
    assert len(cs.get_free_params()) == 2  # p1 and a1
    assert len(cs.get_fixed_params()) == 1  # d1

    print(f"\n{cs.summary()}")


def test_vector_conversion():
    """Test converting parameters to/from vectors."""
    cs = ConstraintSystem()

    p1 = cs.point([1.0, 2.0, 3.0], free=True)
    p2 = cs.point([4.0, 5.0, 6.0], free=True)
    d1 = cs.distance(7.0, free=False)

    # Convert free params to vector
    vec = cs.to_vector()
    assert vec.shape == (6,)  # 2 points × 3 coords
    assert jnp.allclose(vec, jnp.array([1, 2, 3, 4, 5, 6]))

    # Modify and restore
    new_vec = jnp.array([10, 20, 30, 40, 50, 60])
    cs.from_vector(new_vec)

    assert jnp.allclose(p1.value, jnp.array([10, 20, 30]))
    assert jnp.allclose(p2.value, jnp.array([40, 50, 60]))
    assert d1.value == 7.0  # Fixed param unchanged


def test_compile_parametric_simple():
    """Test parametric compilation of a simple translated sphere."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Create translated sphere
    offset = jnp.array([2.0, 0.0, 0.0])
    translated = sphere.translate(offset)

    # Compile to parametric form
    eval_fn = compile_parametric(translated, cs)

    # Should have extracted translation as a constraint
    assert len(cs.get_free_params()) > 0
    print(f"\n{cs.summary()}")

    # Evaluate
    params = cs.to_vector()
    query_point = jnp.array([3.0, 0.0, 0.0])
    result = eval_fn(params, query_point)

    # Should be on surface (distance ≈ 0)
    assert jnp.abs(result) < 0.1
    print(f"SDF value at {query_point}: {result}")


def test_parametric_differentiation():
    """Test differentiation with respect to free parameters."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)
    translated = sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Compile
    eval_fn = compile_parametric(translated, cs)

    # Define loss: want sphere surface to pass through target point
    target_point = jnp.array([2.0, 0.0, 0.0])

    def loss(params):
        sdf_value = eval_fn(params, target_point)
        return sdf_value ** 2  # Minimize squared distance

    # Compute gradient
    params = cs.to_vector()
    grad_fn = jax.grad(loss)
    gradient = grad_fn(params)

    print(f"\nInitial params: {params}")
    print(f"Gradient: {gradient}")

    # Gradient should be non-zero (can optimize offset)
    assert jnp.any(jnp.abs(gradient) > 1e-6)


def test_optimize_sphere_position():
    """Test optimizing sphere position to pass through target points."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Start with sphere at origin
    translated = sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Target: sphere surface should pass through these points
    # (points at distance 1.0 from center [2, 0, 0])
    target_points = jnp.array([
        [3.0, 0.0, 0.0],  # +X
        [2.0, 1.0, 0.0],  # +Y
        [2.0, 0.0, 1.0],  # +Z
        [1.0, 0.0, 0.0],  # -X
    ])
    target_values = jnp.zeros(4)  # Want distance = 0 (on surface)

    print("\nOptimizing sphere position...")
    optimized_params, loss_history = optimize_parameters(
        translated,
        target_points,
        target_values,
        cs,
        num_iterations=50,
        learning_rate=0.1
    )

    print(f"\nOptimized parameters: {optimized_params}")
    print(f"Final loss: {loss_history[-1]:.6f}")

    # Check that optimization improved
    assert loss_history[-1] < loss_history[0]

    # Optimized offset should be close to [2, 0, 0]
    free_params = cs.get_free_params()
    if free_params:
        optimized_offset = free_params[0].value
        print(f"Optimized offset: {optimized_offset}")
        # Should be approximately [2, 0, 0]
        assert jnp.allclose(optimized_offset[0], 2.0, atol=0.2)


def test_mixed_free_fixed():
    """Test optimization with mixed free and fixed parameters."""
    cs = ConstraintSystem()
    sphere = Sphere(radius=1.0)

    # Translate with initial offset
    translated = sphere.translate(jnp.array([1.0, 0.0, 0.0]))

    # Compile - this extracts offset as free parameter
    eval_fn = compile_parametric(translated, cs)

    # Now mark offset as fixed
    free_params = cs.get_free_params()
    if free_params:
        free_params[0].free = False

    print(f"\n{cs.summary()}")

    # Should have no free parameters now
    assert len(cs.get_free_params()) == 0
    assert len(cs.get_fixed_params()) > 0
