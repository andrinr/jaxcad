"""Tests for the new @parametric decorator API with PyTrees."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.parametric import parametric
from jaxcad.primitives import Sphere, Box


def test_basic_decorator():
    """Test basic @parametric decorator usage."""

    @parametric
    def simple_sphere():
        return Sphere(radius=1.0)

    # Get initial params
    params = simple_sphere.init_params()

    # Should have sphere_0 with radius
    assert 'sphere_0' in params
    assert 'radius' in params['sphere_0']
    assert float(params['sphere_0']['radius']) == 1.0

    # Evaluate at a point
    point = jnp.array([2.0, 0.0, 0.0])
    value = simple_sphere(params, point)

    # Distance from [2,0,0] to sphere centered at origin with radius 1 should be 1.0
    assert jnp.abs(value - 1.0) < 0.01


def test_decorator_with_transform():
    """Test decorator with transform operations."""

    @parametric
    def translated_sphere():
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([2.0, 0.0, 0.0]))

    params = translated_sphere.init_params()

    # Should have both sphere and translate params
    assert 'sphere_0' in params
    assert 'translate_1' in params
    assert 'offset' in params['translate_1']

    # Check offset value
    offset = params['translate_1']['offset']
    assert jnp.allclose(offset, jnp.array([2.0, 0.0, 0.0]))

    # Evaluate at sphere center (should be at [2,0,0])
    point = jnp.array([2.0, 0.0, 0.0])
    value = translated_sphere(params, point)

    # Should be close to surface (distance ~ -1.0 inside)
    assert jnp.abs(value - (-1.0)) < 0.01


def test_gradient_flow():
    """Test that gradients flow through parameters."""

    @parametric
    def my_sdf():
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([2.0, 0.0, 0.0]))

    params = my_sdf.init_params()

    # Define loss function at a point NOT on the surface
    def loss_fn(p):
        point = jnp.array([4.0, 0.0, 0.0])  # Point away from surface
        return my_sdf(p, point) ** 2

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(params)

    # Gradient should have same structure as params
    assert 'sphere_0' in gradient
    assert 'translate_1' in gradient
    assert 'radius' in gradient['sphere_0']
    assert 'offset' in gradient['translate_1']

    # Offset gradient should be non-zero (translate affects the distance)
    assert jnp.any(gradient['translate_1']['offset'] != 0)


def test_modify_params():
    """Test modifying parameters and re-evaluating."""

    @parametric
    def my_sdf():
        return Sphere(radius=1.0)

    params = my_sdf.init_params()
    point = jnp.array([2.0, 0.0, 0.0])

    # Evaluate with original params
    value1 = my_sdf(params, point)

    # Modify radius
    params['sphere_0']['radius'] = jnp.array(2.0)
    value2 = my_sdf(params, point)

    # Values should be different
    assert jnp.abs(value1 - value2) > 0.5


def test_complex_expression():
    """Test with complex SDF expression."""

    @parametric
    def complex_sdf():
        sphere = Sphere(radius=1.0)
        box = Box(size=jnp.array([1.5, 1.5, 1.5]))

        return (
            sphere.translate(jnp.array([1.0, 0.0, 0.0]))
            | box.rotate('z', jnp.pi/4)
        ).twist(0.5)

    params = complex_sdf.init_params()

    # Should have multiple parameter groups (exact keys may vary)
    assert 'sphere_0' in params
    # Check that we have box params (could be box_1, box_2, etc.)
    box_keys = [k for k in params.keys() if k.startswith('box_')]
    assert len(box_keys) > 0

    # Evaluate
    point = jnp.array([1.5, 0.0, 0.0])
    value = complex_sdf(params, point)

    # Should return a valid distance
    assert jnp.isfinite(value)


def test_optimization():
    """Test optimizing parameters to minimize loss."""

    @parametric
    def my_sdf():
        sphere = Sphere(radius=1.0)
        return sphere.translate(jnp.array([0.0, 0.0, 0.0]))

    # Target: sphere surface should pass through [2,0,0]
    target_point = jnp.array([2.0, 0.0, 0.0])

    def loss_fn(params):
        value = my_sdf(params, target_point)
        return value ** 2  # Want distance = 0

    # Optimize
    params = my_sdf.init_params()
    grad_fn = jax.jit(jax.grad(loss_fn))

    learning_rate = 0.1
    for _ in range(50):
        grad = grad_fn(params)
        # Update params (pytree math)
        params = jax.tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            params, grad
        )

    # Check final loss
    final_loss = loss_fn(params)
    assert final_loss < 0.01  # Should converge

    # With both radius and offset free, the optimizer can adjust either/both
    # to make the surface pass through [2,0,0]. Check that we found a solution.
    optimized_radius = params['sphere_0']['radius']
    optimized_offset = params['translate_1']['offset']

    # Distance from sphere center to target should equal radius
    center = optimized_offset
    distance_to_target = jnp.linalg.norm(target_point - center)
    assert jnp.allclose(distance_to_target, optimized_radius, atol=0.1)


def test_direct_sdf_compilation():
    """Test using decorator directly on SDF instance."""

    sphere = Sphere(radius=1.0)
    translated = sphere.translate(jnp.array([2.0, 0.0, 0.0]))

    # Apply parametric directly (not as decorator)
    child_sdf = parametric(translated)

    params = child_sdf.init_params()
    point = jnp.array([2.0, 0.0, 0.0])
    value = child_sdf(params, point)

    assert jnp.isfinite(value)


def test_multiple_transforms():
    """Test SDF with multiple sequential transforms."""

    @parametric
    def multi_transform():
        sphere = Sphere(radius=1.0)
        return (sphere
                .translate(jnp.array([1.0, 0.0, 0.0]))
                .rotate('z', jnp.pi/6)
                .scale(2.0))

    params = multi_transform.init_params()

    # Should have sphere, translate, rotate, scale params
    assert 'sphere_0' in params
    assert 'translate_1' in params
    assert 'rotate_2' in params
    assert 'scale_3' in params

    # Evaluate
    point = jnp.array([2.0, 0.0, 0.0])
    value = multi_transform(params, point)
    assert jnp.isfinite(value)

    # Test gradient
    def loss_fn(p):
        return multi_transform(p, point) ** 2

    grad = jax.grad(loss_fn)(params)

    # All parameter groups should have gradients
    assert 'sphere_0' in grad
    assert 'translate_1' in grad
    assert 'rotate_2' in grad
    assert 'scale_3' in grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
