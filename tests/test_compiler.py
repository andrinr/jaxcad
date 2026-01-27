"""Tests for the compiler module - parameter extraction and compilation."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.compiler import extract_parameters, compile_to_function
from jaxcad.parameters import Scalar, Vector
from jaxcad.primitives import Sphere
from jaxcad.transforms import Translate, Scale
from jaxcad.boolean import Union


def test_extract_free_fixed_parameters():
    """Test extracting free vs fixed parameters."""
    radius = Scalar(value=1.0, free=True, name='radius')
    offset = Vector(value=[1.0, 0.0, 0.0], free=False, name='offset')

    sphere = Sphere(radius=radius)
    translated = Translate(sphere, offset=offset)

    free_params, fixed_params = extract_parameters(translated)

    assert 'sphere_1.radius' in free_params
    assert 'translate_0.offset' in fixed_params
    assert len(free_params) == 1
    assert len(fixed_params) == 1


def test_compile_to_function_basic():
    """Test basic compilation to pure function."""
    radius = Scalar(value=1.0, free=True, name='radius')
    sphere = Sphere(radius=radius)

    free_params, fixed_params = extract_parameters(sphere)
    sdf_fn = compile_to_function(sphere)

    # Evaluate at a point
    point = jnp.array([2.0, 0.0, 0.0])
    distance = sdf_fn(point, free_params, fixed_params)

    # Distance from [2,0,0] to sphere with radius 1 should be 1.0
    assert jnp.abs(distance - 1.0) < 0.01


def test_compile_to_function_jittable():
    """Test that compiled function is JIT-compatible."""
    radius = Scalar(value=1.5, free=True, name='radius')
    offset = Vector(value=[1.0, 0.0, 0.0], free=True, name='offset')

    sphere = Sphere(radius=radius)
    translated = Translate(sphere, offset=offset)

    free_params, fixed_params = extract_parameters(translated)
    sdf_fn = compile_to_function(translated)

    # JIT compile the function
    @jax.jit
    def jitted_eval(point, free_p, fixed_p):
        return sdf_fn(point, free_p, fixed_p)

    # Test point
    point = jnp.array([2.0, 0.0, 0.0])

    # Should work without errors
    result = jitted_eval(point, free_params, fixed_params)

    assert jnp.isfinite(result)
    # Distance from [2,0,0] to sphere at [1,0,0] with radius 1.5 should be 1-1.5 = -0.5
    assert jnp.abs(result - (-0.5)) < 0.01


def test_gradient_through_compiled_function():
    """Test that gradients work through compiled function."""
    radius = Scalar(value=1.0, free=True, name='radius')
    sphere = Sphere(radius=radius)

    free_params, fixed_params = extract_parameters(sphere)
    sdf_fn = compile_to_function(sphere)

    # Define loss function
    def loss_fn(free_p):
        point = jnp.array([2.0, 0.0, 0.0])
        distance = sdf_fn(point, free_p, fixed_params)
        return distance ** 2

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(free_params)

    # Should have gradient for the radius parameter
    assert 'sphere_0.radius' in gradient
    assert jnp.isfinite(gradient['sphere_0.radius'].value)


def test_jit_with_gradient():
    """Test JIT compilation combined with gradient computation."""
    radius = Scalar(value=1.0, free=True, name='radius')
    offset = Vector(value=[0.0, 0.0, 0.0], free=True, name='offset')

    sphere = Sphere(radius=radius)
    translated = Translate(sphere, offset=offset)

    free_params, fixed_params = extract_parameters(translated)
    sdf_fn = compile_to_function(translated)

    # Target point
    target = jnp.array([2.5, 0.0, 0.0])

    # Loss function with JIT
    @jax.jit
    def loss_fn(free_p):
        distance = sdf_fn(target, free_p, fixed_params)
        return distance ** 2

    # Gradient function with JIT
    grad_fn = jax.jit(jax.grad(loss_fn))

    # Should compute without errors
    loss = loss_fn(free_params)
    grad = grad_fn(free_params)

    assert jnp.isfinite(loss)
    assert 'sphere_1.radius' in grad
    assert 'translate_0.offset' in grad
    assert jnp.isfinite(grad['sphere_1.radius'].value)
    assert jnp.all(jnp.isfinite(grad['translate_0.offset'].xyz))


def test_vmap_with_compiled_function():
    """Test that compiled function works with vmap."""
    radius = Scalar(value=1.0, free=False, name='radius')
    sphere = Sphere(radius=radius)

    free_params, fixed_params = extract_parameters(sphere)
    sdf_fn = compile_to_function(sphere)

    # Batch of points
    points = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])

    # Vectorized evaluation
    distances = jax.vmap(lambda p: sdf_fn(p, free_params, fixed_params))(points)

    assert distances.shape == (4,)
    assert jnp.all(jnp.isfinite(distances))

    # Check expected values
    # Center: -1.0, surface: 0.0, distances: 1.0, 2.0
    expected = jnp.array([-1.0, 0.0, 1.0, 2.0])
    assert jnp.allclose(distances, expected, atol=0.01)


def test_complex_expression():
    """Test compilation of complex SDF expression."""
    r1 = Scalar(value=1.0, free=True, name='r1')
    r2 = Scalar(value=0.8, free=True, name='r2')
    offset1 = Vector(value=[1.0, 0.0, 0.0], free=False)
    offset2 = Vector(value=[-1.0, 0.0, 0.0], free=False)
    scale_vec = Vector(value=[1.0, 1.0, 2.0], free=False)

    sphere1 = Sphere(radius=r1)
    sphere2 = Sphere(radius=r2)

    translated1 = Translate(sphere1, offset=offset1)
    translated2 = Translate(sphere2, offset=offset2)

    union = Union(translated1, translated2)
    scaled = Scale(union, scale=scale_vec)

    # Extract and compile
    free_params, fixed_params = extract_parameters(scaled)
    sdf_fn = compile_to_function(scaled)

    # Should have 2 free parameters (both radii)
    assert len(free_params) == 2

    # JIT compile
    @jax.jit
    def eval_fn(point, free_p):
        return sdf_fn(point, free_p, fixed_params)

    point = jnp.array([0.0, 0.0, 0.0])
    result = eval_fn(point, free_params)

    assert jnp.isfinite(result)


def test_optimization_convergence():
    """Test that optimization actually converges using compiled function."""
    radius = Scalar(value=0.5, free=True, name='radius')
    sphere = Sphere(radius=radius)

    free_params, fixed_params = extract_parameters(sphere)
    sdf_fn = compile_to_function(sphere)

    target_point = jnp.array([2.0, 0.0, 0.0])

    # Loss function
    @jax.jit
    def loss_fn(free_p):
        distance = sdf_fn(target_point, free_p, fixed_params)
        return distance ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))

    # Optimize
    learning_rate = 0.1
    for _ in range(30):
        grad = grad_fn(free_params)
        # Update radius
        new_radius = free_params['sphere_0.radius'].value - learning_rate * grad['sphere_0.radius'].value
        free_params['sphere_0.radius'] = Scalar(value=new_radius, free=True, name='radius')

    # Should converge to radius = 2.0
    final_radius = free_params['sphere_0.radius'].value
    assert jnp.abs(final_radius - 2.0) < 0.1


def test_fixed_parameters_dont_change():
    """Test that fixed parameters remain constant during optimization."""
    radius = Scalar(value=1.0, free=False, name='radius')  # Fixed
    offset = Vector(value=[0.0, 0.0, 0.0], free=True, name='offset')  # Free

    sphere = Sphere(radius=radius)
    translated = Translate(sphere, offset=offset)

    free_params, fixed_params = extract_parameters(translated)
    sdf_fn = compile_to_function(translated)

    # Check initial state
    assert 'sphere_1.radius' in fixed_params
    assert 'translate_0.offset' in free_params
    initial_radius = fixed_params['sphere_1.radius'].value

    # Optimize (only offset should change)
    target = jnp.array([2.0, 0.0, 0.0])

    @jax.jit
    def loss_fn(free_p):
        return sdf_fn(target, free_p, fixed_params) ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))

    for _ in range(20):
        grad = grad_fn(free_params)
        new_offset = free_params['translate_0.offset'].xyz - 0.1 * grad['translate_0.offset'].xyz
        free_params['translate_0.offset'] = Vector(value=new_offset, free=True, name='offset')

    # Radius should be unchanged
    assert fixed_params['sphere_1.radius'].value == initial_radius


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
