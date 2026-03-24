"""Tests for SDF compilation to pure JAX functions."""

import pytest
import jax.numpy as jnp

from jaxcad.geometry.parameters import Scalar, Vector
from jaxcad.sdf.primitives.sphere import Sphere
from jaxcad.sdf.primitives.box import Box
from jaxcad.compiler import functionalize, extract_parameters


def test_compile_sphere_basic():
    """Test compiling a sphere to a function."""
    radius = Scalar(1.0, free=True, name='radius')
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Query at origin (inside sphere)
    point = jnp.array([0., 0., 0.])
    free_vals = {'sphere_0.radius': 1.0}
    fixed_vals = {}

    distance = sdf_fn(point, free_vals, fixed_vals)

    # At origin, distance should be -radius
    assert jnp.isclose(distance, -1.0)


def test_compile_sphere_outside():
    """Test compiled sphere SDF for point outside."""
    radius = Scalar(1.0, free=False, name='radius')
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Query at distance 2 from origin
    point = jnp.array([2., 0., 0.])
    free_vals = {}
    fixed_vals = {'sphere_0.radius': 1.0}

    distance = sdf_fn(point, free_vals, fixed_vals)

    # Distance should be 2 - 1 = 1
    assert jnp.isclose(distance, 1.0)


def test_compile_with_parameter_variation():
    """Test that compiled function responds to parameter changes."""
    radius = Scalar(1.0, free=True, name='radius')
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)
    point = jnp.array([2., 0., 0.])

    # Test with radius = 1.0
    dist1 = sdf_fn(point, {'sphere_0.radius': 1.0}, {})
    assert jnp.isclose(dist1, 1.0)

    # Test with radius = 1.5
    dist2 = sdf_fn(point, {'sphere_0.radius': 1.5}, {})
    assert jnp.isclose(dist2, 0.5)

    # Test with radius = 2.0
    dist3 = sdf_fn(point, {'sphere_0.radius': 2.0}, {})
    assert jnp.isclose(dist3, 0.0)


def test_compile_box():
    """Test compiling a box to a function."""
    size = Vector([1., 1., 1.], free=True, name='size')
    box = Box(size=size)

    sdf_fn = functionalize(box)

    # Query at origin (inside box)
    point = jnp.array([0., 0., 0.])
    free_vals = {'box_0.size': jnp.array([1., 1., 1.])}
    fixed_vals = {}

    distance = sdf_fn(point, free_vals, fixed_vals)

    # Should be inside the box
    assert distance < 0


def test_compile_consistency():
    """Test that compiled function gives same results as direct evaluation."""
    radius = Scalar(1.5, free=False, name='radius')
    sphere = Sphere(radius=radius)

    sdf_fn = functionalize(sphere)

    # Test multiple points
    test_points = [
        jnp.array([0., 0., 0.]),
        jnp.array([1., 0., 0.]),
        jnp.array([2., 0., 0.]),
        jnp.array([1., 1., 0.]),
    ]

    for point in test_points:
        # Direct evaluation using __call__
        direct_dist = sphere(point)

        # Compiled evaluation
        compiled_dist = sdf_fn(point, {}, {'sphere_0.radius': 1.5})

        # Should be very close
        assert jnp.isclose(direct_dist, compiled_dist, atol=1e-6)
