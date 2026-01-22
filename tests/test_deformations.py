"""Tests for deformation transformations."""

import jax.numpy as jnp
import pytest

from jaxcad.primitives import Box, Cylinder, Sphere
from jaxcad.transforms import Twist, Bend, Taper, RepeatInfinite, Mirror


def test_twist():
    """Test twist transformation."""
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))
    twisted = Twist(box, axis='z', strength=1.0)

    # Should be callable
    result = twisted(jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isfinite(result)


def test_twist_method():
    """Test twist via method API."""
    cylinder = Cylinder(radius=1.0, height=2.0)
    twisted = cylinder.twist('z', strength=2.0)

    # Center should still be inside
    result = twisted(jnp.array([0.0, 0.0, 0.0]))
    assert result < 0


def test_bend():
    """Test bend transformation."""
    cylinder = Cylinder(radius=0.5, height=2.0)
    bent = Bend(cylinder, axis='z', strength=0.5)

    result = bent(jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isfinite(result)


def test_bend_method():
    """Test bend via method API."""
    box = Box(size=jnp.array([0.5, 0.5, 1.5]))
    bent = box.bend('z', strength=1.0)

    result = bent(jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isfinite(result)


def test_taper():
    """Test taper transformation."""
    cylinder = Cylinder(radius=1.0, height=2.0)
    tapered = Taper(cylinder, axis='z', strength=0.3)

    result = tapered(jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isfinite(result)


def test_taper_method():
    """Test taper via method API."""
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))
    tapered = box.taper('z', strength=0.5)

    result = tapered(jnp.array([0.0, 0.0, 0.0]))
    assert result < 0  # Center should be inside


def test_repeat_infinite():
    """Test infinite repetition."""
    sphere = Sphere(radius=0.3)
    repeated = RepeatInfinite(sphere, spacing=jnp.array([2.0, 2.0, 2.0]))

    # Points at grid positions should give same result
    result1 = repeated(jnp.array([0.0, 0.0, 0.0]))
    result2 = repeated(jnp.array([2.0, 0.0, 0.0]))
    result3 = repeated(jnp.array([0.0, 2.0, 0.0]))

    assert jnp.allclose(result1, result2, atol=1e-5)
    assert jnp.allclose(result1, result3, atol=1e-5)


def test_repeat_infinite_method():
    """Test infinite repetition via method API."""
    sphere = Sphere(radius=0.4)
    repeated = sphere.repeat_infinite(jnp.array([2.0, 2.0, 2.0]))

    result = repeated(jnp.array([0.0, 0.0, 0.0]))
    assert result < 0  # Center should be inside a sphere


def test_mirror():
    """Test mirror transformation."""
    sphere = Sphere(radius=1.0)
    # Translate then mirror
    translated = sphere.translate(jnp.array([2.0, 0.0, 0.0]))
    mirrored = Mirror(translated, axis='x', offset=0.0)

    # Points on both sides should give same distance
    result1 = mirrored(jnp.array([2.0, 0.0, 0.0]))
    result2 = mirrored(jnp.array([-2.0, 0.0, 0.0]))

    assert jnp.allclose(result1, result2, atol=1e-5)


def test_mirror_method():
    """Test mirror via method API."""
    box = Box(size=jnp.array([1.0, 0.5, 0.5]))
    translated = box.translate(jnp.array([1.0, 0.0, 0.0]))
    mirrored = translated.mirror('x')

    # Should be symmetric
    result1 = mirrored(jnp.array([1.0, 0.0, 0.0]))
    result2 = mirrored(jnp.array([-1.0, 0.0, 0.0]))

    assert jnp.allclose(result1, result2, atol=1e-5)


def test_complex_chain():
    """Test chaining complex transformations."""
    sphere = Sphere(radius=0.5)

    # Chain multiple deformations
    transformed = (
        sphere
        .twist('z', 1.0)
        .taper('z', 0.2)
        .translate(jnp.array([1.0, 0.0, 0.0]))
        .mirror('x')
    )

    result = transformed(jnp.array([0.0, 0.0, 0.0]))
    assert jnp.isfinite(result)
