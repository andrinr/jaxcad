"""Tests for transformation operations."""

import jax.numpy as jnp
import pytest

from jaxcad.primitives import Box, Sphere
from jaxcad.transforms import Rotate, Scale, Translate


def test_translate():
    """Test translation transformation."""
    sphere = Sphere(radius=1.0)
    offset = jnp.array([1.0, 0.0, 0.0])
    translated = Translate(sphere, offset)

    # Point at (1, 0, 0) should be at center of translated sphere
    assert translated(jnp.array([1.0, 0.0, 0.0])) == pytest.approx(-1.0, abs=1e-5)

    # Point at (2, 0, 0) should be on surface
    assert translated(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-5)


def test_translate_method():
    """Test translation via method API."""
    sphere = Sphere(radius=1.0)
    translated = sphere.translate(jnp.array([1.0, 0.0, 0.0]))

    assert translated(jnp.array([1.0, 0.0, 0.0])) == pytest.approx(-1.0, abs=1e-5)


def test_uniform_scale():
    """Test uniform scaling."""
    sphere = Sphere(radius=1.0)
    scaled = Scale(sphere, 2.0)

    # Sphere with radius 1 scaled by 2 should have radius 2
    assert scaled(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-5)
    assert scaled(jnp.array([0.0, 0.0, 0.0])) == pytest.approx(-2.0, abs=1e-5)


def test_scale_method():
    """Test scaling via method API."""
    sphere = Sphere(radius=1.0)
    scaled = sphere.scale(2.0)

    assert scaled(jnp.array([2.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-5)


def test_rotate_z():
    """Test rotation around Z axis."""
    box = Box(size=jnp.array([2.0, 1.0, 1.0]))
    rotated = Rotate(box, 'z', jnp.pi / 2)

    # Box has half-extents [2.0, 1.0, 1.0], so it extends ±2 in X, ±1 in Y/Z
    # After 90° rotation around Z, the X axis becomes Y axis
    # Point at (0, 2.0, 0) should be on surface (at +Y boundary)
    assert rotated(jnp.array([0.0, 2.0, 0.0])) == pytest.approx(0.0, abs=1e-4)


def test_rotate_method():
    """Test rotation via method API."""
    sphere = Sphere(radius=1.0)
    rotated = sphere.rotate('z', jnp.pi / 2)

    # Rotating a sphere shouldn't change its shape
    assert rotated(jnp.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-5)


def test_chained_transforms():
    """Test chaining multiple transformations."""
    sphere = Sphere(radius=1.0)

    # Chain: scale, rotate, translate
    transformed = (
        sphere
        .scale(2.0)
        .rotate('z', jnp.pi / 4)
        .translate(jnp.array([1.0, 0.0, 0.0]))
    )

    # Should be callable
    result = transformed(jnp.array([1.0, 0.0, 0.0]))
    assert jnp.isfinite(result)
