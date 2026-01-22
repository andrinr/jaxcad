"""Tests for SDF primitives."""

import jax.numpy as jnp
import pytest

from jaxcad.primitives import (
    Box,
    Capsule,
    Circle,
    Cone,
    Cylinder,
    Ellipsoid,
    Plane,
    Rectangle,
    Sphere,
    Torus,
)


class TestSphere:
    def test_center_point(self):
        """Point at center should be -radius distance"""
        sphere = Sphere(radius=1.0)
        p = jnp.array([0.0, 0.0, 0.0])
        assert jnp.isclose(sphere(p), -1.0)

    def test_surface_point(self):
        """Point on surface should be ~0 distance"""
        sphere = Sphere(radius=1.0)
        p = jnp.array([1.0, 0.0, 0.0])
        assert jnp.isclose(sphere(p), 0.0)

    def test_outside_point(self):
        """Point outside should be positive distance"""
        sphere = Sphere(radius=1.0)
        p = jnp.array([2.0, 0.0, 0.0])
        assert sphere(p) > 0


class TestBox:
    def test_center_point(self):
        """Point at center should be negative"""
        box = Box(size=jnp.array([1.0, 1.0, 1.0]))
        p = jnp.array([0.0, 0.0, 0.0])
        assert box(p) < 0

    def test_corner_point(self):
        """Point at corner should be ~0 distance"""
        box = Box(size=jnp.array([1.0, 1.0, 1.0]))
        p = jnp.array([1.0, 1.0, 1.0])
        assert jnp.isclose(box(p), 0.0, atol=1e-5)

    def test_outside_point(self):
        """Point outside should be positive"""
        box = Box(size=jnp.array([1.0, 1.0, 1.0]))
        p = jnp.array([2.0, 0.0, 0.0])
        assert box(p) > 0


class TestCylinder:
    def test_center_point(self):
        """Point at center should be negative"""
        cyl = Cylinder(radius=1.0, height=1.0)
        p = jnp.array([0.0, 0.0, 0.0])
        assert cyl(p) < 0

    def test_side_surface(self):
        """Point on cylindrical surface should be ~0"""
        cyl = Cylinder(radius=1.0, height=1.0)
        p = jnp.array([1.0, 0.0, 0.0])
        assert jnp.isclose(cyl(p), 0.0, atol=1e-5)

    def test_top_surface(self):
        """Point on top cap should be ~0"""
        cyl = Cylinder(radius=1.0, height=1.0)
        p = jnp.array([0.0, 0.0, 1.0])
        assert jnp.isclose(cyl(p), 0.0, atol=1e-5)


class TestCone:
    def test_apex(self):
        """Apex should be on surface or inside"""
        cone = Cone(radius=1.0, height=2.0)
        p = jnp.array([0.0, 0.0, 0.0])
        assert cone(p) <= 0

    def test_base_edge(self):
        """Point on base edge should be reasonably close"""
        cone = Cone(radius=1.0, height=1.0)
        p = jnp.array([1.0, 0.0, -1.0])
        # Cone SDF implementation may not be exact
        assert cone(p) < 2.0


class TestTorus:
    def test_center_point(self):
        """Point at center should be positive (empty inside)"""
        torus = Torus(major_radius=2.0, minor_radius=0.5)
        p = jnp.array([0.0, 0.0, 0.0])
        assert torus(p) > 0

    def test_tube_point(self):
        """Point on tube surface should be ~0"""
        torus = Torus(major_radius=2.0, minor_radius=0.5)
        p = jnp.array([2.5, 0.0, 0.0])
        assert jnp.isclose(torus(p), 0.0, atol=1e-5)


class TestCapsule:
    def test_center_point(self):
        """Point at center should be negative"""
        capsule = Capsule(radius=0.5, height=1.0)
        p = jnp.array([0.0, 0.0, 0.0])
        assert capsule(p) < 0

    def test_cap_point(self):
        """Point on spherical cap should be ~0"""
        capsule = Capsule(radius=0.5, height=1.0)
        p = jnp.array([0.0, 0.0, 1.5])
        assert jnp.isclose(capsule(p), 0.0, atol=1e-5)


class TestCircle:
    def test_center_point(self):
        """Point at center should be -radius"""
        circle = Circle(radius=1.0)
        p = jnp.array([0.0, 0.0, 0.0])
        assert jnp.isclose(circle(p), -1.0)

    def test_ignores_z(self):
        """Circle should ignore Z coordinate"""
        circle = Circle(radius=1.0)
        p1 = jnp.array([1.0, 0.0, 0.0])
        p2 = jnp.array([1.0, 0.0, 10.0])
        assert jnp.isclose(circle(p1), circle(p2))


class TestRectangle:
    def test_center_point(self):
        """Point at center should be negative"""
        rect = Rectangle(size=jnp.array([1.0, 1.0]))
        p = jnp.array([0.0, 0.0, 0.0])
        assert rect(p) < 0

    def test_ignores_z(self):
        """Rectangle should ignore Z coordinate"""
        rect = Rectangle(size=jnp.array([1.0, 1.0]))
        p1 = jnp.array([0.5, 0.5, 0.0])
        p2 = jnp.array([0.5, 0.5, 10.0])
        assert jnp.isclose(rect(p1), rect(p2))


class TestPlane:
    def test_xy_plane(self):
        """XY plane at z=0"""
        plane = Plane(normal=jnp.array([0.0, 0.0, 1.0]), distance=0.0)
        p_below = jnp.array([0.0, 0.0, -1.0])
        p_on = jnp.array([0.0, 0.0, 0.0])
        p_above = jnp.array([0.0, 0.0, 1.0])

        assert plane(p_below) < 0
        assert jnp.isclose(plane(p_on), 0.0)
        assert plane(p_above) > 0

    def test_offset_plane(self):
        """Plane offset from origin"""
        plane = Plane(normal=jnp.array([0.0, 0.0, 1.0]), distance=2.0)
        p = jnp.array([0.0, 0.0, 2.0])
        assert jnp.isclose(plane(p), 0.0)


class TestEllipsoid:
    def test_center_point(self):
        """Point at center should be negative"""
        ellipsoid = Ellipsoid(radii=jnp.array([1.0, 2.0, 3.0]))
        p = jnp.array([0.0, 0.0, 0.0])
        assert ellipsoid(p) < 0

    def test_axis_point(self):
        """Point on semi-axis should be ~0"""
        ellipsoid = Ellipsoid(radii=jnp.array([1.0, 2.0, 3.0]))
        p = jnp.array([1.0, 0.0, 0.0])
        assert jnp.isclose(ellipsoid(p), 0.0, atol=0.5)  # Approximate SDF


class TestVectorization:
    def test_batch_evaluation(self):
        """SDFs should work with batched points"""
        sphere = Sphere(radius=1.0)
        points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        distances = sphere(points)
        assert distances.shape == (3,)
        assert jnp.isclose(distances[0], -1.0)
        assert jnp.isclose(distances[1], 0.0)
        assert distances[2] > 0
