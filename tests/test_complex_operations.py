"""Tests for complex geometry operations."""

import jax
import jax.numpy as jnp

from jaxcad.boolean import merge, smooth_vertices, subdivide_faces
from jaxcad.modifications import taper, twist
from jaxcad.operations import array_circular, array_linear, extrude, loft, revolve, sweep
from jaxcad.primitives import box
from jaxcad.sketch import circle, polygon, rectangle, regular_polygon


class TestSketch:
    """Tests for 2D sketch operations."""

    def test_rectangle(self):
        """Test rectangle creation."""
        profile = rectangle(jnp.zeros(2), 2.0, 1.0)
        assert profile.points.shape == (4, 2)
        assert profile.closed

    def test_circle(self):
        """Test circle creation."""
        profile = circle(jnp.zeros(2), 1.0, resolution=16)
        assert profile.points.shape == (16, 2)
        assert profile.closed

    def test_regular_polygon(self):
        """Test regular polygon creation."""
        profile = regular_polygon(jnp.zeros(2), 1.0, n_sides=6)
        assert profile.points.shape == (6, 2)


class TestExtrude:
    """Tests for extrude operation."""

    def test_extrude_rectangle(self):
        """Test extruding a rectangle."""
        profile = rectangle(jnp.zeros(2), 2.0, 1.0)
        solid = extrude(profile, height=3.0)

        assert solid.vertices.shape[0] == 8  # 4 bottom + 4 top
        assert solid.vertices.shape[1] == 3
        assert solid.faces.shape[1] == 3

    def test_extrude_circle(self):
        """Test extruding a circle."""
        profile = circle(jnp.zeros(2), 1.0, resolution=16)
        solid = extrude(profile, height=2.0)

        assert solid.vertices.shape[0] == 32  # 16 bottom + 16 top
        assert solid.vertices.shape[1] == 3

    def test_extrude_gradient(self):
        """Test gradient through extrude operation."""

        def extruded_vertex_z(height):
            profile = rectangle(jnp.zeros(2), 1.0, 1.0)
            solid = extrude(profile, height)
            # Return z-coordinate of a top vertex
            return solid.vertices[4, 2]

        height = 3.0
        grad = jax.grad(extruded_vertex_z)(height)

        # Gradient should be 1 (z increases linearly with height)
        assert jnp.allclose(grad, 1.0, atol=1e-5)


class TestRevolve:
    """Tests for revolve operation."""

    def test_revolve_profile(self):
        """Test revolving a profile."""
        profile_points = jnp.array([[0.5, 0.0], [1.0, 1.0], [0.5, 2.0]])
        profile = polygon(profile_points, closed=False)
        solid = revolve(profile, resolution=16)

        assert solid.vertices.shape[0] == 3 * 16  # 3 profile points * 16 revolution steps
        assert solid.vertices.shape[1] == 3

    def test_revolve_gradient(self):
        """Test gradient through revolve operation."""

        def revolved_volume_proxy(radius):
            profile_points = jnp.array([[radius, 0.0], [radius, 1.0]])
            profile = polygon(profile_points, closed=False)
            solid = revolve(profile, resolution=8)

            # Sum of squared distances from origin
            return jnp.sum(solid.vertices**2)

        radius = 1.5
        grad = jax.grad(revolved_volume_proxy)(radius)

        assert jnp.isfinite(grad), "Gradient should be finite"
        assert grad != 0, "Gradient should be non-zero"


class TestLoft:
    """Tests for loft operation."""

    def test_loft_two_profiles(self):
        """Test lofting between two profiles."""
        profile1 = circle(jnp.zeros(2), 1.0, resolution=8)
        profile2 = circle(jnp.zeros(2), 0.5, resolution=8)
        heights = jnp.array([0.0, 2.0])

        solid = loft([profile1, profile2], heights)

        assert solid.vertices.shape[0] == 16  # 8 points * 2 profiles
        assert solid.vertices.shape[1] == 3

    def test_loft_gradient(self):
        """Test gradient through loft operation."""

        def lofted_vertex_position(top_radius):
            profile1 = circle(jnp.zeros(2), 1.0, resolution=8)
            profile2 = circle(jnp.zeros(2), top_radius, resolution=8)
            heights = jnp.array([0.0, 2.0])
            solid = loft([profile1, profile2], heights)
            return jnp.sum(solid.vertices[8:])  # Sum of top profile vertices

        radius = 0.5
        grad = jax.grad(lofted_vertex_position)(radius)

        assert jnp.isfinite(grad), "Gradient should be finite"


class TestSweep:
    """Tests for sweep operation."""

    def test_sweep_along_path(self):
        """Test sweeping along a path."""
        profile = circle(jnp.zeros(2), 0.3, resolution=4)
        path = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        solid = sweep(profile, path)

        assert solid.vertices.shape[0] == 4 * 3  # 4 profile points * 3 path points
        assert solid.vertices.shape[1] == 3


class TestArrays:
    """Tests for array operations."""

    def test_linear_array(self):
        """Test linear array."""
        base = box(jnp.zeros(3), jnp.ones(3))
        arrayed = array_linear(base, jnp.array([1.0, 0.0, 0.0]), count=3, spacing=2.0)

        assert arrayed.vertices.shape[0] == 24  # 8 vertices * 3 copies
        assert arrayed.faces.shape[0] == 36  # 12 faces * 3 copies

    def test_circular_array(self):
        """Test circular array."""
        base = box(jnp.array([2.0, 0.0, 0.0]), jnp.ones(3))
        arrayed = array_circular(base, jnp.array([0.0, 0.0, 1.0]), jnp.zeros(3), count=4)

        assert arrayed.vertices.shape[0] == 32  # 8 vertices * 4 copies
        assert arrayed.faces.shape[0] == 48  # 12 faces * 4 copies

    def test_array_gradient(self):
        """Test gradient through array operation."""

        def arrayed_total_volume(spacing):
            base = box(jnp.zeros(3), jnp.ones(3))
            arrayed = array_linear(base, jnp.array([1.0, 0.0, 0.0]), count=3, spacing=spacing)

            # Compute bounding box volume
            min_coords = jnp.min(arrayed.vertices, axis=0)
            max_coords = jnp.max(arrayed.vertices, axis=0)
            return jnp.prod(max_coords - min_coords)

        spacing = 2.0
        grad = jax.grad(arrayed_total_volume)(spacing)

        assert jnp.isfinite(grad), "Gradient should be finite"


class TestModifications:
    """Tests for modification operations."""

    def test_twist(self):
        """Test twist operation."""
        solid = box(jnp.zeros(3), jnp.array([1.0, 1.0, 2.0]))
        twisted = twist(solid, jnp.array([0.0, 0.0, 1.0]), jnp.pi / 4)

        assert twisted.vertices.shape == solid.vertices.shape
        # Check that vertices have moved
        assert not jnp.allclose(twisted.vertices, solid.vertices)

    def test_taper(self):
        """Test taper operation."""
        solid = box(jnp.zeros(3), jnp.array([2.0, 2.0, 2.0]))
        tapered = taper(solid, jnp.array([0.0, 0.0, 1.0]), scale_top=0.5)

        assert tapered.vertices.shape == solid.vertices.shape

    def test_twist_gradient(self):
        """Test gradient through twist operation."""

        def twisted_vertex_position(angle):
            solid = box(jnp.zeros(3), jnp.ones(3))
            twisted = twist(solid, jnp.array([0.0, 0.0, 1.0]), angle)
            return jnp.sum(twisted.vertices[0])

        angle = jnp.pi / 4
        grad = jax.grad(twisted_vertex_position)(angle)

        assert jnp.isfinite(grad), "Gradient should be finite"

    def test_taper_gradient(self):
        """Test gradient through taper operation."""

        def tapered_vertex_x(scale_top):
            solid = box(jnp.zeros(3), jnp.array([2.0, 2.0, 2.0]))
            tapered = taper(solid, jnp.array([0.0, 0.0, 1.0]), scale_top)
            # Return x-coordinate of a top vertex
            return tapered.vertices[4, 0]

        scale_top = 0.5
        grad = jax.grad(tapered_vertex_x)(scale_top)

        assert jnp.isfinite(grad), "Gradient should be finite"


class TestBoolean:
    """Tests for boolean operations."""

    def test_merge(self):
        """Test merging two solids."""
        s1 = box(jnp.array([0.0, 0.0, 0.0]), jnp.ones(3))
        s2 = box(jnp.array([2.0, 0.0, 0.0]), jnp.ones(3))
        merged = merge(s1, s2)

        assert merged.vertices.shape[0] == 16  # 8 + 8
        assert merged.faces.shape[0] == 24  # 12 + 12

    def test_smooth_vertices(self):
        """Test vertex smoothing."""
        solid = box(jnp.zeros(3), jnp.ones(3))
        smoothed = smooth_vertices(solid, iterations=1, factor=0.1)

        assert smoothed.vertices.shape == solid.vertices.shape

    def test_subdivide_faces(self):
        """Test face subdivision."""
        solid = box(jnp.zeros(3), jnp.ones(3))
        subdivided = subdivide_faces(solid)

        # Each face becomes 4 faces
        assert subdivided.faces.shape[0] == 48  # 12 * 4


class TestComplexPipeline:
    """Tests for complex operation pipelines."""

    def test_extrude_twist_gradient(self):
        """Test gradient through extrude + twist pipeline."""

        def pipeline(params):
            width, height, angle = params
            profile = rectangle(jnp.zeros(2), width, width)
            solid = extrude(profile, height)
            twisted = twist(solid, jnp.array([0.0, 0.0, 1.0]), angle)
            return jnp.sum(twisted.vertices**2)

        params = jnp.array([1.0, 2.0, jnp.pi / 4])
        grad = jax.grad(pipeline)(params)

        assert jnp.all(jnp.isfinite(grad)), "All gradients should be finite"
        # Width and height gradients should be non-zero, angle might be near zero
        assert grad[0] != 0 and grad[1] != 0, "Width and height gradients should be non-zero"

    def test_revolve_taper_gradient(self):
        """Test gradient through revolve + taper pipeline."""

        def pipeline(params):
            radius, scale = params
            profile_points = jnp.array([[radius, 0.0], [radius, 2.0]])
            profile = polygon(profile_points, closed=False)
            solid = revolve(profile, resolution=8)
            tapered = taper(solid, jnp.array([0.0, 0.0, 1.0]), scale)
            return jnp.sum(tapered.vertices[0])

        params = jnp.array([1.0, 0.5])
        grad = jax.grad(pipeline)(params)

        assert jnp.all(jnp.isfinite(grad)), "All gradients should be finite"
