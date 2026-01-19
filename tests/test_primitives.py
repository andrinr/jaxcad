"""Tests for primitive shape generators."""

import jax.numpy as jnp

from jaxcad.primitives import box, cylinder, sphere


class TestBox:
    """Tests for box primitive."""

    def test_box_creation(self):
        """Test that box creates correct number of vertices and faces."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([2.0, 2.0, 2.0])
        solid = box(center, size)

        assert solid.vertices.shape == (8, 3), "Box should have 8 vertices"
        assert solid.faces.shape == (12, 3), "Box should have 12 triangular faces"

    def test_box_center(self):
        """Test that box is centered correctly."""
        center = jnp.array([1.0, 2.0, 3.0])
        size = jnp.array([2.0, 2.0, 2.0])
        solid = box(center, size)

        computed_center = jnp.mean(solid.vertices, axis=0)
        assert jnp.allclose(computed_center, center), "Box center should match input"

    def test_box_size(self):
        """Test that box has correct dimensions."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([2.0, 4.0, 6.0])
        solid = box(center, size)

        min_coords = jnp.min(solid.vertices, axis=0)
        max_coords = jnp.max(solid.vertices, axis=0)
        computed_size = max_coords - min_coords

        assert jnp.allclose(computed_size, size), "Box size should match input"

    def test_box_vertices_valid(self):
        """Test that all box vertices are finite."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        assert jnp.all(jnp.isfinite(solid.vertices)), "All vertices should be finite"

    def test_box_face_indices_valid(self):
        """Test that face indices are within valid range."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        assert jnp.all(solid.faces >= 0), "Face indices should be non-negative"
        assert jnp.all(solid.faces < 8), "Face indices should be less than num vertices"


class TestSphere:
    """Tests for sphere primitive."""

    def test_sphere_creation(self):
        """Test that sphere creates vertices and faces."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 1.0
        resolution = 8
        solid = sphere(center, radius, resolution)

        expected_vertices = (resolution // 2 + 1) * resolution
        assert solid.vertices.shape[0] == expected_vertices
        assert solid.vertices.shape[1] == 3
        assert solid.faces.shape[1] == 3

    def test_sphere_center(self):
        """Test that sphere is centered correctly."""
        center = jnp.array([1.0, 2.0, 3.0])
        radius = 1.0
        solid = sphere(center, radius, resolution=16)

        computed_center = jnp.mean(solid.vertices, axis=0)
        assert jnp.allclose(computed_center, center, atol=0.1), (
            "Sphere center should be close to input"
        )

    def test_sphere_radius(self):
        """Test that sphere has correct radius."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 2.5
        solid = sphere(center, radius, resolution=16)

        # Check distances from center
        distances = jnp.linalg.norm(solid.vertices - center, axis=1)
        assert jnp.allclose(distances, radius, atol=1e-5), (
            "All vertices should be at radius distance"
        )

    def test_sphere_vertices_valid(self):
        """Test that all sphere vertices are finite."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 1.0
        solid = sphere(center, radius, resolution=8)

        assert jnp.all(jnp.isfinite(solid.vertices)), "All vertices should be finite"


class TestCylinder:
    """Tests for cylinder primitive."""

    def test_cylinder_creation(self):
        """Test that cylinder creates vertices and faces."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 1.0
        height = 2.0
        resolution = 8
        solid = cylinder(center, radius, height, resolution)

        # 2 circles + 2 center points
        expected_vertices = 2 * resolution + 2
        assert solid.vertices.shape[0] == expected_vertices
        assert solid.vertices.shape[1] == 3
        assert solid.faces.shape[1] == 3

    def test_cylinder_height(self):
        """Test that cylinder has correct height."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 1.0
        height = 3.0
        solid = cylinder(center, radius, height, resolution=16)

        min_z = jnp.min(solid.vertices[:, 2])
        max_z = jnp.max(solid.vertices[:, 2])
        computed_height = max_z - min_z

        assert jnp.allclose(computed_height, height, atol=1e-5), (
            "Cylinder height should match input"
        )

    def test_cylinder_radius(self):
        """Test that cylinder has correct radius."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 2.0
        height = 2.0
        solid = cylinder(center, radius, height, resolution=32)

        # Check radius for circle vertices (exclude center points)
        circle_vertices = solid.vertices[:-2]  # Exclude last 2 center points
        radii = jnp.sqrt(circle_vertices[:, 0] ** 2 + circle_vertices[:, 1] ** 2)

        # Filter out center points which have radius 0
        circle_radii = radii[radii > 0.5]
        assert jnp.allclose(circle_radii, radius, atol=1e-5), (
            "Circle vertices should be at correct radius"
        )

    def test_cylinder_vertices_valid(self):
        """Test that all cylinder vertices are finite."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 1.0
        height = 2.0
        solid = cylinder(center, radius, height, resolution=8)

        assert jnp.all(jnp.isfinite(solid.vertices)), "All vertices should be finite"
