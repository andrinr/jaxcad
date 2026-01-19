"""Tests for transformation operations."""

import jax.numpy as jnp

from jaxcad.primitives import box, sphere
from jaxcad.transforms import rotate, scale, transform, translate


class TestTranslate:
    """Tests for translation transformation."""

    def test_translate_basic(self):
        """Test basic translation."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([2.0, 2.0, 2.0])
        solid = box(center, size)

        offset = jnp.array([1.0, 2.0, 3.0])
        translated = translate(solid, offset)

        # Check that all vertices are translated by offset
        expected_vertices = solid.vertices + offset
        assert jnp.allclose(translated.vertices, expected_vertices), (
            "Translation should shift all vertices"
        )

    def test_translate_preserves_shape(self):
        """Test that translation preserves shape."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([2.0, 2.0, 2.0])
        solid = box(center, size)

        offset = jnp.array([5.0, 0.0, 0.0])
        translated = translate(solid, offset)

        # Compute relative positions (should be unchanged)
        original_relative = solid.vertices - jnp.mean(solid.vertices, axis=0)
        translated_relative = translated.vertices - jnp.mean(translated.vertices, axis=0)

        assert jnp.allclose(original_relative, translated_relative), (
            "Relative positions should be preserved"
        )

    def test_translate_faces_unchanged(self):
        """Test that translation doesn't change face indices."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        offset = jnp.array([1.0, 1.0, 1.0])
        translated = translate(solid, offset)

        assert jnp.array_equal(translated.faces, solid.faces), "Face indices should be unchanged"


class TestRotate:
    """Tests for rotation transformation."""

    def test_rotate_90_degrees(self):
        """Test 90-degree rotation around z-axis."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([2.0, 2.0, 2.0])
        solid = box(center, size)

        axis = jnp.array([0.0, 0.0, 1.0])
        angle = jnp.pi / 2  # 90 degrees
        rotated = rotate(solid, axis, angle)

        # A point at (1, 0, z) should rotate to approximately (0, 1, z)
        # Box vertices are at (±1, ±1, ±1), find vertex at (1, 0, 0) - but box has corners not edge midpoints
        # Let's find vertex at (1, 1, 1) which should rotate to (-1, 1, 1)
        target = jnp.array([1.0, 1.0, 1.0])
        distances = jnp.linalg.norm(solid.vertices - target, axis=1)
        idx = jnp.argmin(distances)

        rotated_point = rotated.vertices[idx]
        expected = jnp.array([-1.0, 1.0, 1.0])

        assert jnp.allclose(rotated_point, expected, atol=1e-5), (
            "90-degree rotation should work correctly"
        )

    def test_rotate_preserves_distances(self):
        """Test that rotation preserves distances from origin."""
        center = jnp.array([0.0, 0.0, 0.0])
        radius = 2.0
        solid = sphere(center, radius, resolution=8)

        axis = jnp.array([1.0, 1.0, 1.0])
        angle = jnp.pi / 3
        rotated = rotate(solid, axis, angle)

        # Distances from origin should be preserved
        original_distances = jnp.linalg.norm(solid.vertices, axis=1)
        rotated_distances = jnp.linalg.norm(rotated.vertices, axis=1)

        assert jnp.allclose(original_distances, rotated_distances, atol=1e-5), (
            "Rotation should preserve distances"
        )

    def test_rotate_360_degrees(self):
        """Test that 360-degree rotation returns to original position."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        axis = jnp.array([0.0, 1.0, 0.0])
        angle = 2 * jnp.pi  # 360 degrees
        rotated = rotate(solid, axis, angle)

        assert jnp.allclose(rotated.vertices, solid.vertices, atol=1e-5), (
            "360-degree rotation should return to original"
        )

    def test_rotate_with_origin(self):
        """Test rotation around a custom origin point."""
        center = jnp.array([2.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        axis = jnp.array([0.0, 0.0, 1.0])
        angle = jnp.pi / 2
        origin = jnp.array([0.0, 0.0, 0.0])
        rotated = rotate(solid, axis, angle, origin)

        # Box center should rotate from (2, 0, 0) to approximately (0, 2, 0)
        rotated_center = jnp.mean(rotated.vertices, axis=0)
        expected_center = jnp.array([0.0, 2.0, 0.0])

        assert jnp.allclose(rotated_center, expected_center, atol=1e-5), (
            "Rotation around custom origin should work"
        )


class TestScale:
    """Tests for scaling transformation."""

    def test_scale_uniform(self):
        """Test uniform scaling."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        factors = jnp.array([2.0, 2.0, 2.0])
        scaled = scale(solid, factors)

        # Vertices should be scaled by factors
        expected_vertices = solid.vertices * factors
        assert jnp.allclose(scaled.vertices, expected_vertices, atol=1e-5), (
            "Uniform scaling should work correctly"
        )

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([2.0, 2.0, 2.0])
        solid = box(center, size)

        factors = jnp.array([2.0, 1.0, 0.5])
        scaled = scale(solid, factors)

        # Check bounding box dimensions
        min_coords = jnp.min(scaled.vertices, axis=0)
        max_coords = jnp.max(scaled.vertices, axis=0)
        scaled_size = max_coords - min_coords

        expected_size = size * factors
        assert jnp.allclose(scaled_size, expected_size, atol=1e-5), (
            "Non-uniform scaling should work correctly"
        )

    def test_scale_with_origin(self):
        """Test scaling around a custom origin."""
        center = jnp.array([2.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        factors = jnp.array([2.0, 2.0, 2.0])
        origin = jnp.array([0.0, 0.0, 0.0])
        scaled = scale(solid, factors, origin)

        # Box center should scale from (2, 0, 0) to (4, 0, 0)
        scaled_center = jnp.mean(scaled.vertices, axis=0)
        expected_center = jnp.array([4.0, 0.0, 0.0])

        assert jnp.allclose(scaled_center, expected_center, atol=1e-5), (
            "Scaling around custom origin should work"
        )

    def test_scale_identity(self):
        """Test that scale by 1 returns original."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        factors = jnp.array([1.0, 1.0, 1.0])
        scaled = scale(solid, factors)

        assert jnp.allclose(scaled.vertices, solid.vertices, atol=1e-5), (
            "Scale by 1 should be identity"
        )


class TestTransform:
    """Tests for general transformation matrix."""

    def test_transform_identity(self):
        """Test identity transformation."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        matrix = jnp.eye(4)
        transformed = transform(solid, matrix)

        assert jnp.allclose(transformed.vertices, solid.vertices, atol=1e-5), (
            "Identity transform should not change vertices"
        )

    def test_transform_translation(self):
        """Test translation via transformation matrix."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        # Translation matrix
        matrix = jnp.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
        )
        transformed = transform(solid, matrix)

        offset = jnp.array([1.0, 2.0, 3.0])
        expected_vertices = solid.vertices + offset

        assert jnp.allclose(transformed.vertices, expected_vertices, atol=1e-5), (
            "Translation matrix should work correctly"
        )

    def test_transform_preserves_faces(self):
        """Test that transformation doesn't change face indices."""
        center = jnp.array([0.0, 0.0, 0.0])
        size = jnp.array([1.0, 1.0, 1.0])
        solid = box(center, size)

        matrix = jnp.eye(4)
        transformed = transform(solid, matrix)

        assert jnp.array_equal(transformed.faces, solid.faces), "Face indices should be unchanged"
