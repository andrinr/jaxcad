"""Tests for gradient computation and correctness."""

import jax
import jax.numpy as jnp

from jaxcad.primitives import box, cylinder, sphere
from jaxcad.transforms import rotate, scale, translate


def numerical_gradient(f, x, eps=1e-4):
    """Compute numerical gradient using finite differences.

    Args:
        f: Function to differentiate
        x: Point at which to compute gradient
        eps: Finite difference step size

    Returns:
        Numerical gradient approximation
    """
    grad = jnp.zeros_like(x)
    x_flat = x.flatten()
    grad_flat = grad.flatten()

    for i in range(len(x_flat)):
        x_plus = x_flat.at[i].add(eps)
        x_minus = x_flat.at[i].add(-eps)

        f_plus = f(x_plus.reshape(x.shape))
        f_minus = f(x_minus.reshape(x.shape))

        grad_flat = grad_flat.at[i].set((f_plus - f_minus) / (2 * eps))

    return grad_flat.reshape(x.shape)


class TestBoxGradients:
    """Tests for box primitive gradients."""

    def test_box_gradient_center(self):
        """Test gradient of vertex position w.r.t. box center."""

        def vertex_x(center):
            size = jnp.array([1.0, 1.0, 1.0])
            solid = box(center, size)
            return solid.vertices[0, 0]  # x-coordinate of first vertex

        center = jnp.array([2.0, 3.0, 4.0])

        # Analytical gradient using JAX
        grad_analytical = jax.grad(vertex_x)(center)

        # Numerical gradient
        grad_numerical = numerical_gradient(vertex_x, center)

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-3), (
            f"Analytical gradient {grad_analytical} should match numerical {grad_numerical}"
        )

    def test_box_gradient_size(self):
        """Test gradient of vertex position w.r.t. box size."""

        def vertex_position(size):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = box(center, size)
            return jnp.sum(solid.vertices[0])  # Sum of coordinates of first vertex

        size = jnp.array([2.0, 3.0, 4.0])

        # Analytical gradient
        grad_analytical = jax.grad(vertex_position)(size)

        # Numerical gradient
        grad_numerical = numerical_gradient(vertex_position, size)

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-3), (
            "Gradient of vertex position w.r.t. size should be correct"
        )

    def test_box_volume_gradient(self):
        """Test gradient of box volume w.r.t. size."""

        def volume(size):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = box(center, size)
            min_coords = jnp.min(solid.vertices, axis=0)
            max_coords = jnp.max(solid.vertices, axis=0)
            dims = max_coords - min_coords
            return jnp.prod(dims)

        size = jnp.array([2.0, 3.0, 4.0])

        # Analytical gradient
        grad_analytical = jax.grad(volume)(size)

        # Expected: gradient of volume w.r.t each dimension is product of other two
        # ∂V/∂x = y*z, ∂V/∂y = x*z, ∂V/∂z = x*y
        grad_expected = jnp.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])

        assert jnp.allclose(grad_analytical, grad_expected, atol=1e-4), (
            f"Volume gradient {grad_analytical} should match expected {grad_expected}"
        )


class TestSphereGradients:
    """Tests for sphere primitive gradients."""

    def test_sphere_gradient_center(self):
        """Test gradient of vertex position w.r.t. sphere center."""

        def vertex_position(center):
            radius = 2.0
            solid = sphere(center, radius, resolution=8)
            return jnp.sum(solid.vertices[5])  # Sum of coordinates of vertex 5

        center = jnp.array([1.0, 2.0, 3.0])

        # Analytical gradient
        grad_analytical = jax.grad(vertex_position)(center)

        # Numerical gradient
        grad_numerical = numerical_gradient(vertex_position, center)

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-3), (
            "Gradient of sphere vertex w.r.t. center should be correct"
        )

    def test_sphere_gradient_radius(self):
        """Test gradient of vertex distance w.r.t. radius."""

        def max_distance(radius):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = sphere(center, radius, resolution=8)
            distances = jnp.linalg.norm(solid.vertices - center, axis=1)
            return jnp.max(distances)

        radius = 2.5

        # Analytical gradient
        grad_analytical = jax.grad(max_distance)(radius)

        # Expected: max distance should equal radius, so gradient should be 1
        assert jnp.allclose(grad_analytical, 1.0, atol=1e-4), (
            "Gradient of max distance w.r.t. radius should be 1"
        )


class TestCylinderGradients:
    """Tests for cylinder primitive gradients."""

    def test_cylinder_gradient_radius(self):
        """Test gradient of vertex position w.r.t. cylinder radius."""

        def vertex_x(radius):
            center = jnp.array([0.0, 0.0, 0.0])
            height = 2.0
            solid = cylinder(center, radius, height, resolution=8)
            return solid.vertices[0, 0]  # x-coordinate of first vertex

        radius = 1.5

        # Analytical gradient
        grad_analytical = jax.grad(vertex_x)(radius)

        # Numerical gradient
        grad_numerical = numerical_gradient(vertex_x, jnp.array(radius))

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-3), (
            "Gradient of cylinder vertex w.r.t. radius should be correct"
        )

    def test_cylinder_gradient_height(self):
        """Test gradient of vertex position w.r.t. cylinder height."""

        def vertex_z(height):
            center = jnp.array([0.0, 0.0, 0.0])
            radius = 1.0
            solid = cylinder(center, radius, height, resolution=8)
            # Get a top vertex (not center)
            return solid.vertices[8, 2]  # z-coordinate of first top vertex

        height = 3.0

        # Analytical gradient
        grad_analytical = jax.grad(vertex_z)(height)

        # Numerical gradient
        grad_numerical = numerical_gradient(vertex_z, jnp.array(height))

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-3), (
            "Gradient of cylinder vertex z w.r.t. height should be correct"
        )


class TestTransformGradients:
    """Tests for transformation gradients."""

    def test_translate_gradient(self):
        """Test gradient through translation."""

        def translated_vertex_sum(offset):
            center = jnp.array([0.0, 0.0, 0.0])
            size = jnp.array([1.0, 1.0, 1.0])
            solid = box(center, size)
            translated = translate(solid, offset)
            return jnp.sum(translated.vertices[0])

        offset = jnp.array([1.0, 2.0, 3.0])

        # Analytical gradient
        grad_analytical = jax.grad(translated_vertex_sum)(offset)

        # Expected: gradient should be [1, 1, 1] since we sum all coordinates
        grad_expected = jnp.array([1.0, 1.0, 1.0])

        assert jnp.allclose(grad_analytical, grad_expected, atol=1e-4), (
            "Gradient through translation should be correct"
        )

    def test_rotate_gradient(self):
        """Test gradient through rotation."""

        def rotated_vertex_x(angle):
            center = jnp.array([0.0, 0.0, 0.0])
            size = jnp.array([2.0, 2.0, 2.0])
            solid = box(center, size)
            axis = jnp.array([0.0, 0.0, 1.0])
            rotated = rotate(solid, axis, angle)
            return rotated.vertices[1, 0]  # x-coordinate of vertex 1

        angle = jnp.pi / 6  # 30 degrees

        # Analytical gradient
        grad_analytical = jax.grad(rotated_vertex_x)(angle)

        # Numerical gradient
        grad_numerical = numerical_gradient(rotated_vertex_x, jnp.array(angle))

        assert jnp.allclose(grad_analytical, grad_numerical, atol=1e-3), (
            "Gradient through rotation should be correct"
        )

    def test_scale_gradient(self):
        """Test gradient through scaling."""

        def scaled_vertex_position(factors):
            center = jnp.array([0.0, 0.0, 0.0])
            size = jnp.array([1.0, 1.0, 1.0])
            solid = box(center, size)
            scaled = scale(solid, factors)
            return jnp.sum(scaled.vertices[0])

        factors = jnp.array([2.0, 3.0, 4.0])

        # Analytical gradient
        grad_analytical = jax.grad(scaled_vertex_position)(factors)

        # Numerical gradient
        grad_numerical = numerical_gradient(scaled_vertex_position, factors)

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-2), (
            "Gradient through scaling should be correct"
        )


class TestComposedGradients:
    """Tests for gradients through composed operations."""

    def test_composed_transforms_gradient(self):
        """Test gradient through multiple transformations."""

        def composed_pipeline(params):
            center, size_scale, trans_x, rot_angle = params[0], params[1], params[2], params[3]

            # Create box with parametric size
            box_center = jnp.array([0.0, 0.0, 0.0])
            size = jnp.array([size_scale, size_scale, size_scale])
            solid = box(box_center, size)

            # Translate
            solid = translate(solid, jnp.array([trans_x, 0.0, 0.0]))

            # Rotate
            axis = jnp.array([0.0, 0.0, 1.0])
            solid = rotate(solid, axis, rot_angle)

            # Scale
            solid = scale(solid, jnp.array([center, 1.0, 1.0]))

            return jnp.sum(solid.vertices[0])

        params = jnp.array([1.5, 2.0, 1.0, jnp.pi / 4])

        # Analytical gradient using JAX
        grad_analytical = jax.grad(composed_pipeline)(params)

        # Numerical gradient
        grad_numerical = numerical_gradient(composed_pipeline, params)

        assert jnp.allclose(grad_analytical, grad_numerical, atol=5e-2), (
            "Gradient through composed transformations should be correct"
        )

    def test_gradient_of_all_vertices(self):
        """Test that we can compute gradients for all vertices simultaneously."""

        def all_vertices_loss(size):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = box(center, size)
            # Sum of squared vertex coordinates (to create dependency on size)
            return jnp.sum(solid.vertices**2)

        size = jnp.array([1.0, 2.0, 3.0])

        # Compute gradient
        grad = jax.grad(all_vertices_loss)(size)

        # Check that gradient is finite
        assert jnp.all(jnp.isfinite(grad)), "Gradient of all vertices should be finite"

        # Check that gradient is non-zero (vertices depend on size)
        assert jnp.all(grad != 0), "Gradient should be non-zero"


class TestVectorization:
    """Tests for vectorized gradient computation."""

    def test_vmap_gradient(self):
        """Test that gradients work with vmap."""

        def vertex_position(size):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = box(center, size)
            return jnp.sum(solid.vertices[0])

        # Multiple sizes to process
        sizes = jnp.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )

        # Vectorize gradient computation
        grad_fn = jax.grad(vertex_position)
        batched_grad_fn = jax.vmap(grad_fn)

        grads = batched_grad_fn(sizes)

        assert grads.shape == (3, 3), "Batched gradients should have correct shape"
        assert jnp.all(jnp.isfinite(grads)), "All batched gradients should be finite"

    def test_jacobian_computation(self):
        """Test Jacobian computation for vector outputs."""

        def vertex_position(size):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = box(center, size)
            return solid.vertices[0]  # Returns 3D vector

        size = jnp.array([2.0, 3.0, 4.0])

        # Compute Jacobian (3x3 matrix)
        jacobian = jax.jacfwd(vertex_position)(size)

        assert jacobian.shape == (3, 3), "Jacobian should be 3x3"
        assert jnp.all(jnp.isfinite(jacobian)), "Jacobian should be finite"

    def test_hessian_computation(self):
        """Test second-order derivatives (Hessian)."""

        def scalar_output(size):
            center = jnp.array([0.0, 0.0, 0.0])
            solid = box(center, size)
            return jnp.sum(solid.vertices[0] ** 2)

        size = jnp.array([1.0, 2.0, 3.0])

        # Compute Hessian
        hessian = jax.hessian(scalar_output)(size)

        assert hessian.shape == (3, 3), "Hessian should be 3x3"
        assert jnp.all(jnp.isfinite(hessian)), "Hessian should be finite"
