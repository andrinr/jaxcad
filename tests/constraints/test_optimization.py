"""Tests for constraint integration with optimization."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.constraints import ConstraintGraph, DistanceConstraint
from jaxcad.geometry.parameters import Vector


def test_constrained_optimization_gradient_flow():
    """Test that gradients flow correctly through constrained optimization."""
    # Two points that should maintain distance 1.0
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    graph = ConstraintGraph()
    graph.add_constraint(DistanceConstraint(p1, p2, 1.0))

    # Get reduced DOF
    reduced, null_space = graph.extract_free_dof([p1, p2])

    # Base point
    base_point = jnp.concatenate([p1.xyz, p2.xyz])

    # Define a loss function: move p1 to [0, 1, 0]
    def loss_fn(reduced_params):
        full = base_point + null_space @ reduced_params
        p1_new = full[:3]
        target = jnp.array([0.0, 1.0, 0.0])
        return jnp.sum((p1_new - target) ** 2)

    # Initial loss
    initial_loss = loss_fn(reduced)

    # Gradient
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(reduced)

    # Gradient should be well-defined
    assert grad.shape == reduced.shape
    assert not jnp.any(jnp.isnan(grad))

    # Perform optimization steps
    learning_rate = 0.05
    current_reduced = reduced

    for _ in range(20):
        grad = grad_fn(current_reduced)
        current_reduced = current_reduced - learning_rate * grad

    # Check that loss decreased
    final_loss = loss_fn(current_reduced)
    assert final_loss < initial_loss


def test_constraint_preserves_dof_in_optimization():
    """Test that constraints correctly reduce DOF during optimization."""
    # Three points with two distance constraints
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    p3 = Vector([0.5, 0.5, 0], free=True, name="p3")

    graph = ConstraintGraph()
    graph.add_constraint(DistanceConstraint(p1, p2, 1.0))
    graph.add_constraint(DistanceConstraint(p1, p3, 0.7071))  # ~√2/2

    # 9 DOF - 2 constraints = 7 DOF
    reduced, null_space = graph.extract_free_dof([p1, p2, p3])

    assert reduced.shape[0] == 7

    # Base point
    base_point = jnp.concatenate([p1.xyz, p2.xyz, p3.xyz])

    # Define optimization problem: minimize z-coordinates
    def loss_fn(reduced_params):
        full = base_point + null_space @ reduced_params
        # Minimize sum of z-coordinates
        return jnp.sum(full[2::3])  # Every 3rd element starting from index 2

    # Check gradients work
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(reduced)

    assert grad.shape == (7,)
    assert not jnp.any(jnp.isnan(grad))


@pytest.mark.parametrize("learning_rate", [0.01, 0.05, 0.1])
def test_optimization_convergence_with_learning_rate(learning_rate):
    """Test optimization convergence with different learning rates."""
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")

    graph = ConstraintGraph()
    graph.add_constraint(DistanceConstraint(p1, p2, 1.0))

    reduced, null_space = graph.extract_free_dof([p1, p2])
    base_point = jnp.concatenate([p1.xyz, p2.xyz])

    def loss_fn(reduced_params):
        full = base_point + null_space @ reduced_params
        p1_new = full[:3]
        target = jnp.array([0.0, 1.0, 0.0])
        return jnp.sum((p1_new - target) ** 2)

    initial_loss = loss_fn(reduced)

    grad_fn = jax.grad(loss_fn)
    current_reduced = reduced

    for _ in range(50):
        grad = grad_fn(current_reduced)
        current_reduced = current_reduced - learning_rate * grad

    final_loss = loss_fn(current_reduced)

    # All learning rates should reduce loss
    assert final_loss < initial_loss
