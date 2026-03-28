"""Tests for constraint integration with optimization."""

import jax
import jax.numpy as jnp
import pytest

from jaxcad.constraints import DistanceConstraint, null_space
from jaxcad.geometry.parameters import Vector


def _free_and_meta(*params):
    return (
        {p.name: p.value for p in params},
        {p.name: p for p in params},
    )


def test_constrained_optimization_gradient_flow():
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)
    base_vec = N.pack(free)
    reduced = jnp.zeros(N.shape[1])

    def loss_fn(r):
        p1_new = (base_vec + N.matrix @ r)[:3]
        return jnp.sum((p1_new - jnp.array([0.0, 1.0, 0.0])) ** 2)

    initial_loss = loss_fn(reduced)
    grad_fn = jax.grad(loss_fn)

    grad = grad_fn(reduced)
    assert grad.shape == reduced.shape
    assert not jnp.any(jnp.isnan(grad))

    current = reduced
    for _ in range(20):
        current = current - 0.05 * grad_fn(current)

    assert loss_fn(current) < initial_loss


def test_constraint_preserves_dof_in_optimization():
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    p3 = Vector([0.5, 0.5, 0], free=True, name="p3")
    DistanceConstraint(p1, p2, 1.0)
    DistanceConstraint(p1, p3, 0.7071)

    free, meta = _free_and_meta(p1, p2, p3)
    N = null_space(free, meta)

    assert N.shape[1] == 7

    base_vec = N.pack(free)
    reduced = jnp.zeros(N.shape[1])

    def loss_fn(r):
        return jnp.sum((base_vec + N.matrix @ r)[2::3])

    grad = jax.grad(loss_fn)(reduced)
    assert grad.shape == (7,)
    assert not jnp.any(jnp.isnan(grad))


@pytest.mark.parametrize("learning_rate", [0.01, 0.05, 0.1])
def test_optimization_convergence_with_learning_rate(learning_rate):
    p1 = Vector([0, 0, 0], free=True, name="p1")
    p2 = Vector([1, 0, 0], free=True, name="p2")
    DistanceConstraint(p1, p2, 1.0)

    free, meta = _free_and_meta(p1, p2)
    N = null_space(free, meta)
    base_vec = N.pack(free)
    reduced = jnp.zeros(N.shape[1])

    def loss_fn(r):
        p1_new = (base_vec + N.matrix @ r)[:3]
        return jnp.sum((p1_new - jnp.array([0.0, 1.0, 0.0])) ** 2)

    grad_fn = jax.grad(loss_fn)
    current = reduced
    for _ in range(50):
        current = current - learning_rate * grad_fn(current)

    assert loss_fn(current) < loss_fn(reduced)
