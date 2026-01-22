"""Functional transform API with full differentiability.

This module provides functional transforms where parameters are passed
explicitly, enabling gradient flow through transform parameters.
"""

import jax.numpy as jnp
from jax import Array


def translate_eval(sdf_fn, p: Array, offset: Array) -> Array:
    """Evaluate translated SDF - fully differentiable.

    Args:
        sdf_fn: SDF function to transform
        p: Query point(s)
        offset: Translation vector

    Returns:
        SDF value at translated point
    """
    return sdf_fn(p - offset)


def scale_eval(sdf_fn, p: Array, scale: float | Array) -> Array:
    """Evaluate scaled SDF - fully differentiable.

    Args:
        sdf_fn: SDF function to transform
        p: Query point(s)
        scale: Scale factor

    Returns:
        SDF value at scaled point
    """
    scale_arr = jnp.asarray(scale)
    is_uniform = scale_arr.ndim == 0
    if is_uniform:
        return sdf_fn(p / scale_arr) * scale_arr
    else:
        return sdf_fn(p / scale_arr)


def rotate_z_eval(sdf_fn, p: Array, angle: float) -> Array:
    """Evaluate rotated SDF around Z axis - fully differentiable.

    Args:
        sdf_fn: SDF function to transform
        p: Query point(s)
        angle: Rotation angle in radians

    Returns:
        SDF value at rotated point
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    # Rotation matrix transposed (inverse rotation)
    x = p[..., 0] * c + p[..., 1] * s
    y = -p[..., 0] * s + p[..., 1] * c
    z = p[..., 2]
    p_rotated = jnp.stack([x, y, z], axis=-1)
    return sdf_fn(p_rotated)


def twist_z_eval(sdf_fn, p: Array, strength: float) -> Array:
    """Evaluate twisted SDF around Z axis - fully differentiable.

    Args:
        sdf_fn: SDF function to transform
        p: Query point(s)
        strength: Twist strength (radians per unit length)

    Returns:
        SDF value at twisted point
    """
    z = p[..., 2]
    angle = z * strength
    c, s = jnp.cos(angle), jnp.sin(angle)
    x = p[..., 0] * c + p[..., 1] * s
    y = -p[..., 0] * s + p[..., 1] * c
    p_twisted = jnp.stack([x, y, z], axis=-1)
    return sdf_fn(p_twisted)


def taper_z_eval(sdf_fn, p: Array, strength: float) -> Array:
    """Evaluate tapered SDF along Z axis - fully differentiable.

    Args:
        sdf_fn: SDF function to transform
        p: Query point(s)
        strength: Taper strength

    Returns:
        SDF value at tapered point
    """
    z = p[..., 2]
    scale = 1.0 + z * strength
    x = p[..., 0] / scale
    y = p[..., 1] / scale
    p_tapered = jnp.stack([x, y, z], axis=-1)
    return sdf_fn(p_tapered)


"""
Example of fully differentiable transform parameters:

```python
import jax
import jax.numpy as jnp
from jaxcad.primitives import Sphere
from jaxcad.transforms.functional import translate_eval

# Create base SDF
sphere = Sphere(radius=1.0)

# Define loss function with respect to offset
def loss(offset):
    point = jnp.array([3.0, 0.0, 0.0])
    return translate_eval(sphere, point, offset) ** 2

# Compute gradient with respect to offset
grad_fn = jax.grad(loss)
gradient = grad_fn(jnp.array([0.0, 0.0, 0.0]))
print(f"Gradient: {gradient}")

# Optimize offset to place sphere at target position
offset = jnp.array([0.0, 0.0, 0.0])
learning_rate = 0.1
for _ in range(50):
    offset = offset - learning_rate * grad_fn(offset)
print(f"Optimized offset: {offset}")  # Should be ~[2, 0, 0]
```
"""
