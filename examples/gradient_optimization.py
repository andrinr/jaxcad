"""Example: Gradient-based parameter optimization with the new compilation system.

This example demonstrates:
1. Creating SDFs with explicit free/fixed parameters
2. Extracting parameters from an SDF tree
3. Compiling to a pure function for JAX gradients
4. Optimizing parameters using gradient descent
"""

import jax
import jax.numpy as jnp

from jaxcad.compiler import extract_parameters, compile_to_function
from jaxcad.parameters import Scalar, Vector
from jaxcad.primitives import Sphere
from jaxcad.transforms import Translate, Scale


def example_basic_optimization():
    """Optimize a sphere's radius to fit a target point."""
    print("=" * 70)
    print("EXAMPLE 1: Optimize Sphere Radius")
    print("=" * 70)

    # Create sphere with FREE radius parameter
    radius = Scalar(value=0.5, free=True, name='radius')
    sphere = Sphere(radius=radius)

    # Extract parameters
    free_params, fixed_params = extract_parameters(sphere)
    print(f"\nFree parameters: {list(free_params.keys())}")
    print(f"Fixed parameters: {list(fixed_params.keys())}")

    # Compile to pure function
    sdf_fn = compile_to_function(sphere)

    # Target: sphere surface should pass through this point
    target_point = jnp.array([2.0, 0.0, 0.0])
    print(f"\nGoal: Make sphere surface pass through {target_point}")

    # Define loss function
    def loss_fn(free_params):
        distance = sdf_fn(target_point, free_params, fixed_params)
        return distance ** 2

    # Initial loss
    initial_loss = loss_fn(free_params)
    print(f"Initial radius: {free_params['sphere_0.radius'].value}")
    print(f"Initial loss: {initial_loss:.6f}")

    # Optimize using JAX gradient descent
    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.1

    for i in range(50):
        # Compute gradient
        grad = grad_fn(free_params)

        # Update free parameters
        for key in free_params:
            param = free_params[key]
            param_grad = grad[key]
            # Update the parameter value
            new_value = param.value - learning_rate * param_grad.value
            free_params[key] = type(param)(value=new_value, free=True, name=param.name)

    # Final result
    final_loss = loss_fn(free_params)
    print(f"\nOptimized radius: {free_params['sphere_0.radius'].value}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"✓ Sphere surface now passes through target point!")


def example_multi_parameter():
    """Optimize multiple parameters together."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Optimize Multiple Parameters")
    print("=" * 70)

    # Create sphere with both radius and offset as free parameters
    radius = Scalar(value=1.0, free=True, name='radius')
    offset = Vector(value=[0.0, 0.0, 0.0], free=True, name='offset')

    sphere = Sphere(radius=radius)
    translated = Translate(sphere, offset=offset)

    # Extract and compile
    free_params, fixed_params = extract_parameters(translated)
    sdf_fn = compile_to_function(translated)

    print(f"\nFree parameters: {list(free_params.keys())}")

    # Target point
    target_point = jnp.array([3.0, 1.0, 0.0])
    print(f"Goal: Make surface pass through {target_point}")

    def loss_fn(free_params):
        distance = sdf_fn(target_point, free_params, fixed_params)
        return distance ** 2

    # Initial state
    print(f"\nInitial radius: {free_params['sphere_1.radius'].value}")
    print(f"Initial offset: {free_params['translate_0.offset'].xyz}")
    print(f"Initial loss: {loss_fn(free_params):.6f}")

    # Optimize
    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.1

    for i in range(50):
        grad = grad_fn(free_params)
        for key in free_params:
            param = free_params[key]
            param_grad = grad[key]

            if isinstance(param, Vector):
                # For vectors, update xyz coordinates
                new_xyz = param.xyz - learning_rate * param_grad.xyz
                free_params[key] = Vector(value=new_xyz, free=True, name=param.name)
            else:
                # For scalars
                new_value = param.value - learning_rate * param_grad.value
                free_params[key] = Scalar(value=new_value, free=True, name=param.name)

    # Final result
    print(f"\nOptimized radius: {free_params['sphere_1.radius'].value}")
    print(f"Optimized offset: {free_params['translate_0.offset'].xyz}")
    print(f"Final loss: {loss_fn(free_params):.6f}")
    print(f"✓ Found optimal configuration!")


def example_fixed_parameters():
    """Demonstrate fixed vs free parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Fixed vs Free Parameters")
    print("=" * 70)

    # Radius is FIXED, offset is FREE
    radius = Scalar(value=1.5, free=False, name='radius')  # Fixed!
    offset = Vector(value=[0.0, 0.0, 0.0], free=True, name='offset')  # Free
    scale = Vector(value=[1.0, 1.0, 2.0], free=False, name='scale')  # Fixed!

    sphere = Sphere(radius=radius)
    translated = Translate(sphere, offset=offset)
    scaled = Scale(translated, scale=scale)

    # Extract parameters
    free_params, fixed_params = extract_parameters(scaled)

    print(f"\nFree parameters (optimizable):")
    for key, param in free_params.items():
        print(f"  {key}: {param}")

    print(f"\nFixed parameters (constant):")
    for key, param in fixed_params.items():
        print(f"  {key}: {param}")

    # Compile
    sdf_fn = compile_to_function(scaled)

    # Optimize - only offset will change!
    target_point = jnp.array([2.0, 0.0, 0.0])

    def loss_fn(free_params):
        distance = sdf_fn(target_point, free_params, fixed_params)
        return distance ** 2

    print(f"\nInitial offset: {free_params['translate_1.offset'].xyz}")
    print(f"Fixed radius: {fixed_params['sphere_2.radius'].value}")
    print(f"Fixed scale: {fixed_params['scale_0.scale'].xyz}")

    # Optimize
    grad_fn = jax.jit(jax.grad(loss_fn))
    learning_rate = 0.1

    for i in range(50):
        grad = grad_fn(free_params)
        for key in free_params:
            param = free_params[key]
            param_grad = grad[key]
            new_xyz = param.xyz - learning_rate * param_grad.xyz
            free_params[key] = Vector(value=new_xyz, free=True, name=param.name)

    print(f"\nOptimized offset: {free_params['translate_1.offset'].xyz}")
    print(f"Radius (unchanged): {fixed_params['sphere_2.radius'].value}")
    print(f"Scale (unchanged): {fixed_params['scale_0.scale'].xyz}")
    print(f"✓ Only free parameters were optimized!")


def main():
    """Run all examples."""
    example_basic_optimization()
    example_multi_parameter()
    example_fixed_parameters()

    print("\n" + "=" * 70)
    print("SUMMARY: Gradient-Based Optimization")
    print("=" * 70)
    print("""
Key concepts:

1. EXPLICIT PARAMETERS:
   radius = Scalar(value=1.0, free=True, name='radius')
   offset = Vector(value=[0,0,0], free=False, name='offset')

   Mark parameters as free=True to make them optimizable.

2. PARAMETER EXTRACTION:
   free_params, fixed_params = extract_parameters(sdf)

   Separates parameters into two dictionaries based on their free flag.

3. COMPILATION:
   sdf_fn = compile_to_function(sdf)

   Compiles the SDF tree to a pure function:
   distance = sdf_fn(point, free_params, fixed_params)

4. OPTIMIZATION:
   grad = jax.grad(loss_fn)(free_params)

   Use JAX gradients to optimize free parameters.
   Fixed parameters remain constant during optimization.

5. FINE-GRAINED CONTROL:
   You decide exactly which parameters are optimizable and which are fixed.
   This allows for constrained optimization and better control over the process.

This approach gives you explicit control over the optimization process
while leveraging JAX's powerful automatic differentiation!
    """)


if __name__ == "__main__":
    main()
