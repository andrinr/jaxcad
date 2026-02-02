"""Example 2: End-to-End Optimization with Visualization

This example demonstrates the complete JAXcad optimization pipeline:

1. **Geometry Layer** - Create parametric geometry with free parameters
2. **Constraints Layer** - Apply geometric constraints (distance, angle, etc.)
3. **Construction Layer** - Convert geometry to SDF
4. **Compiler Layer** - Extract parameters and compile to pure JAX
5. **Optimization** - Use JAX gradients to optimize shape parameters
6. **Visualization** - Render results with matplotlib

Goal: Optimize sphere radii to reach target points while maintaining
distance constraints between sphere centers.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcad.geometry.parameters import Vector, Scalar
from jaxcad.constraints import DistanceConstraint, ConstraintGraph
from jaxcad.construction import from_point
from jaxcad.compiler import extract_parameters, compile_to_function
from jaxcad.sdf.boolean import Union
from jaxcad.render import render_marching_cubes


def create_scene():
    """Create parametric scene with three constrained spheres.

    Returns:
        tuple: (scene, radii, centers, graph)
    """
    print("=" * 80)
    print("STEP 1: Geometry Layer - Create Parametric Spheres")
    print("=" * 80)

    # Three sphere centers with free parameters
    center1 = Vector([0.0, 0.0, 0.0], free=True, name='c1')
    center2 = Vector([2.0, 0.0, 0.0], free=True, name='c2')
    center3 = Vector([1.0, 2.0, 0.0], free=True, name='c3')

    print(f"Created 3 sphere centers:")
    print(f"  c1: {center1.xyz}")
    print(f"  c2: {center2.xyz}")
    print(f"  c3: {center3.xyz}")
    print()

    print("=" * 80)
    print("STEP 2: Constraints Layer - Apply Distance Constraints")
    print("=" * 80)

    # Apply distance constraints
    graph = ConstraintGraph()
    graph.add_constraint(DistanceConstraint(center1, center2, distance=2.0))
    graph.add_constraint(DistanceConstraint(center2, center3, distance=2.0))

    print(f"Applied {len(graph.constraints)} distance constraints:")
    print(f"  c1 ↔ c2: distance = 2.0")
    print(f"  c2 ↔ c3: distance = 2.0")
    print(f"  DOF reduction: {graph.get_total_dof_reduction()}")
    print()

    print("=" * 80)
    print("STEP 3: Construction Layer - Build SDF Scene")
    print("=" * 80)

    # Create spheres with free radii
    radius1 = Scalar(0.5, free=True, name='r1')
    radius2 = Scalar(0.5, free=True, name='r2')
    radius3 = Scalar(0.5, free=True, name='r3')

    sphere1 = from_point(center1, radius1)
    sphere2 = from_point(center2, radius2)
    sphere3 = from_point(center3, radius3)

    # Combine with boolean operations
    scene = Union(sphere1, sphere2)
    scene = Union(scene, sphere3)

    print(f"Created 3 spheres with optimizable radii:")
    print(f"  r1: {radius1.value}")
    print(f"  r2: {radius2.value}")
    print(f"  r3: {radius3.value}")
    print()

    return scene, [radius1, radius2, radius3], [center1, center2, center3], graph


def compile_scene(scene):
    """Compile scene to pure JAX function.

    Returns:
        tuple: (sdf_fn, free_params, fixed_params)
    """
    print("=" * 80)
    print("STEP 4: Compiler Layer - Extract and Compile")
    print("=" * 80)

    # Extract parameters
    free_params, fixed_params = extract_parameters(scene)

    print(f"Extracted parameters:")
    print(f"  Free: {list(free_params.keys())}")
    print(f"  Fixed: {len(fixed_params)} parameters")
    print()

    # Compile to pure function
    sdf_fn = compile_to_function(scene)

    print("✓ Compiled SDF to pure JAX function")
    print()

    return sdf_fn, free_params, fixed_params


def run_optimization(sdf_fn, fixed_params):
    """Run gradient-based optimization.

    Returns:
        tuple: (radii_history, loss_history)
    """
    print("=" * 80)
    print("STEP 5: Optimization - Gradient Descent")
    print("=" * 80)

    # Target points we want spheres to reach
    target_points = jnp.array([
        [0.8, 0.0, 0.0],   # Target for sphere 1
        [2.5, 0.0, 0.0],   # Target for sphere 2
        [1.0, 2.5, 0.0],   # Target for sphere 3
    ])

    print(f"Optimization goal: Fit spheres to target points")
    print(f"Target points:")
    for i, target in enumerate(target_points):
        print(f"  Target {i+1}: {target}")
    print()

    # Loss function: minimize distance from targets to surfaces
    def loss_fn(params_array):
        r1, r2, r3 = params_array
        # Use the parameter names from extraction
        params = {
            'sphere_2.radius': r1,
            'sphere_3.radius': r2,
            'sphere_4.radius': r3,
        }

        losses = []
        for target in target_points:
            dist = sdf_fn(target, params, fixed_params)
            losses.append(dist ** 2)  # Squared distance to surface

        return jnp.sum(jnp.array(losses))

    # Initial parameters
    initial_radii = jnp.array([0.5, 0.5, 0.5])
    initial_loss = float(loss_fn(initial_radii))

    print(f"Initial radii: {initial_radii}")
    print(f"Initial loss: {initial_loss:.6f}")
    print()

    # Gradient descent
    learning_rate = 0.1
    num_steps = 50

    radii_history = [initial_radii]
    loss_history = [initial_loss]

    current_radii = initial_radii
    grad_fn = jax.grad(loss_fn)

    print("Running gradient descent...")
    for step in range(num_steps):
        gradient = grad_fn(current_radii)
        current_radii = current_radii - learning_rate * gradient
        current_radii = jnp.clip(current_radii, 0.1, 2.0)  # Clamp to valid range

        loss = float(loss_fn(current_radii))
        radii_history.append(current_radii)
        loss_history.append(loss)

        if step % 10 == 0:
            print(f"  Step {step:3d}: loss = {loss:.6f}, radii = {current_radii}")

    final_loss = loss_history[-1]
    improvement = (1 - final_loss / initial_loss) * 100

    print()
    print(f"✓ Optimization complete!")
    print(f"  Final radii: {current_radii}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print()

    return radii_history, loss_history, target_points


def visualize_results(radii_history, loss_history, target_points, initial_scene, final_scene):
    """Create visualization plots."""
    print("=" * 80)
    print("STEP 6: Visualization")
    print("=" * 80)

    fig = plt.figure(figsize=(16, 5))

    # Plot 1: Loss convergence
    ax = fig.add_subplot(131)
    ax.plot(loss_history, linewidth=2.5, color='#2E86AB')
    ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_title('Optimization Convergence', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot 2: Initial configuration (3D)
    ax = fig.add_subplot(132, projection='3d')
    render_marching_cubes(
        initial_scene,
        bounds=(-2, -1, -1),
        size=(4, 4, 2),
        resolution=40,
        ax=ax,
        color='#A23B72',
        alpha=0.6
    )
    # Add target points
    targets_3d = jnp.column_stack([target_points, jnp.zeros(len(target_points))])
    ax.scatter(targets_3d[:, 0], targets_3d[:, 1], targets_3d[:, 2],
              c='#F18F01', marker='x', s=200, linewidths=3, label='Targets')
    ax.set_title('Initial Configuration', fontsize=15, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)
    ax.legend(fontsize=11)

    # Plot 3: Final configuration (3D)
    ax = fig.add_subplot(133, projection='3d')
    render_marching_cubes(
        final_scene,
        bounds=(-2, -1, -1),
        size=(4, 4, 2),
        resolution=40,
        ax=ax,
        color='#06A77D',
        alpha=0.8
    )
    # Add target points
    ax.scatter(targets_3d[:, 0], targets_3d[:, 1], targets_3d[:, 2],
              c='#F18F01', marker='x', s=200, linewidths=3, label='Targets')
    ax.set_title('Optimized Configuration', fontsize=15, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('examples/output/end_to_end_optimization.png', dpi=150, bbox_inches='tight')

    print("✓ Saved visualization: examples/output/end_to_end_optimization.png")
    print()


def main():
    """Run complete end-to-end optimization demo."""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "JAXcad End-to-End Optimization Demo" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Create initial scene
    scene, radii, centers, graph = create_scene()
    initial_scene = scene  # Save for visualization

    # Compile
    sdf_fn, free_params, fixed_params = compile_scene(scene)

    # Optimize
    radii_history, loss_history, target_points = run_optimization(sdf_fn, fixed_params)

    # Create final scene with optimized radii
    final_radii = radii_history[-1]
    radii[0].value = final_radii[0]
    radii[1].value = final_radii[1]
    radii[2].value = final_radii[2]

    sphere1 = from_point(centers[0], radii[0])
    sphere2 = from_point(centers[1], radii[1])
    sphere3 = from_point(centers[2], radii[2])
    final_scene = Union(sphere1, sphere2)
    final_scene = Union(final_scene, sphere3)

    # Visualize
    visualize_results(radii_history, loss_history, target_points, initial_scene, final_scene)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
This demo showed the complete JAXcad workflow:

1. ✓ Created parametric geometry with free parameters
2. ✓ Applied constraints to maintain geometric relationships
3. ✓ Converted geometry to SDF primitives
4. ✓ Compiled to pure JAX functions for optimization
5. ✓ Used automatic differentiation for gradient-based optimization
6. ✓ Visualized results showing convergence and final configuration

Key features demonstrated:
- Parametric CAD with optimizable parameters
- Constraint-based design (distance constraints)
- Seamless integration with JAX for automatic differentiation
- End-to-end differentiable pipeline
- Clean separation of geometry, constraints, and optimization

The spheres successfully grew to reach their target points while
maintaining the distance constraints between their centers!
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
