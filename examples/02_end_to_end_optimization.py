"""Example 2: Constrained Optimization with Null Space Projection

Demonstrates:
- Implicit constraint discovery (no explicit ConstraintGraph)
- Null space projection for constrained optimization
- 3D visualization of optimization progress
"""

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio

from jaxcad.geometry.parameters import Vector, Scalar
from jaxcad.constraints import DistanceConstraint
from jaxcad.construction import from_point
from jaxcad.compiler import extract_parameters, extract_parameters_with_constraints, to_function
from jaxcad.sdf.boolean import Union
from jaxcad.render import render_marching_cubes

print("\nJAXcad: Constrained Optimization with Null Space Projection")
print("=" * 60)

# Create parametric geometry with constraints
center1 = Vector([0.0, 0.0, 0.0], free=True, name='c1')
center2 = Vector([0.2, 0.0, 0.0], free=True, name='c2')
center3 = Vector([0.1, 0.1, 0.0], free=True, name='c3')

DistanceConstraint(center1, center2, distance=2.0)
DistanceConstraint(center2, center3, distance=2.0)

radius1 = Scalar(0.5, free=True, name='r1')
radius2 = Scalar(0.5, free=True, name='r2')
radius3 = Scalar(0.5, free=True, name='r3')

sphere1 = from_point(center1, radius1)
sphere2 = from_point(center2, radius2)
sphere3 = from_point(center3, radius3)

initial_scene = Union(Union(sphere1, sphere2), sphere3)

# Extract parameters and discover constraints
reduced_params, null_space, base_point, param_list = extract_parameters_with_constraints(initial_scene)

print(f"Parameters: {base_point.shape[0]} DOF → {reduced_params.shape[0]} DOF (null space)")
print(f"Constraints: 2 distance constraints (auto-discovered)")

# Compile SDF
sdf_fn = to_function(initial_scene)
free_params, fixed_params = extract_parameters(initial_scene)

# Target points
target_points = jnp.array([
    [0.8, 0.0, 0.0],
    [2.5, 0.0, 0.0],
    [1.0, 2.5, 0.0],
])

# Loss function in reduced space
def loss_fn(reduced):
    full_params_flat = base_point + null_space @ reduced
    r1, r2, r3 = full_params_flat[-3:]

    params = {
        'sphere_2.radius': r1,
        'sphere_3.radius': r2,
        'sphere_4.radius': r3,
    }

    losses = []
    for target in target_points:
        dist = sdf_fn(target, params, fixed_params)
        losses.append(dist ** 2)

    return jnp.sum(jnp.array(losses))

# Optimize
initial_loss = float(loss_fn(reduced_params))
print(f"\nOptimizing (initial loss: {initial_loss:.3f})...")

learning_rate = 0.1
num_steps = 8
current_reduced = reduced_params
grad_fn = jax.grad(loss_fn)

# Store history for animation
reduced_history = [reduced_params]

for step in range(num_steps):
    gradient = grad_fn(current_reduced)
    current_reduced = current_reduced - learning_rate * gradient
    print(f"current reduced params (step {step+1}): {current_reduced}")
    reduced_history.append(current_reduced)

final_loss = float(loss_fn(current_reduced))
print(f"Final loss: {final_loss:.3f} (improvement: {(1 - final_loss/initial_loss)*100:.1f}%)")

# Project back to full space
final_full_params = base_point + null_space @ current_reduced
final_radii = final_full_params[-3:]

# Build final scene
final_r1 = Scalar(float(final_radii[0]), free=False, name='r1_final')
final_r2 = Scalar(float(final_radii[1]), free=False, name='r2_final')
final_r3 = Scalar(float(final_radii[2]), free=False, name='r3_final')

final_sphere1 = from_point(center1, final_r1)
final_sphere2 = from_point(center2, final_r2)
final_sphere3 = from_point(center3, final_r3)

final_scene = Union(Union(final_sphere1, final_sphere2), final_sphere3)

# Visualize
print("\nRendering...")
fig = plt.figure(figsize=(12, 5))

# Initial configuration
ax1 = fig.add_subplot(121, projection='3d')
render_marching_cubes(
    initial_scene,
    bounds=(-1, -1, -1),
    size=(4, 4, 3),
    resolution=40,
    ax=ax1,
    color='#A23B72',
    alpha=0.7
)
ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
           c='#F18F01', marker='x', s=200, linewidths=3, label='Targets')
ax1.set_title('Initial Configuration', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.view_init(elev=20, azim=45)
ax1.legend()

# Optimized configuration
ax2 = fig.add_subplot(122, projection='3d')
render_marching_cubes(
    final_scene,
    bounds=(-1, -1, -1),
    size=(4, 4, 3),
    resolution=40,
    ax=ax2,
    color='#06A77D',
    alpha=0.7
)
ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
           c='#F18F01', marker='x', s=200, linewidths=3, label='Targets')
ax2.set_title('Optimized Configuration', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.view_init(elev=20, azim=45)
ax2.legend()

plt.tight_layout()
plt.savefig('examples/output/constrained_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: examples/output/constrained_optimization.png")

# Create optimization animation
print("\nCreating optimization animation...")
os.makedirs('examples/output/tmp_frames', exist_ok=True)

for i, reduced in enumerate(reduced_history):
    # Project to full space
    full_params = base_point + null_space @ reduced
    radii = full_params[-3:]

    # Build scene with these parameters
    r1 = Scalar(float(radii[0]), free=False, name=f'r1_step{i}')
    r2 = Scalar(float(radii[1]), free=False, name=f'r2_step{i}')
    r3 = Scalar(float(radii[2]), free=False, name=f'r3_step{i}')

    s1 = from_point(center1, r1)
    s2 = from_point(center2, r2)
    s3 = from_point(center3, r3)

    step_scene = Union(Union(s1, s2), s3)

    # Render this step
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    render_marching_cubes(
        step_scene,
        bounds=(-1, -1, -1),
        size=(4, 4, 3),
        resolution=40,
        ax=ax,
        color='#06A77D',
        alpha=0.7
    )

    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
               c='#F18F01', marker='x', s=200, linewidths=3, label='Targets')

    loss = float(loss_fn(reduced))
    ax.set_title(f'Step {i}/{len(reduced_history)-1} (loss: {loss:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)
    ax.legend()

    frame_path = f'examples/output/tmp_frames/frame_{i:03d}.png'
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Rendered frame {i+1}/{len(reduced_history)}")

# Create GIF
images = []
for i in range(len(reduced_history)):
    filename = f'examples/output/tmp_frames/frame_{i:03d}.png'
    images.append(imageio.imread(filename))

imageio.mimsave('examples/output/optimization_progress.gif', images, fps=2, loop=0)
print("✓ Saved: examples/output/optimization_progress.gif\n")
