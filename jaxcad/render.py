"""Rendering utilities for visualizing SDFs."""

from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from jaxcad.sdf import SDF


def render_raymarched(
    sdf: SDF,
    camera_pos: Array = jnp.array([5.0, 5.0, 5.0]),
    look_at: Array = jnp.array([0.0, 0.0, 0.0]),
    resolution: Tuple[int, int] = (200, 200),
    max_steps: int = 64,
    max_dist: float = 20.0,
    eps: float = 1e-3,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """Render SDF using sphere tracing (raymarching) - fully vectorized with JAX.

    Args:
        sdf: The SDF to render
        camera_pos: Camera position
        look_at: Point to look at
        resolution: Image resolution (height, width)
        max_steps: Maximum raymarching steps
        max_dist: Maximum ray distance
        eps: Surface threshold
        ax: Matplotlib axes (creates new if None)
        title: Plot title

    Returns:
        The matplotlib axes object
    """
    import jax

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    height, width = resolution

    # Camera setup
    forward = look_at - camera_pos
    forward = forward / jnp.linalg.norm(forward)
    right = jnp.cross(jnp.array([0.0, 0.0, 1.0]), forward)
    right = right / jnp.linalg.norm(right)
    up = jnp.cross(forward, right)

    # Generate all ray directions (vectorized)
    fov = 0.8
    j_coords = jnp.arange(width)
    i_coords = jnp.arange(height)
    j_grid, i_grid = jnp.meshgrid(j_coords, i_coords)

    u = (j_grid / width - 0.5) * fov
    v = (i_grid / height - 0.5) * fov

    # Ray directions for all pixels (height x width x 3)
    ray_dirs = (forward[None, None, :] +
                u[:, :, None] * right[None, None, :] +
                v[:, :, None] * up[None, None, :])

    # Normalize all rays
    ray_dirs = ray_dirs / jnp.linalg.norm(ray_dirs, axis=2, keepdims=True)

    # Vectorized sphere tracing
    # Flatten to (height*width, 3) for batch evaluation
    flat_rays = ray_dirs.reshape(-1, 3)
    n_rays = flat_rays.shape[0]

    # Initialize ray distances
    t = jnp.zeros(n_rays)

    # Sphere trace all rays in parallel using fori_loop
    def march_step(_, t):
        # Compute positions for all rays
        positions = camera_pos[None, :] + t[:, None] * flat_rays

        # Evaluate SDF for all positions at once using vmap
        distances = jax.vmap(sdf)(positions)

        # March rays forward - rays naturally stop when distance is small
        return t + 0.9*jnp.maximum(jnp.abs(distances), eps * 0.5)

    t = jax.lax.fori_loop(0, max_steps, march_step, t)

    # Reshape back to image
    t = t.reshape(height, width)

    # Compute final distances to determine hits
    final_positions = camera_pos[None, None, :] + t[:, :, None] * ray_dirs
    final_distances = jax.vmap(jax.vmap(sdf))(final_positions)

    # Determine hits
    hit = jnp.abs(final_distances) < eps * 5

    # Compute normals using gradient for hit surfaces
    def compute_normal(pos):
        """Compute surface normal using SDF gradient."""
        grad_fn = jax.grad(lambda p: sdf(p))
        normal = grad_fn(pos)
        norm = jnp.linalg.norm(normal)
        return jnp.where(norm > 1e-6, normal / norm, jnp.array([0., 0., 1.]))

    # Compute normals for all hit positions
    normals = jax.vmap(jax.vmap(compute_normal))(final_positions)

    # Phong shading
    light_dir = jnp.array([0.5, 0.5, -1.0])
    light_dir = light_dir / jnp.linalg.norm(light_dir)

    # Diffuse lighting (Lambert)
    diffuse = jnp.maximum(0.0, jnp.sum(normals * light_dir[None, None, :], axis=2))

    # Specular lighting (Blinn-Phong)
    view_dir = -ray_dirs
    halfway = view_dir + light_dir[None, None, :]
    halfway = halfway / jnp.linalg.norm(halfway, axis=2, keepdims=True)
    specular = jnp.maximum(0.0, jnp.sum(normals * halfway, axis=2)) ** 32

    # Combine lighting: ambient + diffuse + specular
    ambient = 0.2
    image = ambient + 0.6 * diffuse + 0.4 * specular

    # Apply hits mask and background
    image = jnp.where(hit, image, 0.0)

    # Display
    ax.imshow(np.array(image), cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title('Raymarched Render', fontsize=12)

    return ax


def render_marching_cubes(
    sdf: SDF,
    bounds: Tuple[float, float, float] = (-3, -3, -3),
    size: Tuple[float, float, float] = (6, 6, 6),
    resolution: int = 50,
    ax: Optional[plt.Axes] = None,
    color: str = 'cyan',
    alpha: float = 0.7,
    title: Optional[str] = None
) -> plt.Axes:
    """Render SDF using marching cubes to extract mesh.

    Requires scikit-image for marching cubes algorithm.

    Args:
        sdf: The SDF to render
        bounds: Lower corner (x, y, z)
        size: Size in each dimension (dx, dy, dz)
        resolution: Grid resolution per dimension
        ax: Matplotlib 3D axes (creates new if None)
        color: Mesh color
        alpha: Mesh transparency
        title: Plot title

    Returns:
        The matplotlib axes object
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            "render_marching_cubes requires scikit-image. "
            "Install with: pip install scikit-image"
        )

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Create volume - fully vectorized with JAX
    import jax

    x = jnp.linspace(bounds[0], bounds[0] + size[0], resolution)
    y = jnp.linspace(bounds[1], bounds[1] + size[1], resolution)
    z = jnp.linspace(bounds[2], bounds[2] + size[2], resolution)

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    # Stack into points array (resolution^3, 3)
    points = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Evaluate all points at once using vmap
    volume_flat = jax.vmap(sdf)(points)

    # Reshape to 3D volume
    volume = np.array(volume_flat.reshape(resolution, resolution, resolution))

    # Extract mesh using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=(
            size[0] / resolution,
            size[1] / resolution,
            size[2] / resolution
        ))

        # Offset vertices to correct position
        verts[:, 0] += bounds[0]
        verts[:, 1] += bounds[1]
        verts[:, 2] += bounds[2]

        # Create mesh
        mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor='k', linewidths=0.1)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)

        # Set axis limits
        ax.set_xlim(bounds[0], bounds[0] + size[0])
        ax.set_ylim(bounds[1], bounds[1] + size[1])
        ax.set_zlim(bounds[2], bounds[2] + size[2])

    except (ValueError, RuntimeError) as e:
        print(f"Warning: Marching cubes failed - {e}")
        print("The SDF might not have a zero-level surface in the given bounds.")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title('3D Mesh (Marching Cubes)', fontsize=12)

    return ax


# Convenience function
def render(
    sdf: SDF,
    method: str = 'raymarch',
    **kwargs
) -> plt.Axes:
    """Render an SDF using the specified method.

    Args:
        sdf: The SDF to render
        method: Rendering method - 'raymarch' or 'marching_cubes'
        **kwargs: Additional arguments passed to the specific render function

    Returns:
        Matplotlib axes object
    """
    if method == 'raymarch':
        return render_raymarched(sdf, **kwargs)
    elif method == 'marching_cubes':
        return render_marching_cubes(sdf, **kwargs)
    else:
        raise ValueError(f"Unknown render method: {method}. Use 'raymarch' or 'marching_cubes'")
