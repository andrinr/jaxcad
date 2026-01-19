"""Visualization utilities for JaxCAD using Matplotlib."""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from jaxcad.core import Solid


def plot_solid(
    solid: Solid,
    ax=None,
    color="lightblue",
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
    show=True,
    title="JaxCAD Solid",
):
    """Plot a single solid using Matplotlib 3D.

    Args:
        solid: JaxCAD solid to plot
        ax: Matplotlib 3D axis (creates new figure if None)
        color: Color of the mesh surface
        alpha: Transparency (0=transparent, 1=opaque)
        edgecolor: Color of the edges
        linewidth: Width of the edges
        show: Whether to call plt.show() at the end
        title: Plot title

    Returns:
        Matplotlib axis object

    Example:
        >>> from jaxcad import box
        >>> import jax.numpy as jnp
        >>> solid = box(jnp.zeros(3), jnp.ones(3))
        >>> plot_solid(solid, color='red')
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization. Install with: uv sync")

    # Create figure and axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Convert JAX arrays to numpy
    vertices = np.array(solid.vertices)
    faces = np.array(solid.faces)

    # Create triangular mesh
    triangles = vertices[faces]

    # Create 3D polygon collection
    mesh = Poly3DCollection(triangles, alpha=alpha, facecolor=color, edgecolor=edgecolor)
    mesh.set_linewidth(linewidth)
    ax.add_collection3d(mesh)

    # Set axis limits based on vertices
    margin = 0.1
    x_min, y_min, z_min = vertices.min(axis=0) - margin
    x_max, y_max, z_max = vertices.max(axis=0) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Equal aspect ratio
    ax.set_box_aspect(
        [x_max - x_min, y_max - y_min, z_max - z_min]
    )

    if show:
        plt.show()

    return ax


def plot_solids(
    solids: list[Solid],
    colors: list[str] = None,
    labels: list[str] = None,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
    title="JaxCAD Solids",
    show=True,
):
    """Plot multiple solids in the same scene.

    Args:
        solids: List of JaxCAD solids to plot
        colors: List of colors for each solid (default: automatic)
        labels: List of labels for each solid
        alpha: Transparency (0=transparent, 1=opaque)
        edgecolor: Color of the edges
        linewidth: Width of the edges
        title: Plot title
        show: Whether to call plt.show() at the end

    Returns:
        Matplotlib axis object

    Example:
        >>> from jaxcad import box, sphere
        >>> import jax.numpy as jnp
        >>> s1 = box(jnp.zeros(3), jnp.ones(3))
        >>> s2 = sphere(jnp.array([2., 0., 0.]), 0.5)
        >>> plot_solids([s1, s2], colors=['red', 'blue'])
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization. Install with: uv sync")

    if colors is None:
        # Default color palette
        default_colors = [
            "lightblue",
            "lightcoral",
            "lightgreen",
            "lightyellow",
            "lightpink",
            "lightcyan",
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(solids))]

    if labels is None:
        labels = [f"Solid {i+1}" for i in range(len(solids))]

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Collect all vertices for axis limits
    all_vertices = []

    # Plot each solid
    for i, solid in enumerate(solids):
        vertices = np.array(solid.vertices)
        faces = np.array(solid.faces)
        all_vertices.append(vertices)

        # Create triangular mesh
        triangles = vertices[faces]

        # Create 3D polygon collection
        mesh = Poly3DCollection(
            triangles, alpha=alpha, facecolor=colors[i], edgecolor=edgecolor, label=labels[i]
        )
        mesh.set_linewidth(linewidth)
        ax.add_collection3d(mesh)

    # Set axis limits based on all vertices
    all_vertices = np.concatenate(all_vertices, axis=0)
    margin = 0.1
    x_min, y_min, z_min = all_vertices.min(axis=0) - margin
    x_max, y_max, z_max = all_vertices.max(axis=0) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Equal aspect ratio
    ax.set_box_aspect(
        [x_max - x_min, y_max - y_min, z_max - z_min]
    )

    # Add legend
    ax.legend()

    if show:
        plt.show()

    return ax
