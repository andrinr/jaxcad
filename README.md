# JaxCAD

Fully differentiable CAD software built with JAX, inspired by CadQuery. JaxCAD enables gradient-based optimization of 3D geometry by making all CAD operations differentiable.

## Features

### Core Capabilities
- **Differentiable Primitives**: Box, sphere, cylinder with differentiable parameters
- **Differentiable Transformations**: Translate, rotate, scale with full gradient support
- **2D Sketching**: Rectangle, circle, polygon, regular polygon profiles
- **Extrude & Revolve**: Create 3D solids from 2D profiles
- **Advanced Operations**: Loft, sweep, linear/circular arrays
- **Modifications**: Twist, taper, thicken operations
- **Mesh Operations**: Merge, smooth, subdivide faces

### Differentiability
- **Vectorized Operations**: Leverages JAX's `vmap` for batch processing
- **Pure Functions**: All operations are pure functions compatible with JAX autodiff
- **Gradient-Based Optimization**: Optimize shape parameters to meet design objectives
- **Full gradient support** through all operations including complex pipelines

## Installation

### With uv (recommended)
```bash
uv sync
```

### With pip
```bash
pip install -e .
```

## Quick Start

### Basic Primitives
```python
import jax.numpy as jnp
from jaxcad import box, sphere, translate, rotate

# Create a box
center = jnp.array([0., 0., 0.])
size = jnp.array([2., 2., 2.])
my_box = box(center, size)

# Transform it
translated = translate(my_box, jnp.array([1., 0., 0.]))
rotated = rotate(translated, jnp.array([0., 0., 1.]), jnp.pi / 4)

print(f"Vertices shape: {rotated.vertices.shape}")
print(f"Faces shape: {rotated.faces.shape}")
```

### Complex Geometry
```python
from jaxcad import rectangle, circle, extrude, revolve, twist, array_circular

# Extrude a 2D profile
profile = rectangle(jnp.zeros(2), width=2.0, height=1.0)
solid = extrude(profile, height=3.0)

# Apply a twist
twisted = twist(solid, axis=jnp.array([0., 0., 1.]), angle=jnp.pi/2)

# Create circular array
wheel = array_circular(twisted, jnp.array([0., 0., 1.]),
                      jnp.zeros(3), count=6)
```

## Differentiability Example

```python
import jax
import jax.numpy as jnp
from jaxcad import box

def box_volume(size):
    """Compute box volume from mesh."""
    center = jnp.array([0., 0., 0.])
    solid = box(center, size)

    # Compute bounding box volume
    min_coords = jnp.min(solid.vertices, axis=0)
    max_coords = jnp.max(solid.vertices, axis=0)
    dims = max_coords - min_coords
    return jnp.prod(dims)

# Compute gradient of volume with respect to size
size = jnp.array([2., 3., 4.])
grad_fn = jax.grad(box_volume)
gradient = grad_fn(size)

print(f"∂(volume)/∂(size) = {gradient}")  # [12, 8, 6]
```

## Visualization

JaxCAD includes matplotlib-based 3D visualization:

```python
from jaxcad import box, plot_solid
import jax.numpy as jnp

# Create and visualize a box
solid = box(jnp.zeros(3), jnp.ones(3))
plot_solid(solid, color='lightblue', title='My Box')
```

You can also plot multiple solids:

```python
from jaxcad import box, sphere, plot_solids

solid1 = box(jnp.zeros(3), jnp.ones(3))
solid2 = sphere(jnp.array([2., 0., 0.]), 0.5)
plot_solids([solid1, solid2], colors=['red', 'blue'], labels=['Box', 'Sphere'])
```

## Examples

Run the included examples:

```bash
# Basic differentiability examples (gradients, Jacobians, Hessians)
uv run python examples/basic_differentiability.py

# Shape optimization using gradients
uv run python examples/optimization_example.py

# Complex geometry with visualization (extrude, revolve, loft, sweep, arrays, modifications)
uv run python examples/complex_geometry.py
```

## API Reference

### Core Types
- `Solid` - 3D solid with `vertices` (N×3) and `faces` (M×3) arrays
- `Profile2D` - 2D profile with `points` (N×2) and `closed` boolean

### Primitives
- `box(center, size)` - Create a box primitive
- `sphere(center, radius, resolution=16)` - Create a UV sphere
- `cylinder(center, radius, height, resolution=32)` - Create a cylinder

### 2D Sketching
- `rectangle(center, width, height)` - Create rectangular profile
- `circle(center, radius, resolution=32)` - Create circular profile
- `polygon(points, closed=True)` - Create polygon from points
- `regular_polygon(center, radius, n_sides)` - Create regular polygon
- `offset_profile(profile, distance)` - Offset profile inward/outward

### Operations
- `extrude(profile, height)` - Extrude 2D profile along z-axis
- `revolve(profile, angle=2π, resolution=32)` - Revolve profile around axis
- `loft(profiles, heights)` - Loft between multiple profiles
- `sweep(profile, path)` - Sweep profile along 3D path
- `array_linear(solid, direction, count, spacing)` - Create linear array
- `array_circular(solid, axis, center, count)` - Create circular array

### Transformations
- `translate(solid, offset)` - Translate by offset vector
- `rotate(solid, axis, angle, origin=None)` - Rotate around axis
- `scale(solid, factors, origin=None)` - Scale by factors
- `transform(solid, matrix)` - Apply 4x4 transformation matrix

### Modifications
- `twist(solid, axis, angle)` - Apply twist deformation
- `taper(solid, axis, scale_top, scale_bottom=1)` - Apply taper
- `thicken(solid, thickness)` - Thicken surface by offsetting
- `chamfer_vertex(solid, vertex_idx, distance)` - Chamfer vertex
- `fillet_vertex(solid, vertex_idx, radius)` - Fillet vertex

### Boolean & Mesh Operations
- `merge(solid1, solid2)` - Merge two solids
- `smooth_vertices(solid, iterations, factor)` - Laplacian smoothing
- `subdivide_faces(solid)` - Subdivide each face into 4 triangles
- `compute_vertex_normals(solid)` - Compute per-vertex normals

### Visualization
- `plot_solid(solid, color, alpha, title)` - Plot a single solid using matplotlib
- `plot_solids(solids, colors, labels, title)` - Plot multiple solids in the same scene

## Architecture

JaxCAD uses mesh-based B-rep (boundary representation) with:
- Vertices stored as JAX arrays (N, 3)
- Faces stored as integer index arrays (M, 3)
- All operations implemented as pure, differentiable functions
- Full compatibility with JAX transformations (grad, vmap, jit)

## Development

### Linting and Formatting

JaxCAD uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for lint issues
uv run ruff check jaxcad/ tests/ examples/

# Auto-fix lint issues
uv run ruff check jaxcad/ tests/ examples/ --fix

# Format code
uv run ruff format jaxcad/ tests/ examples/
```

### Testing

All operations are thoroughly tested with **65 passing tests**:

```bash
uv run pytest tests/ -v
```

Tests cover:
- Primitive shapes and their gradients
- Transformations and composed operations
- Sketch operations (2D profiles)
- Complex operations (extrude, revolve, loft, sweep)
- Modifications (twist, taper, thicken)
- Arrays and mesh operations
- Gradient correctness (analytical vs numerical)
- Vectorization (vmap, Jacobian, Hessian)

## Future Work

- [x] Sketch-based modeling (2D profiles + extrude/revolve) ✓
- [x] Advanced mesh operations (subdivision, smoothing) ✓
- [x] Matplotlib-based 3D visualization ✓
- [ ] True CSG boolean operations (union, difference, intersection)
- [ ] NURBS curve and surface support
- [ ] Export to standard CAD formats (STEP, STL)
- [ ] JIT compilation for performance
- [ ] More sophisticated filleting/chamfering with new geometry

## License

MIT
