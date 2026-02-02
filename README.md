# JAXcad

> **⚠️ Experimental Project** - Still in early development, currently collecting ideas and prototyping.

**Differentiable CAD** built on JAX and Signed Distance Functions (SDFs).

JAXcad combines parametric geometry, geometric constraints, and automatic differentiation to enable gradient-based shape optimization with a clean, layered architecture.

## Features

- **Layered Architecture**: Clean separation between geometry, constraints, construction, and compilation
- **Parametric Design**: Define shapes with free and fixed parameters
- **Geometric Constraints**: Distance, angle, parallel, perpendicular constraints with automatic DOF reduction
- **Construction System**: Bridge from geometric primitives to SDFs
- **JAX Integration**: Automatic differentiation for gradient-based optimization
- **3D Rendering**: Marching cubes visualization with matplotlib

## Quick Example

```python
import jax
import jax.numpy as jnp
from jaxcad.geometry.parameters import Vector, Scalar
from jaxcad.constraints import DistanceConstraint, ConstraintGraph
from jaxcad.construction import from_point
from jaxcad.compiler import extract_parameters, compile_to_function

# Layer 1: Geometry - Create parametric spheres
center1 = Vector([0.0, 0.0, 0.0], free=True, name='c1')
center2 = Vector([2.0, 0.0, 0.0], free=True, name='c2')

# Layer 2: Constraints - Fix distance between centers
graph = ConstraintGraph()
graph.add_constraint(DistanceConstraint(center1, center2, distance=2.0))

# Layer 3: Construction - Create SDF from geometry
radius = Scalar(0.5, free=True, name='radius')
sphere1 = from_point(center1, radius)
sphere2 = from_point(center2, radius)
scene = sphere1 | sphere2  # Union

# Layer 4: Compiler - Extract and compile
sdf_fn = compile_to_function(scene)

# Layer 5: Optimization - Use JAX gradients
def loss_fn(r):
    params = {'union_0.sphere_0.radius': r}
    target = jnp.array([0.8, 0.0, 0.0])
    dist = sdf_fn(target, params, {})
    return dist ** 2

# Optimize with automatic differentiation
grad_fn = jax.grad(loss_fn)
current_radius = 0.5
for step in range(50):
    gradient = grad_fn(current_radius)
    current_radius -= 0.1 * gradient

print(f"Optimized radius: {current_radius}")
```

## Architecture

JAXcad is built with a clean layered architecture:

```
┌─────────────────────────────────────┐
│  Layer 1: Geometry                  │  Parametric primitives (Line, Circle, Rectangle)
│  (jaxcad/geometry/)                 │  Parameters: Vector, Scalar (free or fixed)
└────────────┬────────────────────────┘
             │
┌────────────┴────────────────────────┐
│  Layer 2: Constraints               │  DistanceConstraint, AngleConstraint, etc.
│  (jaxcad/constraints/)              │  ConstraintGraph manages DOF reduction
└────────────┬────────────────────────┘
             │
┌────────────┴────────────────────────┐
│  Layer 3: Construction              │  Bridges geometry → SDF
│  (jaxcad/construction/)             │  extrude(), from_line(), from_circle()
└────────────┬────────────────────────┘
             │
┌────────────┴────────────────────────┐
│  Layer 4: Compiler                  │  Parameter extraction and compilation
│  (jaxcad/compiler/)                 │  compile_to_function() → pure JAX
└────────────┬────────────────────────┘
             │
┌────────────┴────────────────────────┐
│  Layer 5: SDF                       │  Primitives, transforms, booleans
│  (jaxcad/sdf/)                      │  Box, Sphere, Cylinder, etc.
└─────────────────────────────────────┘
```

Each layer is independent and can be used standalone!

## Examples

### Example 1: Primitives and Transforms

Create and render 3D scenes with primitives, transforms, and boolean operations:

```python
from jaxcad.sdf.primitives import Sphere, Box, Cylinder
from jaxcad.sdf.boolean import Union, Difference
from jaxcad.render import render_marching_cubes

# Create scene
platform = Box(size=[3, 3, 0.2]).translate([0, 0, -0.5])
sphere = Sphere(radius=0.8).translate([0, 0, 0.3])
pillar = Cylinder(radius=0.15, height=0.6).translate([1.5, 1.5, -0.3])

# Boolean operations
hole = Cylinder(radius=0.4, height=1.0)
sphere_with_hole = Difference(sphere, hole)

# Combine
scene = Union(platform, pillar)
scene = Union(scene, sphere_with_hole)

# Render with marching cubes
render_marching_cubes(scene, resolution=60)
```

Run with: `python examples/01_primitives_and_transforms.py`

### Example 2: End-to-End Optimization

Complete pipeline from parametric geometry to gradient-based optimization:

```python
# 1. Create parametric geometry with constraints
center1 = Vector([0, 0, 0], free=True, name='c1')
center2 = Vector([2, 0, 0], free=True, name='c2')

graph = ConstraintGraph()
graph.add_constraint(DistanceConstraint(center1, center2, distance=2.0))

# 2. Build SDF scene
radius1 = Scalar(0.5, free=True, name='r1')
sphere1 = from_point(center1, radius1)
scene = sphere1 | sphere2 | sphere3

# 3. Compile to pure JAX
sdf_fn = compile_to_function(scene)

# 4. Optimize with JAX gradients
def loss_fn(radii):
    # Evaluate SDF at target points
    return sum_of_squared_distances_to_targets

grad_fn = jax.grad(loss_fn)
for step in range(50):
    radii -= 0.1 * grad_fn(radii)

# Result: Spheres grow to reach targets while maintaining constraints!
```

Run with: `python examples/02_end_to_end_optimization.py`

### Example 3: Layered Construction Demo

Comprehensive demonstration of all layers working together:

```python
# Shows complete workflow:
# - Parametric geometry with free parameters
# - Distance and perpendicular constraints
# - Construction functions (from_line, extrude, from_circle)
# - Constraint-aware parameter extraction
# - Optimization in reduced DOF space
```

Run with: `python examples/layered_construction_demo.py`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jaxcad.git
cd jaxcad

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

Requirements:
- Python 3.10+
- JAX
- NumPy
- Matplotlib
- scikit-image (for marching cubes rendering)

## Testing

Comprehensive test suite with 161 tests covering all layers:

```bash
# Run all tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run specific test suites
pytest tests/geometry/      # Geometry primitives (36 tests)
pytest tests/constraints/   # Constraint system (42 tests)
pytest tests/construction/  # Construction layer (23 tests)
pytest tests/compiler/      # Compiler system (10 tests)
pytest tests/test_integration_e2e.py  # Integration tests (9 tests)
```

## Roadmap

- [x] Layered architecture (geometry, constraints, construction, compiler, SDF)
- [x] Parametric system with free/fixed parameters
- [x] Geometric constraints (distance, angle, parallel, perpendicular)
- [x] Construction functions (extrude, from_line, from_circle, from_point)
- [x] JAX compilation and automatic differentiation
- [x] 3D rendering with marching cubes
- [ ] Transform system (rotate, scale) fully integrated with construction
- [ ] More geometric primitives (Polygon, Bezier curves, etc.)
- [ ] Shader compilation (JAX → StableHLO → GLSL)
- [ ] Real-time GPU rendering
- [ ] Advanced constraints (tangent, coincident, etc.)
- [ ] Mesh export (STL, OBJ)

## Acknowledgments

- Primitive SDFs based on [Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions/)
- Marching cubes algorithm from scikit-image
- Built with JAX for automatic differentiation

## License

MIT License - see LICENSE file for details

Copyright (c) 2025 JAXcad Contributors
