# JaxCAD Architecture

## Module Organization

### `parameters.py` - Optimizable Parameters
Contains parameter types for gradient-based optimization:
- `Parameter` - Base class for all optimizable parameters
- `Scalar` - Single values (radius, distance, angle, scale, etc.)
- `Vector` - 3D/4D vectors with homogeneous coordinates [x, y, z, w]
  - 3D input `[x, y, z]` → automatically extends to `[x, y, z, 1]` (point)
  - 4D input `[x, y, z, 0]` → direction vector (unaffected by translation)
  - Properties: `.xyz`, `.w`, `.is_point`, `.is_direction`
- Aliases: `Distance`, `Angle`, `Point` (backwards compatibility)

All parameters can be marked as `free=True` for optimization or `free=False` (default) to keep fixed.

### `constraints.py` - Constraint System (FUTURE)
Reserved for future geometric constraint system:
- Distance constraints between points
- Angle constraints between lines
- Parallel/perpendicular relationships
- Tangency constraints
- Automatic DOF reduction

Currently re-exports parameter types for backwards compatibility.

### `parametric.py` - Decorator for Compilation
Contains the `@parametric` decorator that:
- Extracts SDF computation graph
- Identifies free parameters
- Creates differentiable evaluation function
- Enables JAX gradient-based optimization

### `compiler/` - Graph Compilation
- `graph.py` - SDF expression graph extraction and representation
- `optimize.py` - Graph optimization passes
- Future: Shader compilation to GLSL/Slang via StableHLO

## Design Rationale

### Why separate parameters from constraints?

**Parameters** (`parameters.py`):
- Low-level optimization primitives
- Used directly in SDF primitives and transforms
- Simple: mark values as free/fixed
- No dependencies between parameters

**Constraints** (`constraints.py`, future):
- High-level geometric relationships
- Automatically reduce degrees of freedom
- Express design intent (e.g., "these lines are parallel")
- Builds parameter space from constraint graph

### Why homogeneous coordinates for Vector?

1. **Unified representation** - Points (w=1) and directions (w=0) in same type
2. **GPU compatibility** - Standard in graphics pipelines and shaders
3. **Efficient transforms** - Translation encoded in matrix multiplication
4. **Easy composition** - Chain transforms via 4×4 matrix multiplication
5. **Future-proof** - Ready for shader compilation target

The `.xyz` property extracts 3D coordinates when needed for SDF evaluation.

## Usage Patterns

### Basic optimization
```python
from jaxcad.parameters import Scalar, Vector
from jaxcad.parametric import parametric
from jaxcad.primitives import Sphere

# Define free parameters
radius = Scalar(value=1.0, free=True, name='radius')
position = Vector(value=[0, 0, 0], free=True, name='pos')

@parametric
def my_shape():
    return Sphere(radius=radius).translate(position)

# Optimize with JAX
params = my_shape.init_params()
value = my_shape(params, point)
grad = jax.grad(lambda p: my_shape(p, point) ** 2)
```

### Individual transforms (semantic parameters)
```python
from jaxcad.transforms import Translate, Rotate, Scale

# 3 DOF for translation
sphere.translate(Vector([1, 2, 3], free=True))

# 1 DOF for rotation angle
sphere.rotate('z', Scalar(0, free=True))

# 1 DOF for uniform scale
sphere.scale(Scalar(1, free=True))
```

### Future constraint-based (automatic DOF reduction)
```python
# Future API
from jaxcad.constraints import Point, Line, Distance, Parallel

A = Point([0, 0], free=True)  # 2 DOF
B = Point([1, 0], free=True)  # 2 DOF
Distance(A, B, 1.0)           # -1 DOF
L1 = Line([0, 1], [1, 1])
L2 = Line(start=A, end=B)
Parallel(L1, L2)              # -1 DOF
# Total: 2 + 2 - 1 - 1 = 2 DOF
```

## File Locations

```
jaxcad/
├── parameters.py          # Optimizable parameter types
├── constraints.py         # Future: constraint system (currently re-exports parameters)
├── parametric.py          # @parametric decorator
├── compiler/
│   ├── graph.py          # SDF graph extraction
│   └── optimize.py       # Graph optimization passes
├── primitives/           # SDF primitives (Sphere, Box, etc.)
├── transforms/           # Transforms (Translate, Rotate, Scale, etc.)
└── sdf.py               # Base SDF class
```

## Migration Notes

All existing code continues to work thanks to re-exports in `constraints.py`:
```python
# Old import (still works)
from jaxcad.constraints import Vector, Scalar

# New import (preferred)
from jaxcad.parameters import Vector, Scalar
```

Internal compiler code updated to use `jaxcad.parameters` directly.
