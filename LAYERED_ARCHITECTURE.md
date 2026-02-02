# JaxCAD Layered Architecture

## Overview

JaxCAD now features a clean layered architecture where each conceptual layer is its own submodule. This enables flexible, constraint-driven CAD with clear separation of concerns.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 1: Geometry                        │
│  Parametric geometric primitives (Line, Rectangle, Circle) │
│         jaxcad/geometry/primitives/                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Layer 2: Constraints                       │
│   Geometric constraints reduce degrees of freedom (DOF)    │
│         jaxcad/constraints/                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Layer 3: Construction                       │
│    Bridge layer: Geometry primitives → SDF primitives      │
│         jaxcad/construction/                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: Compiler                        │
│  Extract parameters, integrate constraints, compile to JAX │
│         jaxcad/compiler/                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Layer 5: SDF                           │
│   SDF primitives, boolean operations, transformations      │
│         jaxcad/sdf/                                         │
└─────────────────────────────────────────────────────────────┘
```

## Layer Details

### Layer 1: Geometry (`jaxcad/geometry/`)

**Purpose**: Parametric geometric entities with free/fixed parameters

**Contents**:
- `parameters.py` - `Vector`, `Scalar`, `Parameter` base classes
- `primitives/` - Geometric primitives
  - `line.py` - Line segment in 3D
  - `rectangle.py` - Rectangle in 3D space
  - `circle.py` - Circle in 3D space

**Example**:
```python
from jaxcad.geometry import Vector, Scalar
from jaxcad.geometry.primitives import Line, Rectangle, Circle

p1 = Vector([0, 0, 0], free=True, name='p1')
p2 = Vector([2, 0, 0], free=True, name='p2')

line = Line(start=p1, end=p2)
rect = Rectangle(center=p1, width=2.0, height=1.0)
circle = Circle(center=p1, radius=Scalar(1.0, free=True, name='r'))
```

**Key Principles**:
- Geometric entities are **parameter containers**
- Parameters can be `free=True` (optimizable) or `free=False` (fixed)
- No SDF dependencies - pure geometric definitions
- Can be used for visualization, constraints, or SDF construction

### Layer 2: Constraints (`jaxcad/constraints/`)

**Purpose**: Define geometric relationships and reduce DOF

**Contents**:
- `Constraint` base class
- `DistanceConstraint` - Fix distance between two points
- `AngleConstraint` - Fix angle between two vectors
- `ParallelConstraint` - Force vectors parallel
- `PerpendicularConstraint` - Force vectors perpendicular
- `ConstraintGraph` - Manages constraints and computes reduced DOF

**Example**:
```python
from jaxcad.constraints import ConstraintGraph, DistanceConstraint

graph = ConstraintGraph()
graph.add_constraint(DistanceConstraint(p1, p2, distance=2.0))

# Extract reduced DOF (6 DOF → 5 DOF with 1 constraint)
reduced_params, null_space = graph.extract_free_dof([p1, p2])
```

**Key Principles**:
- Each constraint reduces DOF by the number of equations it introduces
- Uses **null space projection** for DOF reduction
- Constraints are linearized (exact for linear constraints, approximate for nonlinear)
- Pure constraint logic - no SDF dependencies

### Layer 3: Construction (`jaxcad/construction/`)

**Purpose**: Bridge from geometry primitives to SDF primitives

**Contents**:
- `extrude(rectangle, depth)` → Box
- `from_line(line, radius)` → Capsule
- `from_circle(circle, height)` → Cylinder
- `from_point(point, radius)` → Sphere

**Example**:
```python
from jaxcad.construction import extrude, from_line, from_circle

# Geometry → SDF conversion
capsule = from_line(line, radius=0.5)
box = extrude(rect, depth=3.0)
cylinder = from_circle(circle, height=5.0)
```

**Key Principles**:
- **Preserves parameter references** - SDF shares parameters with geometry
- Changes to geometric parameters automatically affect SDF
- Each function stores `_source_geometry` for reference
- Acts as the bridge between parametric geometry and implicit SDFs

### Layer 4: Compiler (`jaxcad/compiler/`)

**Purpose**: Extract parameters, integrate constraints, compile to pure JAX functions

**Contents**:
- `extract_parameters(sdf)` - Walk SDF tree, extract free/fixed params
- `extract_parameters_with_constraints(sdf, graph)` - **NEW**: Constraint-aware extraction
- `compile_to_function(sdf)` - Compile to pure JAX function

**Example**:
```python
from jaxcad.compiler import extract_parameters_with_constraints

# Extract constrained parameters for optimization
reduced, null_space, base, params = extract_parameters_with_constraints(
    capsule,  # SDF tree
    graph     # ConstraintGraph
)

# Optimize in reduced space
def loss_fn(reduced_params):
    full = base + null_space @ reduced_params
    # ... use full params for evaluation ...

grad_fn = jax.grad(loss_fn)
```

**Key Principles**:
- Walks SDF tree recursively to extract all parameters
- Integrates ConstraintGraph for DOF reduction
- Returns reduced parameter space ready for optimization
- Compiles to pure functions for JAX tracing

### Layer 5: SDF (`jaxcad/sdf/`)

**Purpose**: Implicit surface representation, boolean operations, rendering

**Contents**:
- `primitives/` - Box, Sphere, Cylinder, Capsule, Cone, Torus, etc.
- `boolean/` - Union, Intersection, Difference, XOR, Smooth operations
- `transforms/` - Translate, Rotate, Scale, Twist, etc.

**Example**:
```python
# Boolean operations
assembly = capsule1 | capsule2  # Union
difference = box - sphere       # Difference

# Transformations
from jaxcad.sdf.transforms import Translate
translated = Translate(capsule, offset=[1, 0, 0])
```

**Key Principles**:
- Each SDF is a **thin wrapper** around a pure function
- Supports fluent API (method chaining)
- Operator overloading (|, &, -)
- Pure functions are the source of truth for computation

## Complete Workflow Example

```python
# 1. Define parametric geometry
from jaxcad.geometry import Vector, Scalar
from jaxcad.geometry.primitives import Line

p1 = Vector([0, 0, 0], free=True, name='p1')
p2 = Vector([3, 0, 0], free=True, name='p2')

line = Line(start=p1, end=p2)

# 2. Add constraints
from jaxcad.constraints import ConstraintGraph, DistanceConstraint

graph = ConstraintGraph()
graph.add_constraint(DistanceConstraint(p1, p2, distance=2.0))

# 3. Construct SDF from geometry
from jaxcad.construction import from_line

capsule = from_line(line, radius=0.5)

# 4. Extract constrained parameters
from jaxcad.compiler import extract_parameters_with_constraints

reduced, null_space, base, params = extract_parameters_with_constraints(
    capsule,
    graph
)

# 5. Optimize in reduced space
import jax

def loss_fn(reduced_params):
    full = base + null_space @ reduced_params
    p1_new = full[:3]
    p2_new = full[3:6]
    # Minimize z-coordinates
    return p1_new[2] + p2_new[2]

grad_fn = jax.grad(loss_fn)
current_reduced = reduced

for _ in range(50):
    grad = grad_fn(current_reduced)
    current_reduced = current_reduced - 0.1 * grad

# Constraint is automatically preserved during optimization!
```

## Directory Structure

```
jaxcad/
├── geometry/                    # Layer 1: Parametric geometry
│   ├── __init__.py
│   ├── parameters.py           # Vector, Scalar, Parameter
│   └── primitives/             # Geometric entities
│       ├── __init__.py
│       ├── line.py
│       ├── rectangle.py
│       └── circle.py
│
├── constraints/                # Layer 2: Constraint system
│   └── __init__.py            # DistanceConstraint, AngleConstraint, etc.
│
├── construction/               # Layer 3: Geometry → SDF bridge
│   └── __init__.py            # extrude(), from_line(), from_circle(), etc.
│
├── compiler/                   # Layer 4: Compilation
│   └── __init__.py            # extract_parameters(), compile_to_function()
│
└── sdf/                        # Layer 5: SDF primitives
    ├── base.py
    ├── primitives/            # Box, Sphere, Cylinder, etc.
    ├── boolean/               # Union, Intersection, Difference
    └── transforms/            # Translate, Rotate, Scale
```

## Design Principles

1. **Layered Independence**: Each layer can be used standalone
2. **No Backward Compatibility**: Clean break, modern API
3. **Parameter Sharing**: Construction layer preserves parameter references
4. **Constraint Integration**: Compiler integrates constraints seamlessly
5. **Pure Functions**: SDF layer uses pure functions for JAX tracing
6. **Null Space Projection**: Constraints reduce DOF via linearized null space

## Benefits

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Flexibility**: Use any layer independently
3. **Constraint-Driven**: Natural integration of geometric constraints
4. **Optimization-Ready**: Automatic DOF reduction for efficient optimization
5. **JAX-Native**: Pure functions work seamlessly with JAX autodiff
6. **Extensible**: Easy to add new geometric primitives, constraints, or construction methods

## Examples

See:
- `examples/layered_construction_demo.py` - Complete workflow examples
- `examples/constraint_system_demo.py` - Constraint system details
- `tests/test_constraint_behavior.py` - Comprehensive tests

## Future Extensions

Potential additions to each layer:

**Geometry Layer**:
- Plane, Arc, Ellipse, BezierCurve, Polyline

**Constraints Layer**:
- Tangency, Coincidence, Symmetry, Equality constraints
- Nonlinear constraint solvers (SQP, Augmented Lagrangian)

**Construction Layer**:
- `revolve(profile, axis, angle)` - Revolution SDFs
- `sweep(profile, path)` - Sweep along path
- `loft(profiles)` - Loft between profiles

**Compiler Layer**:
- Automatic differentiation of constraint residuals
- Constraint satisfaction projection

**Model Layer** (future):
- `ConstrainedModel` class for end-to-end workflow
- Automatic rendering integration
- Export to mesh formats

---

**Date**: 2026-02-02
**Version**: 1.0
**Author**: JaxCAD Team
