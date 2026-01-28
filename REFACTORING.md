# JaxCAD Refactoring: SDF and Geometry Modules

## Overview

We're reorganizing JaxCAD into two primary modules following **Option C: Geometric Entities as SDF Constructors**.

### Design Philosophy

- **Geometry entities** (Line, Rectangle, etc.) are mathematical definitions and constraint specifications
- **SDF primitives** (Box, Sphere, etc.) are computational distance fields
- Geometry entities can be used to construct/constrain SDFs but remain separate
- Dual rendering: geometry → wireframes (cheap), SDFs → meshes (expensive but accurate)

## Current Progress

### ✅ Completed

1. **Directory Structure Created**
   - `jaxcad/sdf/` - All SDF-related code
   - `jaxcad/geometry/` - All geometric entities and parameters

2. **Moved SDF Components to `sdf/`**
   - `sdf/base.py` (was `sdf.py`)
   - `sdf/primitives/` (was `primitives/`)
     - box, sphere, cylinder, cone, capsule, torus, round_box
   - `sdf/boolean/` (was `boolean/`)
     - union, intersection, difference, xor, smooth operations
   - `sdf/transforms/` (was `transforms/`)
     - affine (translate, rotate, scale)
     - deformations (twist)

3. **Moved Geometry Components to `geometry/`**
   - `geometry/parameters.py` (was `parameters.py`)
     - Parameter, Scalar, Vector, Distance, Angle, Point
   - `geometry/constraints.py` (was `constraints.py`)
     - Placeholder for future constraint system

4. **Created New Geometric Entities**
   - **`geometry/line.py`**
     - Parametric line defined by start/end points
     - Methods: `sample(t)`, `direction()`, `length()`, `midpoint()`, `tangent(t)`
     - Methods: `closest_point(p)`, `distance_to_point(p)`
     - Can be used for: construction, constraints, spatial operations (e.g., Repeat)

   - **`geometry/rectangle.py`**
     - Parametric rectangle in 3D space
     - Defined by: center, width, height, normal, u_axis
     - Methods: `sample(u, v)`, `corner(i)`, `corners()`, `area()`, `perimeter()`
     - Methods: `contains_point(p)`
     - Can be used for: planar constraints, extrusion, surface patterns

5. **Updated Internal Imports**
   - All imports within `sdf/` updated to use `jaxcad.sdf.*` paths
   - `geometry/constraints.py` updated to import from `geometry/parameters.py`

### 🚧 In Progress

6. **`geometry/__init__.py`** (next step)

### ⏳ Pending

7. **Update imports in rest of codebase**
   - `compiler.py`
   - `render.py`
   - All example files
   - All test files

8. **Update top-level `jaxcad/__init__.py`**
   - Maintain backward compatibility
   - Re-export commonly used items from new locations

9. **Run tests to verify refactoring**

## New Directory Structure

```
jaxcad/
├── __init__.py              # Top-level exports (backward compatibility)
├── sdf/                     # SDF (Signed Distance Function) module
│   ├── __init__.py          # ✅ Created - exports all SDF components
│   ├── base.py              # ✅ Moved from sdf.py - SDF base class
│   ├── primitives/          # ✅ Moved - SDF primitive shapes
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── box.py
│   │   ├── sphere.py
│   │   ├── cylinder.py
│   │   ├── cone.py
│   │   ├── capsule.py
│   │   ├── torus.py
│   │   └── round_box.py
│   ├── boolean/             # ✅ Moved - CSG operations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── union.py
│   │   ├── intersection.py
│   │   ├── difference.py
│   │   ├── xor.py
│   │   └── smooth.py
│   └── transforms/          # ✅ Moved - SDF transformations
│       ├── __init__.py
│       ├── base.py
│       ├── affine.py
│       └── deformations.py
├── geometry/                # Geometric entities and parameters
│   ├── __init__.py          # ⏳ To create
│   ├── parameters.py        # ✅ Moved - Parameter, Scalar, Vector
│   ├── constraints.py       # ✅ Moved - Future constraint system
│   ├── line.py              # ✅ Created - Line geometric entity
│   └── rectangle.py         # ✅ Created - Rectangle geometric entity
├── compiler.py              # ⏳ Needs import updates
├── render.py                # ⏳ Needs import updates
└── examples/                # ⏳ Need import updates
```

## New Geometric Entities

### Line

```python
from jaxcad.geometry import Line, Vector

# Basic line
line = Line(start=[0, 0, 0], end=[10, 0, 0])

# With free parameters
p1 = Vector([0, 0, 0], free=True, name='start')
p2 = Vector([10, 0, 0], free=True, name='end')
line = Line(start=p1, end=p2)

# Query operations
point = line.sample(0.5)           # Midpoint
direction = line.direction()        # Unit direction vector
length = line.length()              # 10.0
tangent = line.tangent(0.3)        # Constant for lines
closest = line.closest_point([5, 2, 0])
distance = line.distance_to_point([5, 2, 0])
```

### Rectangle

```python
from jaxcad.geometry import Rectangle

# Basic rectangle in XY plane
rect = Rectangle(center=[0, 0, 0], width=2.0, height=1.0)

# Oriented rectangle
rect = Rectangle(
    center=[0, 0, 0],
    width=2.0,
    height=1.0,
    normal=[1, 1, 1, 0]  # Direction vector (w=0)
)

# Query operations
center_pt = rect.sample(0.5, 0.5)  # Center
corner = rect.corner(0)             # Bottom-left corner
all_corners = rect.corners()        # All 4 corners
area = rect.area()                  # 2.0
in_rect = rect.contains_point([0.5, 0.3, 0])
```

## Future Work (Deferred)

The following features are part of the design but not yet implemented:

### 1. Operations Module (`jaxcad/operations/`)

Spatial operations that consume geometry entities and produce SDFs:

```python
# Repeat primitive along a path
from jaxcad.geometry import Line
from jaxcad.sdf import Sphere
from jaxcad.operations import Repeat

line = Line(start=[0, 0, 0], end=[10, 0, 0])
sphere = Sphere(radius=0.5)
result = Repeat(
    primitive=sphere,
    path=line,
    spacing=1.0,  # or count=10
    mode='union'
)
```

### 2. SDF Construction from Geometry

Primitives with `.from_geometry()` constructors:

```python
from jaxcad.geometry import Line, Rectangle
from jaxcad.sdf import Box, Capsule

# Box from line + thickness
line = Line([0, 0, 0], [5, 0, 0])
box = Box.from_line(line, thickness=0.5)

# Box from center + direction
box = Box.oriented(center=[1, 2, 3], direction=[1, 1, 0], size=[1, 2, 3])

# Capsule from line
capsule = Capsule.from_line(line, radius=0.3)

# Extrude rectangle to box
rect = Rectangle(center=[0, 0, 0], width=2, height=1)
box = Box.extrude(rect, depth=3.0)
```

### 3. Rendering Module Updates

- `render_geometry(entity) -> wireframe` for Line, Rectangle, etc.
- Keep existing `render_sdf(sdf) -> mesh` via marching cubes
- Hybrid rendering: show both construction geometry (wireframes) and final shapes (meshes)

### 4. Constraint System

Full implementation of `geometry/constraints.py`:

```python
from jaxcad.geometry import Point, Line, Distance, Angle, Parallel

# Define points with DOF
p1 = Point([0, 0, 0], free=True)  # 3 DOF
p2 = Point([1, 0, 0], free=True)  # 3 DOF

# Add constraints (reduce DOF)
Distance(p1, p2, value=5.0)       # Reduces DOF by 1
line1 = Line(p1, p2)
line2 = Line([0, 1, 0], [1, 1, 0])
Parallel(line1, line2)            # Reduces DOF by 1

# Constraint solver builds valid parameter space
# Total: 3 + 3 - 1 - 1 = 4 DOF
```

### 5. More Geometric Entities

- `Circle(center, radius, normal)`
- `Plane(point, normal)`
- `Arc(center, radius, start_angle, end_angle, normal)`
- `Polyline(points)` - for complex paths
- `BezierCurve(control_points)` - for smooth curves

## Import Strategy

### For Users (Backward Compatible)

```python
# Old imports still work
from jaxcad import Sphere, Box, Union
from jaxcad.parameters import Vector, Scalar

# New imports (recommended)
from jaxcad.sdf import Sphere, Box, Union
from jaxcad.geometry import Vector, Scalar, Line, Rectangle
```

### Internal Imports

```python
# Within sdf/ module
from jaxcad.sdf.base import SDF
from jaxcad.sdf.primitives import Sphere

# Within geometry/ module
from jaxcad.geometry.parameters import Vector, Scalar

# Cross-module
from jaxcad.sdf import Sphere
from jaxcad.geometry import Line
```

## Benefits of This Architecture

1. **Clear separation of concerns**
   - Geometry = specification/constraint
   - SDF = computation/evaluation

2. **Fits constraint system vision**
   - Geometric entities naturally represent constraints
   - Parameters flow through both geometry and SDFs

3. **Flexible rendering**
   - Geometry entities → wireframes (cheap previews)
   - SDFs → meshes via marching cubes (final geometry)

4. **Extensible**
   - Easy to add new geometric entities without touching SDF code
   - Operations can consume geometry to produce SDFs

5. **Parametric construction**
   - Natural: `Box.from_line(line, thickness)`
   - Clear relationship between geometry and SDFs

## Next Steps

1. ✅ Create `geometry/__init__.py`
2. Update all imports in `compiler.py`, `render.py`
3. Update all examples
4. Update all tests
5. Update top-level `jaxcad/__init__.py` for backward compatibility
6. Run full test suite
7. Commit the refactoring

## Notes

- All file moves done with `git mv` to preserve history
- Used `sed` to bulk-update internal imports in `sdf/` module
- Geometric entities use same Parameter system (Scalar, Vector) as SDFs
- Line and Rectangle include comprehensive methods for querying and sampling
