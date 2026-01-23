# JaxCAD

> **⚠️ Experimental Project** - Not a full CAD system. A fun exploration into differentiable geometry modeling with JAX. For now most of the code is vibe coded and very much a work in progress.

Differentiable signed distance functions (SDFs) for shape design and optimization. The desired features are:

- A **fluent API** for building complex 3D geometry that is easy to read and write.
- Fully differentiable using a computational graph transformation mapping a geometry description into functional JAX friendly code. 
- A **parametric system** to mark which parameters are free/fixed for gradient-based optimization.
- A set of constraints that can be composed to express design intent. The degrees of freedom for the parameters should be automatically inferred from the constraints.
- A way to compress designs into simple arithemtic sequences that can be imported and executed in other languages such as GLSL, Rust, Slang.

The dream:

```python
from jaxcad.primitives import Circle, Box
from jaxcad.parametric import Point, Scalar, Line
from jaxcad.constraints import Distance, Parallel

# The @parametric decorator:
# 1. Calls the function to build the compuatation graph
# 2. Extracts all Parameter instances
# 3. Extracts all Constraint instances
# 4. Builds constrained optimization space by reducing dof based on constraints
@jaxcad.parametric
def my_design():
    A = Point([0, 0], free=True) # 3dof
    B = Point([1, 0], free=True) # 3dof
    L1 = Line([0, 1], [1, 1])  # no dof because fixed

    distance = Distance(A, B, 1.0) # reduces dof of each point by 1
    L2 = Line(start=A, end=B)
    parallel = Parallel(L1, L2) # reduces dof of each point by 1

    cylinder = cylinder_from_line(L2, radius=0.1)

    return cylinder

model = my_design()
latent_params = model.init_latent_params()
target_sdf = ...

# Optimization happens in latent space
def loss_fn(latent):
    # Project latent → full params
    full_params = model.project(latent)

    # Evaluate model with full params
    sdf_value = model.eval(full_params, query_point)

    # Add soft constraint penalties for non-geometric constraints
    penalty = soft_constraint_penalty(full_params, model.constraints)

    return (sdf_value - target_sdf) ** 2 + penalty

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(latent_params)

for step in range(100):
    grads = jax.grad(loss_fn)(latent_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    latent_params = optax.apply_updates(latent_params, updates)

# Extract final full parameters
final_params = model.project(latent_params)

```

## What currently works

![Showcase](assets/showcase_geometry.png)

### Quick Start

```python
import jax.numpy as jnp
from jaxcad.primitives import Sphere, Box

# Build shapes with fluent API
sphere = Sphere(radius=1.0).translate([2, 0, 0])
box = Box(size=[1, 1, 1])

# Combine with boolean operators
shape = sphere | box  # Union
shape = sphere & box  # Intersection
shape = sphere - box  # Difference

# Evaluate signed distance at any point
point = jnp.array([0.5, 0.0, 0.0])
distance = shape(point)  # Returns SDF value
```

### Parametric Optimization

```python
import jax
from jaxcad.parametric import parametric

# Define parametric shape
@parametric
def my_shape():
    sphere = Sphere(radius=1.0)
    return sphere.translate([0.0, 0.0, 0.0])

# Optimize to make surface pass through target
params = my_shape.init_params()
target = jnp.array([2.5, 0.0, 0.0])

for _ in range(30):
    grad = jax.grad(lambda p: my_shape(p, target) ** 2)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grad)

# Result: sphere moves to target point!
```

## Parameters and Optimization

Control which values can be optimized during gradient descent:

**Raw values** (fixed by default):
```python
sphere = Sphere(radius=1.0)  # Fixed, won't change during optimization
```

**Free parameters** (can be optimized):
```python
from jaxcad.constraints import Scalar, Point
from jaxcad.parametric import parametric

radius = Scalar(value=1.0, free=True, name='radius')
position = Point(value=[0, 0, 0], free=True, name='pos')

@parametric
def shape():
    return Sphere(radius=radius).translate(position)

# Optimize with JAX gradients
params = shape.init_params()
# params = {'radius': 1.0, 'pos': [0, 0, 0]}
```

**Example: Optimize position, keep size fixed**

```python
import jax
import jax.numpy as jnp
from jaxcad.constraints import Scalar, Point
from jaxcad.primitives import Sphere
from jaxcad.parametric import parametric

# Design intent: sphere size is fixed, but position can be optimized
fixed_radius = Scalar(value=1.0, free=False)      # Cannot change
free_position = Point(value=[0, 0, 0], free=True)   # Can be optimized

@parametric
def constrained_design():
    sphere = Sphere(radius=fixed_radius)
    return sphere.translate(free_position)

# Optimize position to fit a target point, but radius stays 1.0
params = constrained_design.init_params()
target = jnp.array([3.0, 1.0, 0.0])

for _ in range(50):
    grad = jax.grad(lambda p: constrained_design(p, target) ** 2)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grad)

# Result: position optimized to [2.0, 1.0, 0.0], radius remains 1.0!
```

---

## Installation

```bash
uv sync  # or: pip install -e .
```

## Acknowledgments

Primitive SDFs and operations based on [Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions/).

## License

MIT License

Copyright (c) 2025 JaxCAD Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
