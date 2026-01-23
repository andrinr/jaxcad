# JaxCAD

> **⚠️ Experimental Project** - Still in early development, currently mostly collecting ideas and prototyping.

![Showcase](assets/showcase_geometry.png)

Differentiable signed distance functions (SDFs) for shape design and optimization. The goals are:

- A **fluent API** for building complex 3D geometry that is easy to read and write.
- Differentiability is enabled with a JIT compiler that translates a geometry description into functional JAX friendly code.
- A **parametric system** to mark which parameters are free/fixed for gradient-based optimization.
- A set of constraints that can be composed to express design intent. The degrees of freedom for the parameters should be automatically inferred from the constraints.
- **Shader compilation**: Leverage JAX's XLA/StableHLO IR to transpile SDF code into GLSL/Slang for real-time GPU rendering.

```
Design Code (Python/JAX)
        ↓
   Functional IR
   (JAX pytree)
        ↓
   StableHLO IR
   (SSA form)
        ↓
     ┌──┴──┐
     ↓     ↓
   XLA   GLSL/Slang
(CPU/GPU) (Shaders)
```

## The Optimization Dream

```python
from jaxcad.primitives import Cylinder
from jaxcad.constraints import Point, Scalar, Line, Distance, Parallel

# Define parametric constraints
A = Point([0, 0], free=True)  # 2 DOF
B = Point([1, 0], free=True)  # 2 DOF
L1 = Line([0, 1], [1, 1])     # Fixed, no DOF

# Constraints automatically reduce DOF
Distance(A, B, 1.0)           # Reduces total DOF by 1
L2 = Line(start=A, end=B)
Parallel(L1, L2)              # Reduces total DOF by 1

# Build geometry - automatically tracks constraint dependencies
cylinder = Cylinder.from_line(L2, radius=0.1)

# Extract constrained parameter space (2 + 2 - 1 - 1 = 2 DOF)
latent_params = cylinder.init_latent_params()
target_sdf = ...

# Optimization happens in latent space
def loss_fn(latent):
    # Project latent → full params (satisfies geometric constraints)
    full_params = cylinder.project(latent)

    # Evaluate SDF with full params
    sdf_value = cylinder.eval(full_params, query_point)

    # Add soft constraint penalties for non-geometric constraints
    penalty = cylinder.constraint_penalty(full_params)

    return (sdf_value - target_sdf) ** 2 + penalty

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(latent_params)

for step in range(100):
    grads = jax.grad(loss_fn)(latent_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    latent_params = optax.apply_updates(latent_params, updates)

# Extract final full parameters
final_params = cylinder.project(latent_params)

```

## The Compilation Dream

Compile your JAX SDF geometry to real-time GPU shaders for interactive rendering:

```python
from jaxcad.primitives import Sphere, Box
from jaxcad.compiler import compile_to_glsl

# Build geometry with fluent API
sphere = Sphere(radius=1.0).translate([2, 0, 0])
box = Box(size=[1, 1, 1])
shape = sphere | box

# Compile to GLSL shader via JAX XLA/StableHLO IR
glsl_code = compile_to_glsl(shape)
# Output: GLSL fragment shader with sdf(vec3 p) function

# Render in real-time using the generated shader
from jaxcad.render import render_shader

image = render_shader(
    glsl_code,
    resolution=(800, 600),
    camera_pos=[5, 5, 5],
    camera_target=[0, 0, 0]
)
```

## What currently works


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
