# jaxCAD

Differentiable SDF primitives, transformations, and constraint system built with JAX.

> [!WARNING]
> The API is not stable. Expect breaking changes.

![primitives](examples/ior.png)


---

## Development install

Clone this repo and
```bash
cd jaxcad
uv sync
pre-commit install
```

## Tests

```bash
pytest tests/
```

## Docs

Requires [Quarto](https://quarto.org/docs/get-started/) and the `docs` extras:

```bash
pip install -e ".[docs]"
quartodoc build   # generate API reference from docstrings
quarto preview    # serve locally at localhost:4321
```

---

## SDF example

Build a scene from primitives, boolean ops, and transforms:

```python
import jax.numpy as jnp
from jaxcad.sdf import Sphere, Box, Capsule, Cylinder, Torus, Union, Translate

sphere  = Translate(Sphere(radius=0.6),  offset=jnp.array([-1.0, 0.0, 0.0]))
box     = Translate(Box(size=[0.7, 0.7, 1.0]), offset=jnp.array([0.0, 0.0, 0.8]))
capsule = Translate(Capsule(radius=0.3, height=1.3), offset=jnp.array([1.0, 0.0, 0.0]))

scene = Union((sphere, box, capsule), smoothness=0.1)

# Evaluate the SDF at any point
p = jnp.array([0.5, 0.0, 0.0])
print(scene(p))  # signed distance from p to the surface
```

Every node is a pure JAX function — `jax.grad`, `jax.jit`, and `jax.vmap` work directly on the scene.

---

## Rendering example

Assign materials to primitives and render with the sphere-tracing raymarcher.
`background_color`, `refract_steps`, and `ior` are all new in this release.

```python
import jax.numpy as jnp
from jaxcad.render import raymarch, Material
from jaxcad.sdf.primitives import Sphere
from jaxcad.sdf.boolean import Union
from jaxcad.sdf.transforms import Translate

# Glass sphere (ior=1.5) in front of two coloured spheres
glass = Sphere(
    radius=1.0,
    material=Material(color=[0.92, 0.97, 1.0], roughness=0.05, opacity=0.04, ior=1.5),
)
red   = Translate(Sphere(radius=0.65, material=Material(color=[0.93, 0.26, 0.22])),
                  offset=jnp.array([-1.1, 0.5, -3.0]))
green = Translate(Sphere(radius=0.65, material=Material(color=[0.05, 0.72, 0.50])),
                  offset=jnp.array([ 1.1, -0.5, -3.0]))
scene = Union((glass, red, green), smoothness=0.0)

image = raymarch(
    scene,
    camera_pos=jnp.array([0.0, 0.5, 5.5]),
    resolution=(400, 400),
    background_color=jnp.array([0.07, 0.09, 0.16]),  # dark night sky
    refract_steps=48,   # two-bounce Snell's-law refraction
    aa_samples=2,
)
# image is a (400, 400, 3) float32 numpy array
```

**Material parameters:**

| field | default | meaning |
|-------|---------|---------|
| `color` | `[1,1,1]` | RGB surface colour |
| `roughness` | `0.5` | 0 = mirror, 1 = fully diffuse |
| `metallic` | `0.0` | 0 = dielectric, 1 = metallic specular |
| `opacity` | `1.0` | 0 = fully transparent, 1 = opaque |
| `ior` | `1.0` | index of refraction (1.33 water, 1.5 glass, 2.42 diamond) |

**Key render parameters:**

| parameter | default | meaning |
|-----------|---------|---------|
| `background_color` | `[0,0,0]` | colour for rays that miss all geometry |
| `refract_steps` | `0` | interior march steps; 0 disables refraction |

---

## Constraint example — Riemannian gradient descent

Move a point along a constraint manifold using Riemannian gradient descent: gradient steps stay on the tangent plane and a Newton projection snaps back to the manifold after each step.

```python
import jax
import jax.numpy as jnp
import optax

from jaxcad.constraints import (
    DistanceConstraint, Vector,
    null_space, make_manifold_projection,
)
from jaxcad.extraction import extract_parameters

# Constrain p to lie on the sphere |p| = 2
anchor = Vector(jnp.array([0.0, 0.0, 0.0]))
p      = Vector(jnp.array([2.0, 0.0, 0.0]), free=True, name="p")
DistanceConstraint(anchor, p, distance=2.0)

free_params, _, metadata = extract_parameters(p)
target = jnp.array([1.0, 1.5, 0.0])

def objective(params):
    return jnp.sum((params["p"] - target) ** 2)

value_and_grad = jax.value_and_grad(objective)

def riemannian_grad(params):
    """Project gradient onto the tangent plane at the current point."""
    N = null_space(params, metadata)          # tangent-space basis (relinearized)
    loss, g = value_and_grad(params)
    return loss, N @ (g @ N)                  # Riemannian gradient

# Riemannian GD: tangent-plane steps + manifold projection after each update
optimizer = optax.chain(
    optax.sgd(0.15),
    make_manifold_projection(metadata),       # Newton snap-back onto |p|=2
)

params = free_params
state  = optimizer.init(params)
for step in range(20):
    loss, g = riemannian_grad(params)
    updates, state = optimizer.update(g, state, params)
    params = optax.apply_updates(params, updates)

print(params["p"])  # [1.109, 1.664, 0.] — optimal point on |p|=2 closest to target
```

`null_space` recomputes the constraint Jacobian at the current point each step, so the gradient is always projected onto the correct tangent plane. `make_manifold_projection` chains as a standard optax transform and works with any base optimizer.

---

Inspired by [Fidget](https://www.mattkeeter.com/projects/fidget/) and [Inigo Quilez's distance functions](https://iquilezles.org/articles/distfunctions/).

---

![primitives](examples/thingy.png)

## License

[GNU Affero General Public License v3.0](LICENSE) — free for open source use; commercial use requires a separate license. Contact the authors if you want to use jaxcad in a proprietary product.
