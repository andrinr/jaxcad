# JaxCAD

> **⚠️ Fresh Start** - Rebuilding from scratch with SDF-based CSG approach.

Differentiable CAD operations using Signed Distance Functions (SDFs) and Constructive Solid Geometry (CSG) built with JAX.

## Approach

**SDF-based CSG**: All geometry is represented as signed distance functions, with boolean operations implemented as smooth min/max functions for full differentiability.

### Why SDFs + CSG?

- **Intuitive for engineers**: Familiar CAD workflow (primitives + boolean ops)
- **Fully differentiable**: Smooth operations enable gradient flow
- **General**: Can represent most mechanical parts
- **Composable**: Build complex shapes from simple primitives

## Planned Features

**Primitives**: sphere, box, cylinder, cone (as SDF functions)
**Boolean Ops**: union, difference, intersection (smooth versions)
**Transformations**: translate, rotate, scale
**Advanced Ops**: fillet, chamfer (smooth blending)
**Parametric Features**: holes, slots, patterns
**Mesh Export**: Convert SDF to mesh for visualization/manufacturing

## Installation

```bash
uv sync  # or: pip install -e .
```

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
