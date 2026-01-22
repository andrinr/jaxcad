"""Tests for SDF compiler and optimization."""

import jax.numpy as jnp

from jaxcad.compiler import SDFGraph, compile_sdf, optimize_graph
from jaxcad.compiler.graph import extract_graph, OpType
from jaxcad.compiler.optimize import estimate_complexity
from jaxcad.primitives import Box, Sphere
from jaxcad.transforms import Translate


def test_extract_graph_primitive():
    """Test graph extraction from a primitive."""
    sphere = Sphere(radius=1.0)
    graph = extract_graph(sphere)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].op_type == OpType.PRIMITIVE


def test_extract_graph_boolean():
    """Test graph extraction from boolean operations."""
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))
    union = sphere | box

    graph = extract_graph(union)

    # Should have 3 nodes: sphere, box, union
    assert len(graph.nodes) == 3
    assert graph.nodes[-1].op_type == OpType.UNION
    assert len(graph.nodes[-1].children) == 2


def test_extract_graph_transform():
    """Test graph extraction from transforms."""
    sphere = Sphere(radius=1.0)
    translated = sphere.translate(jnp.array([1.0, 0.0, 0.0]))

    graph = extract_graph(translated)

    assert len(graph.nodes) == 2
    assert graph.nodes[-1].op_type == OpType.TRANSLATE
    assert jnp.allclose(graph.nodes[-1].params["offset"], jnp.array([1.0, 0.0, 0.0]))


def test_visualize_graph():
    """Test graph visualization."""
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))
    union = (sphere.translate(jnp.array([1.0, 0.0, 0.0])) | box)

    graph = extract_graph(union)
    viz = graph.visualize()

    assert "union" in viz
    assert "translate" in viz
    assert "Sphere" in viz
    assert "Box" in viz
    print("\nGraph visualization:")
    print(viz)


def test_fuse_transforms():
    """Test transform fusion optimization."""
    sphere = Sphere(radius=1.0)

    # Chain two translations
    translated = (
        sphere
        .translate(jnp.array([1.0, 0.0, 0.0]))
        .translate(jnp.array([0.5, 0.0, 0.0]))
    )

    graph = extract_graph(translated)
    print(f"\nBefore optimization: {len(graph.nodes)} nodes")

    optimized = optimize_graph(graph)
    print(f"After optimization: {len(optimized.nodes)} nodes")

    # Should fuse two translates into one
    # Original: sphere + translate1 + translate2 = 3 nodes
    # Optimized: sphere + translate(combined) = 2 nodes
    assert len(optimized.nodes) < len(graph.nodes)


def test_eliminate_identity():
    """Test identity transform elimination."""
    sphere = Sphere(radius=1.0)

    # Add identity transforms (should be eliminated)
    transformed = (
        sphere
        .translate(jnp.array([0.0, 0.0, 0.0]))  # Identity
        .scale(1.0)  # Identity
    )

    graph = extract_graph(transformed)
    optimized = optimize_graph(graph)

    # Should eliminate both identity transforms
    # Original: sphere + translate + scale = 3 nodes
    # Optimized: sphere = 1 node
    assert len(optimized.nodes) < len(graph.nodes)
    print(f"\nEliminated {len(graph.nodes) - len(optimized.nodes)} identity transforms")


def test_complexity_estimation():
    """Test computational complexity estimation."""
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))

    # Complex expression
    complex_sdf = (
        (sphere.translate(jnp.array([1.0, 0.0, 0.0])) | box)
        .rotate('z', jnp.pi/4)
        .twist('z', 1.0)
    )

    graph = extract_graph(complex_sdf)
    complexity = estimate_complexity(graph)

    print(f"\nComplexity metrics:")
    print(f"  Total nodes: {complexity['total_nodes']}")
    print(f"  Depth: {complexity['depth']}")
    print(f"  Estimated FLOPs: {complexity['estimated_flops']}")

    assert complexity['total_nodes'] > 0
    assert complexity['depth'] > 0
    assert complexity['estimated_flops'] > 0


def test_compile_sdf_basic():
    """Test basic SDF compilation."""
    sphere = Sphere(radius=1.0)

    # Compile with optimization and JIT
    compiled = compile_sdf(sphere, optimize=True, jit=True)

    # Test that it still works
    result = compiled(jnp.array([2.0, 0.0, 0.0]))
    expected = sphere(jnp.array([2.0, 0.0, 0.0]))

    assert jnp.allclose(result, expected)


def test_compile_sdf_optimized():
    """Test that compilation with optimization produces correct results."""
    sphere = Sphere(radius=1.0)

    # Create expression with redundant operations
    sdf = (
        sphere
        .translate(jnp.array([1.0, 0.0, 0.0]))
        .translate(jnp.array([1.0, 0.0, 0.0]))
        .scale(1.0)  # Identity
    )

    # Compile with and without optimization
    compiled_opt = compile_sdf(sdf, optimize=True, jit=False)
    compiled_no_opt = compile_sdf(sdf, optimize=False, jit=False)

    # Both should give same results
    point = jnp.array([3.0, 0.0, 0.0])
    result_opt = compiled_opt(point)
    result_no_opt = compiled_no_opt(point)

    assert jnp.allclose(result_opt, result_no_opt)
    print(f"\nOptimized result: {result_opt}")
    print(f"Unoptimized result: {result_no_opt}")
