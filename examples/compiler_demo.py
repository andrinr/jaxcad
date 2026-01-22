"""Example: SDF compiler and optimization."""

import jax.numpy as jnp
import time

from jaxcad.compiler import compile_sdf
from jaxcad.compiler.graph import extract_graph
from jaxcad.compiler.optimize import estimate_complexity, optimize_graph
from jaxcad.primitives import Box, Cylinder, Sphere


def demo_graph_visualization():
    """Demonstrate graph visualization."""
    print("=" * 60)
    print("GRAPH VISUALIZATION")
    print("=" * 60)

    # Create a complex SDF expression
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([0.8, 0.8, 0.8]))
    cylinder = Cylinder(radius=0.5, height=2.0)

    complex_sdf = (
        (sphere.translate(jnp.array([1.0, 0.0, 0.0])) | box)
        - cylinder.rotate('y', jnp.pi/4)
    ).twist('z', 1.5)

    # Extract and visualize graph
    graph = extract_graph(complex_sdf)
    print("\nExpression graph:")
    print(graph.visualize())

    # Show complexity metrics
    complexity = estimate_complexity(graph)
    print(f"\nComplexity metrics:")
    print(f"  Total nodes: {complexity['total_nodes']}")
    print(f"  Graph depth: {complexity['depth']}")
    print(f"  Estimated FLOPs per evaluation: {complexity['estimated_flops']}")
    print(f"  Operations: {complexity['operations']}")


def demo_optimization():
    """Demonstrate graph optimization."""
    print("\n" + "=" * 60)
    print("GRAPH OPTIMIZATION")
    print("=" * 60)

    sphere = Sphere(radius=1.0)

    # Create expression with redundant operations
    unoptimized = (
        sphere
        .translate(jnp.array([0.5, 0.0, 0.0]))  # Translate 1
        .translate(jnp.array([0.5, 0.0, 0.0]))  # Translate 2 (should fuse)
        .translate(jnp.array([0.0, 0.0, 0.0]))  # Identity (should eliminate)
        .rotate('z', 0.0)                         # Identity (should eliminate)
    )

    # Extract graph before optimization
    graph_before = extract_graph(unoptimized)
    complexity_before = estimate_complexity(graph_before)

    print("\nBefore optimization:")
    print(f"  Nodes: {complexity_before['total_nodes']}")
    print(f"  Estimated FLOPs: {complexity_before['estimated_flops']}")
    print("\nGraph structure:")
    print(graph_before.visualize())

    # Optimize
    graph_after = optimize_graph(graph_before)
    complexity_after = estimate_complexity(graph_after)

    print("\nAfter optimization:")
    print(f"  Nodes: {complexity_after['total_nodes']}")
    print(f"  Estimated FLOPs: {complexity_after['estimated_flops']}")
    print("\nOptimized graph structure:")
    print(graph_after.visualize())

    # Show improvements
    node_reduction = complexity_before['total_nodes'] - complexity_after['total_nodes']
    flop_reduction = complexity_before['estimated_flops'] - complexity_after['estimated_flops']

    print(f"\nOptimization results:")
    print(f"  Nodes eliminated: {node_reduction}")
    print(f"  FLOP reduction: {flop_reduction} ({flop_reduction/complexity_before['estimated_flops']*100:.1f}%)")


def demo_compilation_performance():
    """Demonstrate compilation and performance."""
    print("\n" + "=" * 60)
    print("COMPILATION & PERFORMANCE")
    print("=" * 60)

    # Create a moderately complex SDF
    sphere = Sphere(radius=1.0)
    box = Box(size=jnp.array([1.0, 1.0, 1.0]))

    sdf = (
        sphere.translate(jnp.array([1.0, 0.0, 0.0]))
        | box.rotate('z', jnp.pi/4)
    ).twist('z', 1.0)

    # Compile with different settings
    print("\nCompiling SDF with different settings...")

    # No optimization, no JIT
    compiled_none = compile_sdf(sdf, optimize=False, jit=False)

    # With optimization, no JIT
    compiled_opt = compile_sdf(sdf, optimize=True, jit=False)

    # With optimization and JIT
    compiled_opt_jit = compile_sdf(sdf, optimize=True, jit=True)

    # Test points
    test_points = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
    ])

    # Warm up JIT
    _ = compiled_opt_jit(test_points[0])

    # Benchmark
    num_iterations = 1000

    # Unoptimized, no JIT
    start = time.time()
    for _ in range(num_iterations):
        for point in test_points:
            _ = compiled_none(point)
    time_none = time.time() - start

    # Optimized, no JIT
    start = time.time()
    for _ in range(num_iterations):
        for point in test_points:
            _ = compiled_opt(point)
    time_opt = time.time() - start

    # Optimized + JIT
    start = time.time()
    for _ in range(num_iterations):
        for point in test_points:
            _ = compiled_opt_jit(point)
    time_opt_jit = time.time() - start

    print(f"\nPerformance ({num_iterations} iterations x {len(test_points)} points):")
    print(f"  No opt, no JIT:  {time_none*1000:.2f}ms")
    print(f"  With opt, no JIT: {time_opt*1000:.2f}ms  ({time_none/time_opt:.2f}x)")
    print(f"  With opt + JIT:   {time_opt_jit*1000:.2f}ms  ({time_none/time_opt_jit:.2f}x)")

    # Verify all give same results
    test_point = test_points[0]
    result_none = compiled_none(test_point)
    result_opt = compiled_opt(test_point)
    result_opt_jit = compiled_opt_jit(test_point)

    print(f"\nCorrectness check (all should match):")
    print(f"  No opt:    {result_none}")
    print(f"  Optimized: {result_opt}")
    print(f"  Opt + JIT: {result_opt_jit}")
    assert jnp.allclose(result_none, result_opt)
    assert jnp.allclose(result_none, result_opt_jit)
    print("  ✓ All results match!")


def main():
    """Run all demos."""
    demo_graph_visualization()
    demo_optimization()
    demo_compilation_performance()

    print("\n" + "=" * 60)
    print("COMPILER DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
