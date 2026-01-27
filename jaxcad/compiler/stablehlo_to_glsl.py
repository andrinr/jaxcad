"""StableHLO to GLSL compiler for SDF raymarching shaders.

This module compiles JAX/StableHLO IR to GLSL fragment shaders.
Initially targets the subset of StableHLO operations used by SDFs.

Architecture:
1. Extract StableHLO from JAX function
2. Parse StableHLO operations
3. Map to GLSL equivalents
4. Generate shader code with raymarching boilerplate

Supported StableHLO ops (Phase 1 - SDF subset):
- Arithmetic: add, subtract, multiply, divide, sqrt, abs, negate
- Comparison: compare, select (ternary), minimum, maximum
- Trigonometry: sin, cos, tan, atan2
- Vector: dot_general, reshape, slice, broadcast
- Constants: constant, iota
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import re


@dataclass
class GLSLVariable:
    """Represents a variable in GLSL code."""
    name: str
    glsl_type: str  # float, vec2, vec3, vec4, mat3, mat4
    declaration: Optional[str] = None  # Full declaration if needed


@dataclass
class GLSLFunction:
    """Represents a compiled GLSL function."""
    signature: str  # e.g., "float sdf(vec3 p)"
    body: str  # Function body
    dependencies: Set[str]  # Other functions this depends on


class StableHLOToGLSL:
    """Compiler from StableHLO IR to GLSL shader code.

    Example:
        compiler = StableHLOToGLSL()
        glsl_code = compiler.compile(stablehlo_ir)
        shader = compiler.wrap_in_fragment_shader(glsl_code)
    """

    def __init__(self):
        self.variable_counter = 0
        self.functions: Dict[str, GLSLFunction] = {}
        self.glsl_type_map = {
            'f32': 'float',
            'f64': 'float',  # GLSL doesn't have f64, downcast
            'i32': 'int',
            'i64': 'int',
            'bool': 'bool',
        }

    def fresh_var(self, base: str = "tmp") -> str:
        """Generate a fresh variable name."""
        name = f"{base}_{self.variable_counter}"
        self.variable_counter += 1
        return name

    def map_type(self, stablehlo_type: str) -> str:
        """Map StableHLO type to GLSL type.

        Examples:
            'tensor<f32>' -> 'float'
            'tensor<3xf32>' -> 'vec3'
            'tensor<3x3xf32>' -> 'mat3'
        """
        # Parse tensor shape
        match = re.match(r'tensor<(?:(\d+)x)?(?:(\d+)x)?f32>', stablehlo_type)
        if not match:
            return 'float'

        dim1, dim2 = match.groups()

        if dim2:  # Matrix
            return f'mat{dim1}'
        elif dim1:  # Vector
            dim = int(dim1)
            if dim <= 4:
                return f'vec{dim}'
            return f'float[{dim}]'  # Array for larger vectors
        else:  # Scalar
            return 'float'

    def compile_op(self, op: str, operands: List[str], result_type: str) -> str:
        """Compile a single StableHLO operation to GLSL.

        Args:
            op: Operation name (e.g., 'add', 'multiply', 'sqrt')
            operands: List of operand variable names
            result_type: StableHLO type of result

        Returns:
            GLSL expression for this operation
        """
        glsl_type = self.map_type(result_type)

        # Arithmetic operations
        if op == 'add':
            return f"{operands[0]} + {operands[1]}"
        elif op == 'subtract':
            return f"{operands[0]} - {operands[1]}"
        elif op == 'multiply':
            return f"{operands[0]} * {operands[1]}"
        elif op == 'divide':
            return f"{operands[0]} / {operands[1]}"
        elif op == 'negate':
            return f"-{operands[0]}"

        # Math functions
        elif op == 'sqrt':
            return f"sqrt({operands[0]})"
        elif op == 'abs':
            return f"abs({operands[0]})"
        elif op == 'sin':
            return f"sin({operands[0]})"
        elif op == 'cos':
            return f"cos({operands[0]})"
        elif op == 'tan':
            return f"tan({operands[0]})"
        elif op == 'atan2':
            return f"atan({operands[0]}, {operands[1]})"

        # Comparison and selection
        elif op == 'minimum':
            return f"min({operands[0]}, {operands[1]})"
        elif op == 'maximum':
            return f"max({operands[0]}, {operands[1]})"
        elif op == 'select':
            # StableHLO: select(pred, true_val, false_val)
            # GLSL: pred ? true_val : false_val
            return f"({operands[0]} ? {operands[1]} : {operands[2]})"
        elif op == 'compare':
            # Would need comparison type (EQ, NE, LT, etc.)
            # For now, assume less-than
            return f"({operands[0]} < {operands[1]})"

        # Vector operations
        elif op == 'dot_general':
            # Dot product
            return f"dot({operands[0]}, {operands[1]})"
        elif op == 'length':
            return f"length({operands[0]})"
        elif op == 'normalize':
            return f"normalize({operands[0]})"

        # Fallback
        else:
            return f"/* TODO: {op}({', '.join(operands)}) */"

    def generate_sdf_function(self, name: str, body: str, param_type: str = "vec3") -> str:
        """Generate a complete SDF function in GLSL.

        Args:
            name: Function name (e.g., 'sphere_sdf')
            body: GLSL code for function body
            param_type: Type of point parameter (usually vec3)

        Returns:
            Complete GLSL function definition
        """
        return f"""float {name}({param_type} p) {{
{self.indent(body, 4)}
}}"""

    @staticmethod
    def indent(code: str, spaces: int) -> str:
        """Indent code by given number of spaces."""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else ''
                        for line in code.split('\n'))

    def generate_raymarching_shader(
        self,
        sdf_function: str,
        max_steps: int = 100,
        max_distance: float = 100.0,
        epsilon: float = 0.001
    ) -> str:
        """Generate complete fragment shader with raymarching boilerplate.

        Args:
            sdf_function: GLSL SDF function definition
            max_steps: Maximum raymarching steps
            max_distance: Maximum ray distance
            epsilon: Surface hit threshold

        Returns:
            Complete GLSL fragment shader code
        """
        return f"""#version 330 core

// Fragment shader inputs
in vec2 fragCoord;

// Fragment shader outputs
out vec4 fragColor;

// Uniforms
uniform vec2 iResolution;
uniform float iTime;
uniform vec3 cameraPos;
uniform vec3 cameraTarget;
uniform float cameraFov;

// Constants
const int MAX_STEPS = {max_steps};
const float MAX_DISTANCE = {max_distance};
const float EPSILON = {epsilon};

// ============================================================================
// SDF Function (compiled from JAX/StableHLO)
// ============================================================================

{sdf_function}

// ============================================================================
// Raymarching
// ============================================================================

vec3 estimateNormal(vec3 p) {{
    float eps = EPSILON;
    vec3 n = vec3(
        sdf(vec3(p.x + eps, p.y, p.z)) - sdf(vec3(p.x - eps, p.y, p.z)),
        sdf(vec3(p.x, p.y + eps, p.z)) - sdf(vec3(p.x, p.y - eps, p.z)),
        sdf(vec3(p.x, p.y, p.z + eps)) - sdf(vec3(p.x, p.y, p.z - eps))
    );
    return normalize(n);
}}

float raymarch(vec3 ro, vec3 rd) {{
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {{
        vec3 p = ro + rd * t;
        float d = sdf(p);
        if (d < EPSILON) {{
            return t;
        }}
        t += d;
        if (t > MAX_DISTANCE) {{
            break;
        }}
    }}
    return -1.0;
}}

vec3 render(vec3 ro, vec3 rd) {{
    float t = raymarch(ro, rd);

    if (t < 0.0) {{
        // Miss - return background
        return vec3(0.1, 0.1, 0.15);
    }}

    // Hit - calculate shading
    vec3 p = ro + rd * t;
    vec3 normal = estimateNormal(p);

    // Simple Phong lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(0.0, dot(normal, lightDir));
    float ambient = 0.3;

    vec3 color = vec3(0.8, 0.6, 0.4);
    return color * (ambient + diffuse);
}}

// ============================================================================
// Camera
// ============================================================================

mat3 lookAt(vec3 ro, vec3 target, float roll) {{
    vec3 up = vec3(sin(roll), cos(roll), 0.0);
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(up, forward));
    vec3 newUp = cross(forward, right);
    return mat3(right, newUp, forward);
}}

void main() {{
    // Normalized pixel coordinates (-1 to 1)
    vec2 uv = (2.0 * fragCoord - iResolution) / iResolution.y;

    // Camera setup
    vec3 ro = cameraPos;
    mat3 cam = lookAt(ro, cameraTarget, 0.0);
    vec3 rd = cam * normalize(vec3(uv, cameraFov));

    // Render
    vec3 color = render(ro, rd);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}}
"""


# ============================================================================
# Example Usage
# ============================================================================

def example_sphere_shader():
    """Generate a simple sphere shader as an example."""
    compiler = StableHLOToGLSL()

    # Hand-written SDF for now (would come from StableHLO later)
    sdf_code = """vec3 p_centered = p;
float distance_from_origin = length(p_centered);
float radius = 1.0;
return distance_from_origin - radius;"""

    sdf_function = compiler.generate_sdf_function("sdf", sdf_code)
    shader = compiler.generate_raymarching_shader(sdf_function)

    return shader


def example_union_shader():
    """Generate a shader with boolean union."""
    compiler = StableHLOToGLSL()

    sdf_code = """// Sphere 1
vec3 p1 = p - vec3(-1.0, 0.0, 0.0);
float d1 = length(p1) - 0.8;

// Sphere 2
vec3 p2 = p - vec3(1.0, 0.0, 0.0);
float d2 = length(p2) - 0.8;

// Union
return min(d1, d2);"""

    sdf_function = compiler.generate_sdf_function("sdf", sdf_code)
    shader = compiler.generate_raymarching_shader(sdf_function)

    return shader


if __name__ == "__main__":
    # Generate example shaders
    print("=" * 70)
    print("SPHERE SHADER")
    print("=" * 70)
    print(example_sphere_shader())

    print("\n" + "=" * 70)
    print("UNION SHADER")
    print("=" * 70)
    print(example_union_shader())
