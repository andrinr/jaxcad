"""Transformation operations for SDFs."""

from jaxcad.sdf.base import SDF
from jaxcad.sdf.transforms.affine import Translate, Rotate, Scale
from jaxcad.sdf.transforms.deformations import Twist

# Define transforms with their method names
TRANSFORMS = {
    'translate': Translate,
    'rotate': Rotate,
    'scale': Scale,
    'twist': Twist,
}

# Export class names
__all__ = [cls.__name__ for cls in TRANSFORMS.values()]

# Register transforms as fluent API methods on SDF class
for method_name, transform_class in TRANSFORMS.items():
    SDF.register(method_name, transform_class)
