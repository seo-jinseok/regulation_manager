"""
Query domain module for RAG system.

Provides query transformation and enhancement components:
- ColloquialTransformer: Colloquial-to-formal Korean query transformation
- MorphologicalExpander: Korean morphological analysis and expansion
"""

from .colloquial_transformer import (
    ColloquialPattern,
    ColloquialTransformer,
    TransformResult,
    create_colloquial_transformer,
)
from .morphological_expander import (
    ExpansionMode,
    MorphologicalExpansionResult,
    MorphologicalExpander,
    create_morphological_expander,
)

__all__ = [
    # Colloquial transformation
    "ColloquialTransformer",
    "ColloquialPattern",
    "TransformResult",
    "create_colloquial_transformer",
    # Morphological expansion
    "MorphologicalExpander",
    "MorphologicalExpansionResult",
    "ExpansionMode",
    "create_morphological_expander",
]
