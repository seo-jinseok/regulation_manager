"""
Parsing package for regulation text processing.

This package contains modules for:
- regulation_parser: Core parsing logic for regulation text
- reference_resolver: Cross-reference resolution
- table_extractor: Table extraction from markdown
- id_assigner: Stable ID generation
"""

from .regulation_parser import RegulationParser
from .reference_resolver import ReferenceResolver
from .table_extractor import TableExtractor
from .id_assigner import StableIdAssigner

__all__ = [
    "RegulationParser",
    "ReferenceResolver",
    "TableExtractor",
    "StableIdAssigner",
]
