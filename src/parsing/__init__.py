"""
Parsing package for regulation text processing.

This package contains modules for:
- regulation_parser: Core parsing logic for regulation text
- reference_resolver: Cross-reference resolution
- table_extractor: Table extraction from markdown
- id_assigner: Stable ID generation
- multi_format_parser: Multi-format HWPX parser coordinator
"""

from .id_assigner import StableIdAssigner
from .reference_resolver import ReferenceResolver
from .regulation_parser import RegulationParser
from .table_extractor import TableExtractor
from .multi_format_parser import HWPXMultiFormatParser
from .structure_analyzer import StructureAnalyzer, StructureInfo, get_authority_display_name, get_structure_summary
from .structure_patterns import RegulationAuthority, StructurePattern, detect_authority_from_text

__all__ = [
    "RegulationParser",
    "ReferenceResolver",
    "TableExtractor",
    "StableIdAssigner",
    "HWPXMultiFormatParser",
    "StructureAnalyzer",
    "StructureInfo",
    "StructurePattern",
    "RegulationAuthority",
    "detect_authority_from_text",
    "get_authority_display_name",
    "get_structure_summary",
]
