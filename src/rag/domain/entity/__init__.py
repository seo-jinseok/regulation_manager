"""
Entity recognition module for Regulation RAG System.

This module implements SPEC-RAG-SEARCH-001 TAG-001: Enhanced Entity Recognition.

Components:
- EntityType: Enum of 6 entity types (SECTION, PROCEDURE, REQUIREMENT, BENEFIT, DEADLINE, HYPERNYM)
- EntityMatch: Data class for entity match results
- EntityRecognitionResult: Complete recognition result with expansions
- RegulationEntityRecognizer: Main recognizer class
"""

from .entity_recognizer import RegulationEntityRecognizer
from .entity_types import EntityMatch, EntityRecognitionResult, EntityType

__all__ = [
    "EntityType",
    "EntityMatch",
    "EntityRecognitionResult",
    "RegulationEntityRecognizer",
]
