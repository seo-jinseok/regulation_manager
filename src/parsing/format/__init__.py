"""
Format Classification Module for HWPX Regulation Parsing.

This module provides format classification infrastructure for HWPX regulations,
supporting detection of article, list, guideline, and unstructured formats.

Components:
    FormatType: Enum of regulation format types
    ListPattern: Enum of list pattern types
    FormatClassifier: Main classifier for format detection
    ClassificationResult: Data class for classification results

Example:
    >>> from src.parsing.format import FormatClassifier, FormatType
    >>> classifier = FormatClassifier()
    >>> result = classifier.classify("제1조 목적")
    >>> print(result.format_type)  # FormatType.ARTICLE
    >>> print(result.confidence)  # 0.9+

Reference: SPEC-HWXP-002, TASK-001
"""

from src.parsing.format.format_type import FormatType, ListPattern
from src.parsing.format.format_classifier import (
    FormatClassifier,
    ClassificationResult,
)

__all__ = [
    "FormatType",
    "ListPattern",
    "FormatClassifier",
    "ClassificationResult",
]
