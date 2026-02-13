"""
Regulation detection utilities for HWPX parsing.

This package provides detectors for identifying regulation titles,
articles, and other structural elements in HWPX documents.
"""

from .regulation_title_detector import (
    RegulationTitleDetector,
    TitleMatchResult,
    detect_regulation_title,
)

__all__ = [
    "RegulationTitleDetector",
    "TitleMatchResult",
    "detect_regulation_title",
]
