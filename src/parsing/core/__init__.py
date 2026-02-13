"""
Parsing core utilities for HWPX regulation parsing.

This package provides core text processing and normalization utilities
used across the HWPX parsing pipeline.
"""

from .text_normalizer import TextNormalizer, normalize_regulation_text

__all__ = [
    "TextNormalizer",
    "normalize_regulation_text",
]
