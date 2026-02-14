"""
Citation Domain Module.

Provides citation extraction, validation, and enhancement for Korean regulations.
"""

from .article_number_extractor import ArticleNumber, ArticleNumberExtractor, ArticleType
from .citation_enhancer import CitationEnhancer, EnhancedCitation
from .citation_validator import CitationValidator, ValidationStatus, ValidationResult
from .citation_patterns import CitationFormat, CitationPatterns
from .citation_verification_service import (
    CitationExtractor,
    CitationVerificationService,
    ExtractedCitation,
)

__all__ = [
    # Article number extraction
    "ArticleNumber",
    "ArticleNumberExtractor",
    "ArticleType",
    # Citation enhancement
    "CitationEnhancer",
    "EnhancedCitation",
    # Citation validation
    "CitationValidator",
    "ValidationStatus",
    "ValidationResult",
    # Citation patterns (NEW - SPEC-RAG-Q-004)
    "CitationFormat",
    "CitationPatterns",
    # Citation verification (NEW - SPEC-RAG-Q-004)
    "CitationExtractor",
    "CitationVerificationService",
    "ExtractedCitation",
]
