"""
Domain-specific exceptions for Regulation Manager.

Provides fine-grained exception types for better error handling
and debugging across the application.
"""

from typing import Optional


class RegulationManagerError(Exception):
    """Base exception for all Regulation Manager errors."""

    pass


# ============================================================================
# Parsing Exceptions
# ============================================================================


class ParsingError(RegulationManagerError):
    """Base exception for parsing-related errors."""

    pass


class RegulationParseError(ParsingError):
    """Error occurred while parsing regulation text."""

    def __init__(self, message: str, line_number: Optional[int] = None):
        self.line_number = line_number
        if line_number:
            message = f"Line {line_number}: {message}"
        super().__init__(message)


class TableParseError(ParsingError):
    """Error occurred while parsing a table."""

    pass


class ReferenceResolutionError(ParsingError):
    """Error occurred while resolving a regulation reference."""

    def __init__(self, reference: str, reason: str):
        self.reference = reference
        self.reason = reason
        super().__init__(f"Cannot resolve reference '{reference}': {reason}")


# ============================================================================
# Conversion Exceptions
# ============================================================================


class ConversionError(RegulationManagerError):
    """Base exception for conversion-related errors."""

    pass


class HWPConversionError(ConversionError):
    """Error occurred while converting HWP file."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        super().__init__(f"Failed to convert HWP file '{file_path}': {reason}")


class JSONSchemaError(ConversionError):
    """Output JSON does not conform to expected schema."""

    pass


# ============================================================================
# RAG Exceptions
# ============================================================================


class RAGError(RegulationManagerError):
    """Base exception for RAG system errors."""

    pass


class VectorStoreError(RAGError):
    """Error occurred in vector store operations."""

    pass


class SyncError(RAGError):
    """Error occurred during synchronization."""

    def __init__(self, message: str, added: int = 0, failed: int = 0):
        self.added = added
        self.failed = failed
        super().__init__(f"{message} (added: {added}, failed: {failed})")


class SearchError(RAGError):
    """Error occurred during search."""

    pass


class LLMError(RAGError):
    """Error occurred during LLM interaction."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationError(RegulationManagerError):
    """Configuration-related error."""

    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is not configured."""

    def __init__(self, key_name: str):
        self.key_name = key_name
        super().__init__(f"Missing required API key: {key_name}")
