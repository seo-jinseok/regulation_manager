# Infrastructure Layer
"""
Infrastructure implementations for external systems.

This layer contains concrete implementations of the repository interfaces
defined in the domain layer.
"""

from .json_loader import JSONDocumentLoader

__all__ = [
    "JSONDocumentLoader",
]
