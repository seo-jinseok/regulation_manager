"""
Persona-Aware Response Generation Module.

This module provides persona-specific prompt enhancements for RAG responses.
It enables the system to tailor responses to different user types:
- Freshman students: Simple, clear explanations
- Graduate students: Comprehensive, academic responses
- Professors: Formal, detailed with specific citations
- Staff: Administrative focus with procedures
- Parents: Parent-friendly language
- International students: Mixed Korean/English support

Main Components:
- PersonaAwareGenerator: Core persona prompt enhancement
- PersonaPromptBuilder: Fluent interface for building custom prompts
- create_persona_prompt: Convenience function for quick usage
"""

from .persona_generator import (
    PersonaAwareGenerator,
    PersonaPromptBuilder,
    create_persona_prompt,
)

__all__ = [
    "PersonaAwareGenerator",
    "PersonaPromptBuilder",
    "create_persona_prompt",
]
