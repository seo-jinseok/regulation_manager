"""
Application layer for RAG quality evaluation.

Contains use cases and orchestration logic following Clean Architecture principles.
This layer coordinates domain objects and infrastructure implementations.
"""

from .checkpoint_manager import (
    CheckpointManager,
    EvaluationProgress,
    PersonaProgress,
)
from .evaluation_service import EvaluationService
from .progress_reporter import (
    PersonaProgressInfo,
    ProgressInfo,
    ProgressReporter,
    create_progress_bar,
)
from .resume_controller import (
    MergedResults,
    ResumeContext,
    ResumeController,
)

__all__ = [
    # Evaluation Service
    "EvaluationService",
    # Checkpoint Management (SPEC-RAG-EVAL-001)
    "CheckpointManager",
    "EvaluationProgress",
    "PersonaProgress",
    # Progress Reporting (SPEC-RAG-EVAL-001)
    "ProgressReporter",
    "ProgressInfo",
    "PersonaProgressInfo",
    "create_progress_bar",
    # Resume Control (SPEC-RAG-EVAL-001)
    "ResumeController",
    "ResumeContext",
    "MergedResults",
]
