"""
Domain models for RAG quality evaluation.

Clean Architecture: Domain layer uses only standard library (dataclasses).
These are the core business entities for evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EvaluationFramework(Enum):
    """Evaluation framework options."""

    RAGAS = "ragas"
    DEEPEVAL = "deepeval"


class QualityIssue(Enum):
    """Types of quality issues for categorization."""

    HALLUCINATION = "hallucination"  # Faithfulness < 0.9
    IRRELEVANT_RETRIEVAL = "irrelevant_retrieval"  # Contextual Precision < 0.8
    INCOMPLETE_ANSWER = "incomplete_answer"  # Contextual Recall < 0.8
    IRRELEVANT_ANSWER = "irrelevant_answer"  # Answer Relevancy < 0.85


@dataclass
class EvaluationThresholds:
    """Thresholds for evaluation metrics with staged progression."""

    # Current stage (1=initial, 2=intermediate, 3=target)
    stage: int = 1

    # Stage 1 (Initial - Week 1): Conservative baseline
    faithfulness_stage1: float = 0.60
    answer_relevancy_stage1: float = 0.70
    contextual_precision_stage1: float = 0.65
    contextual_recall_stage1: float = 0.65
    overall_pass_stage1: float = 0.60

    # Stage 2 (Intermediate - Week 2-3): Progressive improvement
    faithfulness_stage2: float = 0.75
    answer_relevancy_stage2: float = 0.75
    contextual_precision_stage2: float = 0.70
    contextual_recall_stage2: float = 0.70
    overall_pass_stage2: float = 0.70

    # Stage 3 (Target - Week 4+): Production quality
    faithfulness_stage3: float = 0.80
    answer_relevancy_stage3: float = 0.80
    contextual_precision_stage3: float = 0.75
    contextual_recall_stage3: float = 0.75
    overall_pass_stage3: float = 0.75

    # Legacy thresholds (backward compatibility, default to stage 3)
    faithfulness: float = 0.80
    answer_relevancy: float = 0.80
    contextual_precision: float = 0.75
    contextual_recall: float = 0.75

    # Critical thresholds for alerts (unchanged)
    faithfulness_critical: float = 0.60
    relevancy_critical: float = 0.60
    precision_critical: float = 0.55
    recall_critical: float = 0.55

    # Stage threshold definitions
    STAGE_THRESHOLDS = {
        1: {
            "faithfulness": 0.60,
            "answer_relevancy": 0.70,
            "contextual_precision": 0.65,
            "contextual_recall": 0.65,
            "overall_pass": 0.60,
        },
        2: {
            "faithfulness": 0.75,
            "answer_relevancy": 0.75,
            "contextual_precision": 0.70,
            "contextual_recall": 0.70,
            "overall_pass": 0.70,
        },
        3: {
            "faithfulness": 0.80,
            "answer_relevancy": 0.80,
            "contextual_precision": 0.75,
            "contextual_recall": 0.75,
            "overall_pass": 0.75,
        },
    }

    def __post_init__(self):
        """Update current thresholds based on stage."""
        thresholds = self.STAGE_THRESHOLDS.get(self.stage, self.STAGE_THRESHOLDS[1])
        self.faithfulness = thresholds["faithfulness"]
        self.answer_relevancy = thresholds["answer_relevancy"]
        self.contextual_precision = thresholds["contextual_precision"]
        self.contextual_recall = thresholds["contextual_recall"]

    @classmethod
    def for_stage(cls, stage: int) -> "EvaluationThresholds":
        """Create thresholds for a specific stage."""
        return cls(stage=stage)

    def get_thresholds_for_stage(self, stage: int) -> dict:
        """Get threshold dictionary for a specific stage."""
        return self.STAGE_THRESHOLDS.get(stage, self.STAGE_THRESHOLDS[1])

    def get_current_stage_name(self) -> str:
        """Get human-readable stage name."""
        stage_names = {
            1: "Initial (Week 1)",
            2: "Intermediate (Week 2-3)",
            3: "Target (Week 4+)",
        }
        return stage_names.get(self.stage, "Unknown")

    def is_below_minimum(self, metric_name: str, score: float) -> bool:
        """Check if score is below minimum threshold for current stage."""
        thresholds = {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "contextual_precision": self.contextual_precision,
            "contextual_recall": self.contextual_recall,
        }
        return score < thresholds.get(metric_name, 0.0)

    def is_below_critical(self, metric_name: str, score: float) -> bool:
        """Check if score is below critical threshold."""
        critical_thresholds = {
            "faithfulness": self.faithfulness_critical,
            "answer_relevancy": self.relevancy_critical,
            "contextual_precision": self.precision_critical,
            "contextual_recall": self.recall_critical,
        }
        return score < critical_thresholds.get(metric_name, 0.0)

    def get_overall_pass_threshold(self) -> float:
        """Get the overall pass threshold for current stage."""
        return self.STAGE_THRESHOLDS.get(self.stage, {}).get("overall_pass", 0.60)


@dataclass
class MetricScore:
    """Score for a single evaluation metric."""

    name: str
    score: float  # 0.0 to 1.0
    passed: bool
    reason: Optional[str] = None


@dataclass
class EvaluationResult:
    """
    Result of RAG quality evaluation.

    Contains scores for all four core metrics and overall assessment.
    """

    query: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float
    overall_score: float
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "contextual_precision": self.contextual_precision,
            "contextual_recall": self.contextual_recall,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            answer=data["answer"],
            contexts=data["contexts"],
            faithfulness=data["faithfulness"],
            answer_relevancy=data["answer_relevancy"],
            contextual_precision=data["contextual_precision"],
            contextual_recall=data["contextual_recall"],
            overall_score=data["overall_score"],
            passed=data["passed"],
            failure_reasons=data.get("failure_reasons", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PersonaProfile:
    """
    User persona for evaluation.

    Defines characteristics, query patterns, and preferences for different user types.
    """

    name: str  # e.g., "freshman", "professor"
    display_name: str  # e.g., "신입생", "교수"
    expertise_level: str  # "beginner", "intermediate", "advanced"
    vocabulary_style: str  # "simple", "academic", "administrative", "mixed"
    query_templates: List[str]
    common_topics: List[str]
    answer_preferences: Dict[str, Any] = field(default_factory=dict)

    def generate_query(self, topic: str) -> str:
        """Generate a query for this persona."""
        import random

        template = random.choice(self.query_templates)
        return template.format(topic=topic)


@dataclass
class TestCase:
    """
    Test case for evaluation.

    Contains query, ground truth, and metadata for testing.
    """

    question: str
    ground_truth: str
    regulation_id: Optional[str] = None
    section_id: Optional[str] = None
    question_type: Optional[str] = None  # "procedural", "conditional", "factual"
    persona: Optional[str] = None
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "regulation_id": self.regulation_id,
            "section_id": self.section_id,
            "question_type": self.question_type,
            "persona": self.persona,
            "valid": self.valid,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            ground_truth=data["ground_truth"],
            regulation_id=data.get("regulation_id"),
            section_id=data.get("section_id"),
            question_type=data.get("question_type"),
            persona=data.get("persona"),
            valid=data.get("valid", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ImprovementSuggestion:
    """
    Suggestion for quality improvement.

    Generated by analyzing evaluation failures and providing actionable recommendations.
    """

    issue_type: str  # QualityIssue enum value
    component: str  # "prompt_engineering", "reranking", "retrieval", "citation"
    recommendation: str
    expected_impact: str  # e.g., "+0.15 Faithfulness score"
    implementation_effort: str  # e.g., "Low (1 hour)"
    affected_count: int  # Number of queries affected
    severity: str = "medium"  # "low", "medium", "high"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "issue_type": self.issue_type,
            "component": self.component,
            "recommendation": self.recommendation,
            "expected_impact": self.expected_impact,
            "implementation_effort": self.implementation_effort,
            "affected_count": self.affected_count,
            "severity": self.severity,
            "parameters": self.parameters,
        }


@dataclass
class EvaluationSummary:
    """
    Summary of evaluation results for a batch.

    Contains aggregate statistics and breakdowns.
    """

    total_queries: int
    passed_queries: int
    pass_rate: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_contextual_precision: float
    avg_contextual_recall: float
    avg_overall_score: float
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_queries": self.total_queries,
            "passed_queries": self.passed_queries,
            "pass_rate": self.pass_rate,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_answer_relevancy": self.avg_answer_relevancy,
            "avg_contextual_precision": self.avg_contextual_precision,
            "avg_contextual_recall": self.avg_contextual_recall,
            "avg_overall_score": self.avg_overall_score,
            "failures_by_type": self.failures_by_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PersonaMetrics:
    """
    Metrics breakdown for a specific persona.

    Contains persona-specific evaluation results.
    """

    persona_name: str
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float
    overall_score: float
    query_count: int
    pass_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "persona_name": self.persona_name,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "contextual_precision": self.contextual_precision,
            "contextual_recall": self.contextual_recall,
            "overall_score": self.overall_score,
            "query_count": self.query_count,
            "pass_rate": self.pass_rate,
            "timestamp": self.timestamp.isoformat(),
        }
