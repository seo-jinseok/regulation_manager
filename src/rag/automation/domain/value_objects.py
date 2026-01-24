"""
Value Objects for RAG Testing Automation System.

Value Objects are immutable objects that represent a concept by its attributes.
Two value objects are equal if their attributes are equal.

Clean Architecture: Domain layer uses only standard library (dataclasses).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class FactCheckStatus(Enum):
    """Status of fact check verification."""

    PASS = "pass"  # Claim is verified
    FAIL = "fail"  # Claim is false
    UNCERTAIN = "uncertain"  # Cannot be verified


@dataclass(frozen=True)
class IntentAnalysis:
    """
    3-level intent analysis for a query.

    Captures surface, hidden, and behavioral intent to improve
    RAG system understanding of user needs.
    """

    surface_intent: str  # 표면적 의도: 쿼리에서 직접 표현된 것
    hidden_intent: str  # 숨겨진 의도: 쿼리 뒤에 있는 실제 니즈
    behavioral_intent: str  # 행동 의도: 사용자가 궁극적으로 하고 싶은 행동


@dataclass(frozen=True)
class FactCheck:
    """
    Fact check result for an answer claim.

    Validates claims made by the RAG system against regulations.
    """

    claim: str  # The claim to verify
    status: FactCheckStatus  # Verification status
    source: str  # Source regulation/article
    confidence: float  # Confidence score (0.0 to 1.0)
    correction: Optional[str] = None  # Corrected information if failed
    explanation: Optional[str] = None  # Additional explanation


@dataclass(frozen=True)
class QualityDimensions:
    """
    Individual dimensions of answer quality.

    Each dimension is scored from 0.0 to 1.0.
    """

    accuracy: float  # 정확성: 규정 내용과 일치
    completeness: float  # 완전성: 질문의 모든 측면에 답변
    relevance: float  # 관련성: 질문 의도에 맞는 답변
    source_citation: float  # 출처 명시: 규정명/조항 인용
    practicality: float  # 실용성: 기한/서류/담당부서 포함 (max 0.5)
    actionability: float  # 행동 가능성: 사용자가 바로 행동 가능 (max 0.5)


@dataclass(frozen=True)
class QualityScore:
    """
    Overall quality score for an answer.

    Total score is 5.0 points maximum.
    Passing threshold: >= 4.0 AND all fact checks pass.
    """

    dimensions: QualityDimensions
    total_score: float  # Sum of all dimensions (max 5.0)
    is_pass: bool  # Whether the answer passes quality threshold

    @property
    def is_partial_success(self) -> bool:
        """Check if score is in partial success range (3.0~3.9)."""
        return 3.0 <= self.total_score < 4.0


@dataclass(frozen=True)
class DifficultyDistribution:
    """
    Distribution of test cases by difficulty.

    Used by PersonaGenerator to ensure balanced difficulty coverage.
    """

    easy_ratio: float = 0.3  # 30% easy
    medium_ratio: float = 0.4  # 40% medium
    hard_ratio: float = 0.3  # 30% hard

    def __post_init__(self) -> None:
        """Validate that ratios sum to 1.0."""
        total = self.easy_ratio + self.medium_ratio + self.hard_ratio
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Difficulty ratios must sum to 1.0, got {total}")


class RAGComponent(Enum):
    """8 RAG components for behavior analysis."""

    SELF_RAG = "self_rag"  # Self-RAG reflection
    HYDE = "hyde"  # Hypothetical Document Embeddings
    CORRECTIVE_RAG = "corrective_rag"  # Corrective RAG retrieval evaluation
    HYBRID_SEARCH = "hybrid_search"  # Hybrid search (dense + sparse)
    BGE_RERANKER = "bge_reranker"  # BGE reranker
    QUERY_ANALYZER = "query_analyzer"  # Query analysis
    DYNAMIC_QUERY_EXPANSION = "dynamic_query_expansion"  # Dynamic query expansion
    FACT_CHECK = "fact_check"  # Fact checking


@dataclass(frozen=True)
class ComponentContribution:
    """
    Contribution score of a RAG component to test result.

    Score ranges from -2 (negative contribution) to +2 (positive contribution).
    """

    component: RAGComponent  # The RAG component
    score: int  # Contribution score (-2 to +2)
    reason: str  # Explanation for the score
    was_executed: bool  # Whether the component was executed

    def __post_init__(self) -> None:
        """Validate score range."""
        if not -2 <= self.score <= 2:
            raise ValueError(
                f"Component score must be between -2 and +2, got {self.score}"
            )


@dataclass(frozen=True)
class ComponentAnalysis:
    """
    Analysis of RAG component behavior for a test result.

    Identifies which components contributed positively or negatively
    to the test outcome.
    """

    test_case_id: str  # Reference to test case
    contributions: List[ComponentContribution]  # All component contributions
    overall_impact: str  # Overall impact assessment
    failure_cause_components: List[RAGComponent]  # Components that caused failure
    timestamp_importance: bool  # Whether timing was a factor

    @property
    def net_impact_score(self) -> int:
        """Calculate net impact score across all components."""
        return sum(c.score for c in self.contributions)

    @property
    def critical_failures(self) -> List[ComponentContribution]:
        """Get components with critical failures (score = -2)."""
        return [c for c in self.contributions if c.score == -2]


@dataclass(frozen=True)
class FiveWhyAnalysis:
    """
    5-Why root cause analysis for test failures.

    Chains 5 levels of "why" questions to identify root cause.
    """

    test_case_id: str  # Reference to test case
    original_failure: str  # Original failure description
    why_chain: List[str]  # 5 levels of "why" answers
    root_cause: str  # Identified root cause
    suggested_fix: str  # Suggested fix
    component_to_patch: Optional[str] = (
        None  # Component to patch (e.g., "intents.json")
    )
    code_change_required: bool = False  # Whether code change is required

    @property
    def analysis_depth(self) -> int:
        """Get depth of analysis (should be 5)."""
        return len(self.why_chain)
