"""
Failure Classifier for RAG Evaluation Results.

Categorizes evaluation failures into types for analysis and improvement.
Part of SPEC-RAG-EVAL-001 Milestone 4: Report Enhancement.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of evaluation failures."""

    HALLUCINATION = "hallucination"  # Content not in source
    MISSING_INFO = "missing_info"  # Incomplete information
    CITATION_ERROR = "citation_error"  # Invalid citation format
    RETRIEVAL_FAILURE = "retrieval_failure"  # Relevant docs not retrieved
    AMBIGUITY = "ambiguity"  # Unclear or ambiguous response
    IRRELEVANCE = "irrelevance"  # Off-topic response
    LOW_QUALITY = "low_quality"  # Generic quality issues
    UNKNOWN = "unknown"  # Unclassified failure


@dataclass
class FailureSummary:
    """Summary of a failure type occurrence."""

    failure_type: FailureType
    count: int
    examples: List[str] = field(default_factory=list)
    affected_personas: List[str] = field(default_factory=list)
    avg_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_type": self.failure_type.value,
            "count": self.count,
            "examples": self.examples[:5],  # Limit examples
            "affected_personas": self.affected_personas,
            "avg_score": round(self.avg_score, 3),
        }


class FailureClassifier:
    """
    Classifies evaluation failures into categories.

    Provides:
    - Single result classification
    - Batch classification with frequency analysis
    - Top failure identification
    - Pattern-based failure detection
    """

    # Patterns for hallucination detection
    HALLUCINATION_PATTERNS = [
        r"\d{2,3}-\d{3,4}-\d{4}",  # Phone numbers
        r"02-\d{3,4}-\d{4}",  # Seoul phone numbers
        r"\d{3}-\d{4}-\d{4}",  # Mobile phone numbers
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email addresses
        r"(?:http|https)://[^\s]+",  # URLs
    ]

    # Patterns for citation errors
    CITATION_PATTERNS = [
        r"「[^」]+」제\d+조",  # Proper citation format
        r"제\d+조(?:제\d+항)?",  # Article reference
    ]

    # Keywords indicating missing information
    MISSING_INFO_KEYWORDS = [
        "기한",
        "절차",
        "서류",
        "자격",
        "요건",
        "방법",
        "신청",
        "승인",
    ]

    # Phrases indicating ambiguity
    AMBIGUITY_PHRASES = [
        "대학마다 다릅니다",
        "확인이 필요합니다",
        "담당 부서에 문의",
        "일반적으로",
        "보통은",
        "대부분",
    ]

    def __init__(self, faithfulness_threshold: float = 0.7, relevancy_threshold: float = 0.7):
        """
        Initialize the failure classifier.

        Args:
            faithfulness_threshold: Threshold below which to check for hallucination
            relevancy_threshold: Threshold below which to check for irrelevance
        """
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold

    def classify(self, result: Dict[str, Any]) -> FailureType:
        """
        Classify a single evaluation result into a failure type.

        Args:
            result: Evaluation result dictionary with scores and content

        Returns:
            FailureType indicating the primary failure category
        """
        # Check if result actually failed
        if result.get("passed", True):
            return FailureType.UNKNOWN

        answer = result.get("answer", "")
        query = result.get("query", "")
        contexts = result.get("contexts", [])

        # Get scores
        faithfulness = result.get("faithfulness", 1.0)
        relevancy = result.get("answer_relevancy", 1.0)
        precision = result.get("contextual_precision", 1.0)
        recall = result.get("contextual_recall", 1.0)

        # Check for hallucination (low faithfulness with potential hallucinated content)
        if faithfulness < self.faithfulness_threshold:
            if self._detect_hallucination_patterns(answer):
                return FailureType.HALLUCINATION

        # Check for retrieval failure (low precision/recall)
        if precision < 0.7 or recall < 0.7:
            return FailureType.RETRIEVAL_FAILURE

        # Check for irrelevance (low relevancy score)
        if relevancy < self.relevancy_threshold:
            return FailureType.IRRELEVANCE

        # Check for missing information
        if self._detect_missing_info(query, answer):
            return FailureType.MISSING_INFO

        # Check for citation errors
        if self._detect_citation_error(answer, contexts):
            return FailureType.CITATION_ERROR

        # Check for ambiguity
        if self._detect_ambiguity(answer):
            return FailureType.AMBIGUITY

        # Default to low quality for unclassified failures
        return FailureType.LOW_QUALITY

    def classify_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[FailureType, int]:
        """
        Classify a batch of results and count failures by type.

        Args:
            results: List of evaluation result dictionaries

        Returns:
            Dictionary mapping FailureType to occurrence count
        """
        counts: Dict[FailureType, int] = {}

        for result in results:
            failure_type = self.classify(result)
            counts[failure_type] = counts.get(failure_type, 0) + 1

        logger.info(f"Classified {len(results)} results into {len(counts)} failure types")
        return counts

    def get_top_failures(
        self,
        results: List[Dict[str, Any]],
        limit: int = 5,
    ) -> List[FailureSummary]:
        """
        Get the top failure types with details.

        Args:
            results: List of evaluation result dictionaries
            limit: Maximum number of failure types to return

        Returns:
            List of FailureSummary objects sorted by frequency
        """
        # Group results by failure type
        failures_by_type: Dict[FailureType, List[Dict[str, Any]]] = {}

        for result in results:
            if result.get("passed", True):
                continue

            failure_type = self.classify(result)
            if failure_type not in failures_by_type:
                failures_by_type[failure_type] = []
            failures_by_type[failure_type].append(result)

        # Create summaries
        summaries = []
        for failure_type, type_results in failures_by_type.items():
            examples = [
                r.get("query", "")[:100]  # Truncate for readability
                for r in type_results[:5]
            ]

            personas = list(set(
                r.get("persona", "unknown")
                for r in type_results
                if r.get("persona")
            ))

            scores = [
                r.get("overall_score", 0)
                for r in type_results
            ]
            avg_score = sum(scores) / len(scores) if scores else 0

            summary = FailureSummary(
                failure_type=failure_type,
                count=len(type_results),
                examples=examples,
                affected_personas=personas,
                avg_score=avg_score,
            )
            summaries.append(summary)

        # Sort by count descending
        summaries.sort(key=lambda s: s.count, reverse=True)

        return summaries[:limit]

    def _detect_hallucination_patterns(self, text: str) -> bool:
        """Check if text contains potential hallucinated content."""
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _detect_missing_info(self, query: str, answer: str) -> bool:
        """Check if answer is missing expected information based on query."""
        # Check if query asks for specific info but answer doesn't provide it
        query_lower = query.lower()
        answer_lower = answer.lower()

        for keyword in self.MISSING_INFO_KEYWORDS:
            if keyword in query_lower and keyword not in answer_lower:
                return True

        # Check for very short answers to complex queries
        if len(query) > 20 and len(answer) < 50:
            return True

        return False

    def _detect_citation_error(self, answer: str, contexts: List[str]) -> bool:
        """Check for citation format errors or unverifiable citations."""
        # Check if answer mentions regulations but without proper citation
        regulation_mentioned = bool(
            re.search(r"규정|학칙|지침|세칙", answer)
        )

        has_proper_citation = any(
            re.search(pattern, answer)
            for pattern in self.CITATION_PATTERNS
        )

        if regulation_mentioned and not has_proper_citation:
            return True

        return False

    def _detect_ambiguity(self, answer: str) -> bool:
        """Check for ambiguous or non-committal responses."""
        for phrase in self.AMBIGUITY_PHRASES:
            if phrase in answer:
                return True
        return False

    def get_failure_report(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive failure report.

        Args:
            results: List of evaluation result dictionaries

        Returns:
            Dictionary with failure analysis report
        """
        total = len(results)
        failed = sum(1 for r in results if not r.get("passed", True))

        if failed == 0:
            return {
                "total_evaluated": total,
                "total_failures": 0,
                "failure_rate": 0.0,
                "failures": [],
                "recommendations": [],
            }

        failure_counts = self.classify_batch(results)
        top_failures = self.get_top_failures(results, limit=5)

        return {
            "total_evaluated": total,
            "total_failures": failed,
            "failure_rate": round(failed / total, 3) if total > 0 else 0.0,
            "failures": {ft.value: count for ft, count in failure_counts.items()},
            "top_failures": [f.to_dict() for f in top_failures],
            "summary": self._generate_summary(top_failures),
        }

    def _generate_summary(self, top_failures: List[FailureSummary]) -> str:
        """Generate a human-readable summary of failures."""
        if not top_failures:
            return "No failures detected."

        parts = []
        for failure in top_failures[:3]:
            parts.append(
                f"{failure.failure_type.value}: {failure.count} occurrences"
            )

        return " | ".join(parts)
