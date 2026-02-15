"""
Recommendation Engine for RAG Evaluation Improvements.

Generates actionable recommendations based on failure patterns.
Part of SPEC-RAG-EVAL-001 Milestone 4: Report Enhancement.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .failure_classifier import FailureType

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"  # Must fix immediately
    HIGH = "high"  # Fix soon
    MEDIUM = "medium"  # Fix eventually
    LOW = "low"  # Nice to have


@dataclass
class Recommendation:
    """A single improvement recommendation."""

    id: str
    title: str
    description: str
    priority: Priority
    failure_types: List[FailureType]
    actions: List[str]
    spec_mapping: Optional[str] = None
    impact_estimate: str = "Unknown"
    effort_estimate: str = "Medium"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "failure_types": [ft.value for ft in self.failure_types],
            "actions": self.actions,
            "spec_mapping": self.spec_mapping,
            "impact_estimate": self.impact_estimate,
            "effort_estimate": self.effort_estimate,
        }


class RecommendationEngine:
    """
    Generates improvement recommendations from failure patterns.

    Provides:
    - Failure-to-recommendation mapping
    - SPEC requirement mapping
    - Priority-based recommendation ranking
    - Actionable improvement suggestions
    """

    # Mapping from failure types to recommendations
    RECOMMENDATION_TEMPLATES = {
        FailureType.HALLUCINATION: Recommendation(
            id="REC-HALL-001",
            title="Reduce Hallucinations in Generated Content",
            description="Implement stricter context grounding and citation requirements to prevent hallucinated content.",
            priority=Priority.CRITICAL,
            failure_types=[FailureType.HALLUCINATION],
            actions=[
                "Add explicit citation format requirements to system prompt",
                "Implement post-generation fact checking against retrieved contexts",
                "Add penalties for unverified claims in evaluation scoring",
                "Create hallucination detection patterns for common cases (phone numbers, URLs, emails)",
            ],
            spec_mapping="SPEC-RAG-Q-001",
            impact_estimate="High - Reduces critical hallucination errors by 70-80%",
            effort_estimate="Medium",
        ),
        FailureType.MISSING_INFO: Recommendation(
            id="REC-MISS-001",
            title="Improve Information Completeness",
            description="Enhance prompts and retrieval to ensure complete information is provided.",
            priority=Priority.HIGH,
            failure_types=[FailureType.MISSING_INFO],
            actions=[
                "Add completeness checklist to generation prompt",
                "Implement multi-hop retrieval for complex queries",
                "Create query decomposition for multi-part questions",
                "Add specific prompts for deadline, procedure, and requirement information",
            ],
            spec_mapping="SPEC-RAG-Q-002",
            impact_estimate="Medium - Improves completeness scores by 40-50%",
            effort_estimate="Medium",
        ),
        FailureType.CITATION_ERROR: Recommendation(
            id="REC-CITE-001",
            title="Enhance Citation Format and Accuracy",
            description="Improve citation format consistency and accuracy in generated responses.",
            priority=Priority.MEDIUM,
            failure_types=[FailureType.CITATION_ERROR],
            actions=[
                "Add citation format examples to system prompt",
                "Implement citation validation against document metadata",
                "Create citation templates for common regulation references",
                "Add post-processing step to verify citation format",
            ],
            spec_mapping="SPEC-RAG-Q-003",
            impact_estimate="Medium - Improves citation accuracy by 50-60%",
            effort_estimate="Low",
        ),
        FailureType.RETRIEVAL_FAILURE: Recommendation(
            id="REC-RETR-001",
            title="Improve Retrieval Relevance",
            description="Enhance retrieval system to find more relevant documents.",
            priority=Priority.HIGH,
            failure_types=[FailureType.RETRIEVAL_FAILURE],
            actions=[
                "Tune chunk size and overlap parameters",
                "Implement hybrid search combining keyword and semantic search",
                "Add reranking step after initial retrieval",
                "Create domain-specific embeddings fine-tuning",
            ],
            spec_mapping="SPEC-RAG-Q-004",
            impact_estimate="High - Improves retrieval precision by 30-40%",
            effort_estimate="High",
        ),
        FailureType.AMBIGUITY: Recommendation(
            id="REC-AMBG-001",
            title="Reduce Ambiguous Responses",
            description="Make responses more specific and actionable.",
            priority=Priority.MEDIUM,
            failure_types=[FailureType.AMBIGUITY],
            actions=[
                "Add anti-ambiguity instructions to system prompt",
                "Require specific citations for general statements",
                "Implement confidence thresholds for uncertain responses",
                "Create persona-specific response templates",
            ],
            spec_mapping="SPEC-RAG-Q-005",
            impact_estimate="Low - Improves response specificity by 20-30%",
            effort_estimate="Low",
        ),
        FailureType.IRRELEVANCE: Recommendation(
            id="REC-IRRE-001",
            title="Improve Query-Response Relevance",
            description="Ensure responses directly address user queries.",
            priority=Priority.HIGH,
            failure_types=[FailureType.IRRELEVANCE],
            actions=[
                "Add query-type detection for different response formats",
                "Implement response-query alignment checking",
                "Create query classification for routing",
                "Add explicit question-answering instructions",
            ],
            spec_mapping="SPEC-RAG-Q-006",
            impact_estimate="Medium - Improves relevancy scores by 30-40%",
            effort_estimate="Medium",
        ),
        FailureType.LOW_QUALITY: Recommendation(
            id="REC-QUAL-001",
            title="Improve Overall Response Quality",
            description="Address general quality issues in generated responses.",
            priority=Priority.LOW,
            failure_types=[FailureType.LOW_QUALITY],
            actions=[
                "Review and update system prompt for clarity",
                "Add response length guidelines",
                "Implement quality scoring feedback loop",
                "Create response templates for common query types",
            ],
            spec_mapping=None,
            impact_estimate="Low - General quality improvement",
            effort_estimate="Low",
        ),
    }

    def __init__(self):
        """Initialize the recommendation engine."""
        self.recommendations = self.RECOMMENDATION_TEMPLATES.copy()

    def generate_recommendations(
        self,
        failures: Dict[FailureType, int],
        threshold: int = 3,
    ) -> List[Recommendation]:
        """
        Generate recommendations based on failure counts.

        Args:
            failures: Dictionary mapping FailureType to occurrence count
            threshold: Minimum failures to trigger recommendation

        Returns:
            List of applicable Recommendation objects
        """
        recommendations = []

        for failure_type, count in failures.items():
            if count < threshold:
                continue

            if failure_type in self.recommendations:
                rec = self.recommendations[failure_type]
                recommendations.append(rec)

        logger.info(
            f"Generated {len(recommendations)} recommendations from "
            f"{sum(1 for c in failures.values() if c >= threshold)} failure types"
        )

        return recommendations

    def map_to_spec(self, failure_type: FailureType) -> Optional[str]:
        """
        Map a failure type to a SPEC requirement.

        Args:
            failure_type: The failure type to map

        Returns:
            SPEC identifier or None if no mapping exists
        """
        if failure_type in self.recommendations:
            return self.recommendations[failure_type].spec_mapping
        return None

    def prioritize(
        self,
        recommendations: List[Recommendation],
        by_impact: bool = False,
    ) -> List[Recommendation]:
        """
        Sort recommendations by priority.

        Args:
            recommendations: List of recommendations to sort
            by_impact: If True, also consider impact estimate

        Returns:
            Sorted list of recommendations
        """
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }

        def sort_key(rec: Recommendation) -> tuple:
            return (priority_order.get(rec.priority, 99), rec.id)

        return sorted(recommendations, key=sort_key)

    def get_action_plan(
        self,
        recommendations: List[Recommendation],
    ) -> Dict[str, Any]:
        """
        Generate an action plan from recommendations.

        Args:
            recommendations: List of recommendations

        Returns:
            Dictionary with organized action plan
        """
        prioritized = self.prioritize(recommendations)

        immediate = [
            r for r in prioritized
            if r.priority in (Priority.CRITICAL, Priority.HIGH)
        ]
        short_term = [
            r for r in prioritized
            if r.priority == Priority.MEDIUM
        ]
        long_term = [
            r for r in prioritized
            if r.priority == Priority.LOW
        ]

        all_actions = []
        for rec in prioritized:
            for i, action in enumerate(rec.actions):
                all_actions.append({
                    "recommendation_id": rec.id,
                    "action": action,
                    "priority": rec.priority.value,
                    "sequence": i + 1,
                })

        return {
            "immediate_actions": [r.to_dict() for r in immediate],
            "short_term_actions": [r.to_dict() for r in short_term],
            "long_term_actions": [r.to_dict() for r in long_term],
            "total_actions": len(all_actions),
            "action_list": all_actions,
        }

    def add_custom_recommendation(
        self,
        failure_type: FailureType,
        recommendation: Recommendation,
    ) -> None:
        """
        Add a custom recommendation for a failure type.

        Args:
            failure_type: The failure type to add recommendation for
            recommendation: The recommendation to add
        """
        self.recommendations[failure_type] = recommendation
        logger.info(f"Added custom recommendation for {failure_type.value}")

    def get_recommendation_summary(
        self,
        recommendations: List[Recommendation],
    ) -> str:
        """
        Generate a human-readable summary of recommendations.

        Args:
            recommendations: List of recommendations

        Returns:
            Summary string
        """
        if not recommendations:
            return "No recommendations at this time."

        lines = ["Recommendations Summary:"]
        prioritized = self.prioritize(recommendations)

        for rec in prioritized[:5]:  # Top 5
            lines.append(
                f"  [{rec.priority.value.upper()}] {rec.title}: "
                f"{rec.impact_estimate}"
            )

        return "\n".join(lines)
