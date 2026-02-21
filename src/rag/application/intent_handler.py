"""
Intent Handler and Clarification Generator for RAG System.

Implements SPEC-RAG-QUALITY-010 Milestone 4: Query Intent Enhancement.

Provides:
- IntentSearchConfig: Intent-specific search configuration
- IntentHandler: Manages intent-based search configs and clarification
- ClarificationGenerator: Generates clarification questions for ambiguous queries
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional

from src.rag.application.intent_classifier import IntentCategory

logger = logging.getLogger(__name__)


@dataclass
class IntentSearchConfig:
    """
    Configuration for intent-specific search behavior.

    Attributes:
        top_k: Number of documents to retrieve.
        procedure_boost: Boost factor for procedure-related documents.
        eligibility_boost: Boost factor for eligibility-related documents.
        deadline_boost: Boost factor for deadline-related documents.
    """

    top_k: int = 5
    procedure_boost: float = 1.0
    eligibility_boost: float = 1.0
    deadline_boost: float = 1.0


@dataclass
class ClarificationGenerator:
    """
    Generates clarification questions for ambiguous queries.

    Creates contextually relevant follow-up questions when intent
    classification confidence is low.
    """

    # Template patterns for each intent category
    CLARIFICATION_TEMPLATES: ClassVar[Dict[IntentCategory, list[str]]] = {
        IntentCategory.PROCEDURE: [
            "구체적으로 어떤 {topic} 절차가 궁금하신가요? (예: 신청 방법, 필요 서류, 진행 순서)",
            "{topic}과 관련하여 어떤 단계를 알고 싶으신가요?",
            "{topic} 신청 절차의 어느 부분이 궁금하신가요?",
        ],
        IntentCategory.ELIGIBILITY: [
            "{topic} 자격 요건을 확인하고 싶으신가요? 구체적인 조건을 말씀해 주시면 정확히 안내해 드릴 수 있습니다.",
            "{topic} 대상이 어떻게 되시나요? (예: 재학생, 신입생, 대학원생)",
            "{topic}과 관련하여 본인의 상황을 말씀해 주시면 자격 여부를 확인해 드리겠습니다.",
        ],
        IntentCategory.DEADLINE: [
            "{topic}의 어떤 기간이 궁금하신가요? (예: 신청 기간, 처리 기간, 유효 기간)",
            "{topic} 관련하여 언제까지의 정보가 필요하신가요?",
            "구체적으로 어떤 {topic} 일정을 찾고 계신가요?",
        ],
        IntentCategory.GENERAL: [],  # No clarification for GENERAL intent
    }

    def generate(
        self,
        query: str,
        intent: IntentCategory,
        confidence: float,
    ) -> Optional[str]:
        """
        Generate a clarification question if confidence is low.

        Args:
            query: Original user query.
            intent: Classified intent category.
            confidence: Classification confidence score (0.0-1.0).

        Returns:
            Clarification question string, or None if no clarification needed.
        """
        # No clarification for high confidence or GENERAL intent
        if confidence >= 0.5 or intent == IntentCategory.GENERAL:
            return None

        templates = self.CLARIFICATION_TEMPLATES.get(intent, [])
        if not templates:
            return None

        # Extract topic from query (simple heuristic)
        topic = self._extract_topic(query)

        # Select first template (could be randomized for variety)
        template = templates[0]

        # Format template with extracted topic
        clarification = template.format(topic=topic)

        logger.debug(
            f"Generated clarification for intent {intent}: {clarification}"
        )

        return clarification

    def _extract_topic(self, query: str) -> str:
        """
        Extract the main topic from a query.

        Simple heuristic that can be enhanced with NLP later.

        Args:
            query: User query string.

        Returns:
            Extracted topic string.
        """
        # Common regulation-related keywords to preserve
        keywords = [
            "장학금",
            "휴학",
            "복학",
            "자퇴",
            "전과",
            "복수전공",
            "부전공",
            "졸업",
            "등록금",
            "수강신청",
            "성적",
            "학점",
            "학사경고",
            "유급",
            "제적",
            "퇴학",
            "재입학",
            "교환학생",
            "인턴십",
            "현장실습",
            "졸업논문",
            "종합시험",
            "외국어시험",
            "컴퓨터활용능력",
        ]

        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower:
                return keyword

        # Default to generic topic
        return "관련 규정"


@dataclass
class IntentHandler:
    """
    Handles intent-based search configuration and clarification.

    Manages intent-specific search parameters and determines when
    clarification questions should be generated.

    Attributes:
        clarification_threshold: Minimum confidence to skip clarification.
        clarification_generator: Generator instance for clarification questions.
    """

    # Confidence threshold for clarification (below this = generate clarification)
    clarification_threshold: float = 0.5

    # Lazy-loaded clarification generator
    _clarification_generator: Optional[ClarificationGenerator] = field(
        default=None, repr=False
    )

    # Intent-specific search configurations
    INTENT_CONFIGS: ClassVar[Dict[IntentCategory, IntentSearchConfig]] = {
        IntentCategory.PROCEDURE: IntentSearchConfig(
            top_k=8,
            procedure_boost=1.5,
            eligibility_boost=1.0,
            deadline_boost=1.0,
        ),
        IntentCategory.ELIGIBILITY: IntentSearchConfig(
            top_k=6,
            procedure_boost=1.0,
            eligibility_boost=1.3,
            deadline_boost=1.0,
        ),
        IntentCategory.DEADLINE: IntentSearchConfig(
            top_k=5,
            procedure_boost=1.0,
            eligibility_boost=1.0,
            deadline_boost=1.4,
        ),
        IntentCategory.GENERAL: IntentSearchConfig(
            top_k=5,
            procedure_boost=1.0,
            eligibility_boost=1.0,
            deadline_boost=1.0,
        ),
    }

    def get_search_config(self, intent: IntentCategory) -> IntentSearchConfig:
        """
        Get search configuration for a specific intent.

        Args:
            intent: Classified intent category.

        Returns:
            IntentSearchConfig with intent-specific parameters.
        """
        config = self.INTENT_CONFIGS.get(intent, self.INTENT_CONFIGS[IntentCategory.GENERAL])

        logger.debug(f"Search config for intent {intent}: top_k={config.top_k}")

        return config

    def should_generate_clarification(self, confidence: float) -> bool:
        """
        Determine if clarification should be generated based on confidence.

        Args:
            confidence: Classification confidence score (0.0-1.0).

        Returns:
            True if clarification should be generated, False otherwise.
        """
        return confidence < self.clarification_threshold

    def get_clarification_generator(self) -> ClarificationGenerator:
        """
        Get the clarification generator instance.

        Uses singleton pattern for efficiency.

        Returns:
            ClarificationGenerator instance.
        """
        if self._clarification_generator is None:
            self._clarification_generator = ClarificationGenerator()

        return self._clarification_generator
