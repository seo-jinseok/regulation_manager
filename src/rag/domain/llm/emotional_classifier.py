"""
Emotional Query Classifier for Regulation RAG System.

Detects emotional state in user queries to enable empathetic response generation.
Implements REQ-EMO-001 through REQ-EMO-015 from SPEC-RAG-001.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Emotional states for query classification (REQ-EMO-001)."""

    NEUTRAL = "neutral"  # Standard factual response
    SEEKING_HELP = "seeking_help"  # Needs detailed explanation with examples
    DISTRESSED = "distressed"  # Requires empathetic acknowledgment + factual content
    FRUSTRATED = "frustrated"  # Needs step-by-step guidance + calming language


@dataclass(frozen=True)
class EmotionalClassificationResult:
    """Result of emotional classification (REQ-EMO-003)."""

    state: EmotionalState
    confidence: float  # 0.0 to 1.0
    detected_keywords: List[str]
    has_urgency: bool = False
    triggers: List[str] = ()  # Specific emotional triggers found


@dataclass
class EmotionalClassifierConfig:
    """Configuration for emotional classifier."""

    # Confidence thresholds (REQ-EMO-010)
    neutral_threshold: float = 0.3
    seeking_help_threshold: float = 0.5
    frustrated_threshold: float = 0.6
    distressed_threshold: float = 0.7

    # Conflict resolution: use highest intensity (REQ-EMO-011)
    priority_order: Tuple[EmotionalState, ...] = (
        EmotionalState.DISTRESSED,
        EmotionalState.FRUSTRATED,
        EmotionalState.SEEKING_HELP,
        EmotionalState.NEUTRAL,
    )


class EmotionalClassifier:
    """
    Classifies emotional state in user queries.

    Implements emotional intent detection per REQ-EMO-008:
    - Detects emotional keywords (힘들어요, 어떡해요, 답답해요, etc.)
    - Classifies into 4 states: NEUTRAL, SEEKING_HELP, DISTRESSED, FRUSTRATED
    - Handles urgency indicators (급해요, 빨리, 지금)
    - Uses highest intensity when conflicts occur (REQ-EMO-011)

    Priority order (highest to lowest intensity):
    1. DISTRESSED - hardship, hopelessness
    2. FRUSTRATED - confusion, complexity
    3. SEEKING_HELP - explicit guidance requests
    4. NEUTRAL - factual queries
    """

    # Emotion keywords from SPEC REQ-EMO-008
    DISTRESSED_KEYWORDS = [
        "힘들어",
        "힘듭니다",
        "어떡해",
        "어떡해요",
        "답답해",
        "답답해요",
        "포기",
        "포기하고",
        "죽고",
        "살기",
        "버겁",
        "버겁워",
        "감당",
        "감당이",
        "너무 힘",
        "너무힘",
        "아파",
        "아파요",
        "괴로워",
        "괴로워요",
        "심각",
        "심각해",
        "못하",
        "못해",
        "불가",
    ]

    FRUSTRATED_KEYWORDS = [
        "안돼",
        "안돼요",
        "안된",
        "안돼",
        "왜 안",
        "왜안",
        "너무 복잡",
        "너무복잡",
        "이해 안",
        "이해안",
        "이해가 안",
        "모르",
        "몰라",
        "몰라요",
        "어떻게",
        "어떻게해",
        "어떻게 해",
        "방법을",
        "방법을",
        "절차가",
        "절차가",
        "알 수",
        "알수",
        "이해가",
        "이해가안",
    ]

    SEEKING_HELP_KEYWORDS = [
        "알려줘",
        "알려줘요",
        "알려주세요",
        "방법",
        "방법은",
        "방법이",
        "어떻게 해",
        "어떻게해",
        "어떻게 하는",
        "어떻게하는",
        "절차",
        "절차는",
        "절차가",
        "방법 알",
        "방법알",
        "어떻게",
        "어떻게돼",
        "어떻게 돼",
        "방법좀",
        "방법 좀",
        "알려",
        "가르쳐",
        "가르쳐줘",
        "가르쳐주세요",
    ]

    # Urgency indicators (REQ-EMO-009)
    URGENCY_KEYWORDS = ["급해", "급해요", "빨리", "지금", "당장", "급함", "급한"]

    def __init__(self, config: Optional[EmotionalClassifierConfig] = None):
        """
        Initialize EmotionalClassifier.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or EmotionalClassifierConfig()

        # Compile keyword patterns for efficient matching
        self._distressed_pattern = self._compile_pattern(self.DISTRESSED_KEYWORDS)
        self._frustrated_pattern = self._compile_pattern(self.FRUSTRATED_KEYWORDS)
        self._seeking_help_pattern = self._compile_pattern(self.SEEKING_HELP_KEYWORDS)
        self._urgency_pattern = self._compile_pattern(self.URGENCY_KEYWORDS)

    def _compile_pattern(self, keywords: List[str]) -> re.Pattern:
        """Compile regex pattern from keyword list."""
        # Sort by length (longest first) for proper matching
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        pattern = "|".join(re.escape(kw) for kw in sorted_keywords)
        return re.compile(pattern)

    def classify(self, query: str) -> EmotionalClassificationResult:
        """
        Classify emotional state of query.

        Args:
            query: User query text.

        Returns:
            EmotionalClassificationResult with state, confidence, and keywords.
        """
        if not query or not query.strip():
            return EmotionalClassificationResult(
                state=EmotionalState.NEUTRAL,
                confidence=1.0,
                detected_keywords=[],
                has_urgency=False,
            )

        # Normalize query for matching
        normalized = query.strip()

        # Detect emotional keywords
        distressed_matches = self._distressed_pattern.findall(normalized)
        frustrated_matches = self._frustrated_pattern.findall(normalized)
        seeking_help_matches = self._seeking_help_pattern.findall(normalized)
        urgency_matches = self._urgency_pattern.findall(normalized)

        # Collect all detected keywords
        detected_keywords = (
            distressed_matches + frustrated_matches + seeking_help_matches
        )

        # Determine urgency
        has_urgency = len(urgency_matches) > 0

        # Calculate scores for each state
        distressed_score = len(distressed_matches) * 2.0  # Higher weight
        frustrated_score = len(frustrated_matches) * 1.5
        seeking_help_score = len(seeking_help_matches) * 1.0

        # Determine state using priority order (REQ-EMO-011)
        # Higher intensity states take precedence
        state = EmotionalState.NEUTRAL
        confidence = 0.0
        triggers = []

        if distressed_score >= 1.0:
            state = EmotionalState.DISTRESSED
            confidence = min(1.0, distressed_score / 2.0)  # Normalize to 0-1 range
            triggers = distressed_matches
        elif frustrated_score >= 1.0:
            state = EmotionalState.FRUSTRATED
            confidence = min(1.0, frustrated_score / 2.0)
            triggers = frustrated_matches
        elif seeking_help_score >= 1.0:
            state = EmotionalState.SEEKING_HELP
            confidence = min(1.0, seeking_help_score / 2.0)
            triggers = seeking_help_matches
        else:
            state = EmotionalState.NEUTRAL
            confidence = 1.0  # High confidence in neutral
            triggers = []

        # Apply confidence thresholds
        if state == EmotionalState.DISTRESSED:
            min_confidence = self.config.distressed_threshold
        elif state == EmotionalState.FRUSTRATED:
            min_confidence = self.config.frustrated_threshold
        elif state == EmotionalState.SEEKING_HELP:
            min_confidence = self.config.seeking_help_threshold
        else:
            min_confidence = self.config.neutral_threshold

        # If confidence below threshold, downgrade to NEUTRAL (REQ-EMO-010)
        if confidence < min_confidence and state != EmotionalState.NEUTRAL:
            state = EmotionalState.NEUTRAL
            confidence = 1.0

        return EmotionalClassificationResult(
            state=state,
            confidence=confidence,
            detected_keywords=detected_keywords,
            has_urgency=has_urgency,
            triggers=triggers,
        )

    def generate_empathy_prompt(
        self, classification: EmotionalClassificationResult, base_prompt: str
    ) -> str:
        """
        Generate empathy-aware prompt based on emotional classification.

        Implements REQ-EMO-004 through REQ-EMO-007.

        Args:
            classification: Emotional classification result.
            base_prompt: Original prompt for response generation.

        Returns:
            Adapted prompt with empathy tone. Does NOT modify factual content (REQ-EMO-014).
        """
        state = classification.state

        if state == EmotionalState.DISTRESSED:
            # REQ-EMO-004: Prepend empathetic acknowledgment
            empathy_prefix = (
                "사용자가 어려운 상황에 처해있는 것 같습니다. "
                "공감과 위로의 말씀과 함께, 규정에 따른 명확한 정보를 제공해주세요."
            )
            return f"{empathy_prefix}\n\n{base_prompt}"

        elif state == EmotionalState.FRUSTRATED:
            # REQ-EMO-005: Use calming language + step-by-step guidance
            empathy_prefix = (
                "사용자가 복잡한 절차나 이해하기 어려운 부분 때문에 "
                "어려움을 겪고 있는 것 같습니다. "
                "단계별로 명확하고 차분하게 설명해주세요."
            )
            return f"{empathy_prefix}\n\n{base_prompt}"

        elif state == EmotionalState.SEEKING_HELP:
            # REQ-EMO-006: Prioritize clarity over brevity
            empathy_prefix = (
                "사용자가 명확한 안내를 원하고 있습니다. "
                "이해하기 쉽게 자세하게 설명해주세요."
            )
            return f"{empathy_prefix}\n\n{base_prompt}"

        # NEUTRAL: No modification to prompt
        return base_prompt

    def get_emotional_metrics(self) -> Dict[str, int]:
        """
        Get emotional state metrics for monitoring (REQ-EMO-003).

        Returns:
            Dictionary with emotion keyword counts.
        """
        return {
            "distressed_keywords_count": len(self.DISTRESSED_KEYWORDS),
            "frustrated_keywords_count": len(self.FRUSTRATED_KEYWORDS),
            "seeking_help_keywords_count": len(self.SEEKING_HELP_KEYWORDS),
            "urgency_keywords_count": len(self.URGENCY_KEYWORDS),
        }
