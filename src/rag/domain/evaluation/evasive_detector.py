"""
EvasiveResponseDetector for detecting evasive response patterns.

Detects when the LLM generates evasive responses instead of using
available context information. Supports regeneration decisions
to improve answer quality.

SPEC-RAG-QUALITY-010: Milestone 5 - Evasive Response Detection

Key features:
- 5 evasive pattern detection (homepage, department, context, regulation, vague)
- Context relevance checking
- Regeneration decision logic
- Confidence scoring
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Confidence threshold for high-confidence evasive detection
HIGH_CONFIDENCE_THRESHOLD = 0.7
MULTI_PATTERN_BOOST = 0.1


@dataclass
class EvasivePattern:
    """
    Definition of an evasive response pattern.

    Attributes:
        name: Unique identifier for the pattern
        pattern: Regex pattern to match
        description: Human-readable description
        severity: Pattern severity (high, medium, low)
    """

    name: str
    pattern: str
    description: str
    severity: str

    def __post_init__(self):
        """Validate pattern after initialization."""
        if self.severity not in ("high", "medium", "low"):
            raise ValueError(
                f"severity must be 'high', 'medium', or 'low', got '{self.severity}'"
            )
        # Compile regex pattern for efficiency
        self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """Check if pattern matches the given text."""
        return bool(self._compiled_pattern.search(text))


@dataclass
class EvasiveDetectionResult:
    """
    Result of evasive response detection.

    Attributes:
        is_evasive: Whether evasive patterns were detected
        detected_patterns: List of detected pattern names
        context_has_info: Whether context contains relevant information
        confidence: Confidence score (0.0-1.0)
    """

    is_evasive: bool
    detected_patterns: List[str] = field(default_factory=list)
    context_has_info: bool = False
    confidence: float = 0.0

    def __post_init__(self):
        """Validate result after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


class EvasiveResponseDetector:
    """
    Detects evasive response patterns in LLM answers.

    This detector identifies when the LLM generates evasive responses
    (deflecting to homepage, department, denying context availability)
    instead of utilizing available context information.

    Usage:
        detector = EvasiveResponseDetector()
        result = detector.detect(answer, context)
        if detector.should_regenerate(result):
            # Trigger regeneration with hint

    The five evasive patterns detected are:
    1. Homepage Deflection: "홈페이지.*참고", "홈페이지.*확인"
    2. Department Deflection: "관련 부서.*문의", "담당 부서.*연락"
    3. Context Denial: "제공된.*컨텍스트.*확인.*없", "컨텍스트에서.*찾을.*없"
    4. Regulation Denial: "규정에서.*확인.*없", "규정에.*명시.*없"
    5. Vague Confirmation: "정확한.*확인.*바랍니다", "자세한.*확인.*필요"
    """

    # Define the five evasive patterns (SPEC-RAG-QUALITY-010 TASK-007)
    EVASIVE_PATTERNS: List[EvasivePattern] = [
        # Pattern 1: Homepage Deflection
        EvasivePattern(
            name="homepage_deflection",
            pattern=r"홈페이지.*참고|홈페이지.*확인|홈페이지.*방문",
            description="Deflects to homepage for information",
            severity="high",
        ),
        # Pattern 2: Department Deflection
        EvasivePattern(
            name="department_deflection",
            pattern=r"관련\s*부서.*문의|담당\s*부서.*연락|해당\s*부서.*문의|부서.*확인",
            description="Deflects to department contact",
            severity="high",
        ),
        # Pattern 3: Context Denial
        EvasivePattern(
            name="context_denial",
            pattern=r"제공된.*컨텍스트.*확인.*없|컨텍스트.*찾을.*없|컨텍스트에.*없",
            description="Denies having information in context",
            severity="high",
        ),
        # Pattern 4: Regulation Denial
        EvasivePattern(
            name="regulation_denial",
            pattern=r"규정.*확인.*없|규정.*명시.*없|규정.*명시.*않|규정.*찾을.*없|규정에.*없|규정에.*않",
            description="Denies information in regulations",
            severity="high",
        ),
        # Pattern 5: Vague Confirmation
        EvasivePattern(
            name="vague_confirmation",
            pattern=r"정확한.*확인.*바랍니다|자세한.*확인.*필요|직접.*확인.*바랍니다|확인.*필요",
            description="Uses vague confirmation requiring verification",
            severity="medium",
        ),
    ]

    # Severity weights for confidence calculation
    SEVERITY_WEIGHTS = {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.5,
    }

    def __init__(self):
        """Initialize evasive response detector."""
        logger.info(
            f"Initialized EvasiveResponseDetector with {len(self.EVASIVE_PATTERNS)} patterns"
        )

    def detect(
        self, answer: str, context: List[str]
    ) -> EvasiveDetectionResult:
        """
        Detect evasive patterns in the answer.

        Args:
            answer: The LLM-generated answer to analyze
            context: List of context documents available

        Returns:
            EvasiveDetectionResult with detection details
        """
        # Handle edge cases
        if not answer or not answer.strip():
            return EvasiveDetectionResult(
                is_evasive=False,
                detected_patterns=[],
                context_has_info=False,
                confidence=0.0,
            )

        # Check context availability
        context_has_info = self._check_context_has_info(context)

        # Detect patterns
        detected_patterns = []
        detected_pattern_objects = []

        for pattern in self.EVASIVE_PATTERNS:
            if pattern.matches(answer):
                detected_patterns.append(pattern.name)
                detected_pattern_objects.append(pattern)
                logger.debug(f"Detected evasive pattern: {pattern.name}")

        # Calculate confidence
        is_evasive = len(detected_patterns) > 0
        confidence = self._calculate_confidence(
            detected_pattern_objects, context_has_info
        )

        logger.info(
            f"Evasive detection: is_evasive={is_evasive}, "
            f"patterns={detected_patterns}, confidence={confidence:.3f}"
        )

        return EvasiveDetectionResult(
            is_evasive=is_evasive,
            detected_patterns=detected_patterns,
            context_has_info=context_has_info,
            confidence=confidence,
        )

    def should_regenerate(self, result: EvasiveDetectionResult) -> bool:
        """
        Determine if regeneration should be triggered.

        Regeneration is triggered when:
        1. Evasive patterns are detected
        2. Context contains relevant information
        3. Confidence is above threshold

        Args:
            result: EvasiveDetectionResult from detect()

        Returns:
            True if regeneration should be attempted
        """
        if not result.is_evasive:
            return False

        if not result.context_has_info:
            # Don't regenerate if context doesn't have info
            # The evasive response might be justified
            return False

        # Regenerate if evasive with context available
        return True

    def get_context_relevance(self, answer: str, context: List[str]) -> float:
        """
        Calculate relevance between answer keywords and context.

        Uses Korean keyword extraction to measure overlap between
        answer keywords and context content.

        Args:
            answer: The answer text
            context: List of context documents

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not context or all(not c.strip() for c in context):
            return 0.0

        if not answer or not answer.strip():
            return 0.0

        # Extract Korean keywords (2+ consecutive Korean characters)
        korean_pattern = re.compile(r"[가-힣]{2,}")

        answer_keywords = set(korean_pattern.findall(answer))
        context_text = " ".join(context)
        context_keywords = set(korean_pattern.findall(context_text))

        if not answer_keywords:
            return 0.5  # Neutral if no keywords to compare

        overlap = len(answer_keywords & context_keywords)
        return overlap / len(answer_keywords)

    def _check_context_has_info(self, context: List[str]) -> bool:
        """
        Check if context contains meaningful information.

        Args:
            context: List of context documents

        Returns:
            True if context has usable information
        """
        if not context:
            return False

        # Check if any context document has meaningful content
        for doc in context:
            if doc and doc.strip():
                # Check for Korean content (indicates actual regulation content)
                if re.search(r"[가-힣]", doc):
                    return True

        return False

    def _calculate_confidence(
        self,
        detected_patterns: List[EvasivePattern],
        context_has_info: bool,
    ) -> float:
        """
        Calculate confidence score for evasive detection.

        Higher confidence when:
        - Multiple patterns detected
        - Patterns are high severity
        - Context has information (justifying regeneration)

        Args:
            detected_patterns: List of detected EvasivePattern objects
            context_has_info: Whether context contains information

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not detected_patterns:
            return 0.0

        # Base confidence from highest severity pattern
        max_severity_weight = max(
            self.SEVERITY_WEIGHTS.get(p.severity, 0.5)
            for p in detected_patterns
        )

        # Boost for multiple patterns
        pattern_boost = min(
            (len(detected_patterns) - 1) * MULTI_PATTERN_BOOST, 0.2
        )

        # Calculate base confidence
        confidence = min(max_severity_weight + pattern_boost, 1.0)

        # Adjust based on context availability
        if context_has_info:
            # Higher confidence when context has info (justifies regeneration)
            confidence = min(confidence * 1.1, 1.0)

        return round(confidence, 3)

    def get_regeneration_hint(
        self, result: EvasiveDetectionResult
    ) -> str:
        """
        Generate hint for regeneration prompt.

        When regeneration is triggered, this hint can be injected
        into the prompt to guide the LLM to use available context.

        Args:
            result: EvasiveDetectionResult from detect()

        Returns:
            Hint string for regeneration prompt
        """
        if not result.detected_patterns:
            return ""

        patterns_str = ", ".join(result.detected_patterns)
        hint = (
            f"다음 정보가 컨텍스트에 있습니다: {patterns_str}. "
            f"컨텍스트의 정보를 사용하여 직접 답변해 주세요."
        )

        return hint
