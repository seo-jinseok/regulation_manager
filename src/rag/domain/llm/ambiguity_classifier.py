"""
Ambiguity Classifier for Regulation Search Queries.

Detects ambiguity levels and generates disambiguation dialogs for clarification.
Supports REQ-AMB-001 through REQ-AMB-015 from SPEC-RAG-001.

Classification Levels:
- CLEAR (0.0-0.3): Direct query with clear intent
- AMBIGUOUS (0.4-0.7): Some ambiguity, present suggestions
- HIGHLY_AMBIGUOUS (0.8-1.0): Require clarification before search
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ...infrastructure.query_analyzer import Audience, QueryAnalyzer

logger = logging.getLogger(__name__)


class AmbiguityLevel(Enum):
    """Ambiguity classification levels."""

    CLEAR = "clear"  # 0.0-0.3: Direct query with clear intent
    AMBIGUOUS = "ambiguous"  # 0.4-0.7: Some ambiguity, present suggestions
    HIGHLY_AMBIGUOUS = "highly_ambiguous"  # 0.8-1.0: Require clarification


@dataclass
class AmbiguityClassifierConfig:
    """Configuration for ambiguity classifier."""

    high_threshold: float = 0.7  # Threshold for HIGHLY_AMBIGUOUS
    low_threshold: float = 0.4  # Threshold for AMBIGUOUS
    max_options: int = 5  # Maximum disambiguation options


@dataclass
class AmbiguityFactors:
    """Factors contributing to ambiguity."""

    audience: bool = False
    regulation_type: bool = False
    article_reference: bool = False


@dataclass
class ClassificationResult:
    """Result of ambiguity classification."""

    level: AmbiguityLevel
    score: float  # 0.0 to 1.0
    ambiguity_factors: Dict[str, bool] = field(default_factory=dict)
    detected_audiences: List[Audience] = field(default_factory=list)
    clarified_query: Optional[str] = None


@dataclass
class DisambiguationOption:
    """Single disambiguation option."""

    label: str  # Human-readable label (e.g., "학생용 휴학 규정")
    clarified_query: str  # Rewritten query (e.g., "학생 휴학 규정")
    relevance_score: float  # 0.0 to 1.0
    audience: Optional[Audience] = None
    explanation: Optional[str] = None  # Optional explanation


@dataclass
class DisambiguationDialog:
    """Disambiguation dialog for user interaction."""

    message: str  # User-facing message
    options: List[DisambiguationOption]  # Ranked options
    skip_allowed: bool = True  # Whether "skip" option is available


class AmbiguityClassifier:
    """
    Classifies query ambiguity and generates disambiguation dialogs.

    Detects:
    - Audience ambiguity (student vs faculty vs staff)
    - Regulation type ambiguity (academic vs personnel vs general)
    - Article reference ambiguity

    Generates disambiguation dialogs with top N ranked suggestions.

    Usage:
        classifier = AmbiguityClassifier()
        result = classifier.classify("휴학 규정")

        if result.level == AmbiguityLevel.AMBIGUOUS:
            dialog = classifier.generate_disambiguation_dialog(result)
            # Present dialog to user
    """

    # Generic regulation terms that increase ambiguity
    GENERIC_REGULATION_TERMS = [
        "규정",
        "내규",
        "세칙",
        "지침",
        "요강",
        "가이드라인",
    ]

    # Specific regulation patterns (should NOT trigger ambiguity)
    SPECIFIC_REGULATION_PATTERN = re.compile(r"[가-힣]{2,8}(규정|학칙|내규|세칙|지침)")

    # Article reference pattern
    ARTICLE_PATTERN = re.compile(r"제\d+조")

    def __init__(
        self,
        config: Optional[AmbiguityClassifierConfig] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ):
        """
        Initialize ambiguity classifier.

        Args:
            config: Classifier configuration
            query_analyzer: Optional QueryAnalyzer for audience detection
        """
        self.config = config or AmbiguityClassifierConfig()
        self._query_analyzer = query_analyzer or QueryAnalyzer(llm_client=None)
        self._learned_preferences: Dict[str, Audience] = {}

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify query ambiguity level.

        Args:
            query: User query text

        Returns:
            ClassificationResult with level, score, and factors
        """
        if not query or not query.strip():
            return ClassificationResult(
                level=AmbiguityLevel.CLEAR,
                score=0.0,
                ambiguity_factors={},
                detected_audiences=[],
            )

        # Normalize query
        query = unicodedata.normalize("NFC", query.strip())

        # Detect audiences
        detected_audiences = self._query_analyzer.detect_audience_candidates(query)

        # Calculate ambiguity factors
        factors = AmbiguityFactors()

        # Factor 1: Audience ambiguity (multiple matches)
        if len(detected_audiences) > 1:
            factors.audience = True
        elif len(detected_audiences) == 1 and detected_audiences[0] == Audience.ALL:
            # No clear audience detected
            factors.audience = True

        # Factor 2: Regulation type ambiguity (generic terms)
        factors.regulation_type = self._has_generic_regulation_terms(query)

        # Factor 3: Article reference ambiguity
        factors.article_reference = not self._has_article_reference(query)

        # Calculate overall ambiguity score
        score = self._calculate_ambiguity_score(factors, query)

        # Determine level
        if score >= self.config.high_threshold:
            level = AmbiguityLevel.HIGHLY_AMBIGUOUS
        elif score >= self.config.low_threshold:
            level = AmbiguityLevel.AMBIGUOUS
        else:
            level = AmbiguityLevel.CLEAR

        # Build result - only include factors that are True
        ambiguity_factors = {}
        if factors.audience:
            ambiguity_factors["audience"] = True
        if factors.regulation_type:
            ambiguity_factors["regulation_type"] = True
        if factors.article_reference:
            ambiguity_factors["article_reference"] = True

        return ClassificationResult(
            level=level,
            score=score,
            ambiguity_factors=ambiguity_factors,
            detected_audiences=detected_audiences,
        )

    def _has_generic_regulation_terms(self, query: str) -> bool:
        """
        Check if query contains generic regulation terms.

        Returns False if query contains specific regulation name
        (e.g., "교원인사규정", "학칙"), True if only generic terms (e.g., "규정").
        """
        # Check for specific regulation pattern first
        if self.SPECIFIC_REGULATION_PATTERN.search(query):
            return False

        # Check for generic terms only
        return any(term in query for term in self.GENERIC_REGULATION_TERMS)

    def _has_article_reference(self, query: str) -> bool:
        """Check if query contains article reference."""
        return bool(self.ARTICLE_PATTERN.search(query))

    def _calculate_ambiguity_score(
        self, factors: AmbiguityFactors, query: str
    ) -> float:
        """
        Calculate overall ambiguity score from factors.

        Scoring:
        - Base score: 0.0
        - Audience ambiguity: +0.30
        - Regulation type ambiguity: +0.35
        - Missing article reference: +0.10
        - Very short query (< 3 chars): +0.25
        - Single word query: +0.15
        """
        score = 0.0

        if factors.audience:
            score += 0.30

        if factors.regulation_type:
            score += 0.35

        if factors.article_reference:
            score += 0.10

        # Very short queries are more ambiguous
        if len(query) < 3:
            score += 0.25

        # Single word queries are ambiguous
        words = query.split()
        if len(words) == 1:
            score += 0.15

        # Cap at 1.0
        return min(score, 1.0)

    def generate_disambiguation_dialog(
        self, classification: ClassificationResult
    ) -> Optional[DisambiguationDialog]:
        """
        Generate disambiguation dialog for ambiguous queries.

        Args:
            classification: Classification result

        Returns:
            DisambiguationDialog or None if query is CLEAR
        """
        if classification.level == AmbiguityLevel.CLEAR:
            return None

        options = self._generate_disambiguation_options(classification)

        # Sort by relevance score (descending)
        options.sort(key=lambda o: o.relevance_score, reverse=True)

        # Limit to max_options
        options = options[: self.config.max_options]

        # Generate user-facing message
        message = self._generate_disambiguation_message(classification)

        return DisambiguationDialog(
            message=message,
            options=options,
            skip_allowed=True,
        )

    def _generate_disambiguation_options(
        self, classification: ClassificationResult
    ) -> List[DisambiguationOption]:
        """Generate ranked disambiguation options."""
        options: List[DisambiguationOption] = []

        # Check for learned preferences
        original_query = ""  # We don't have original query here, need to track
        if original_query in self._learned_preferences:
            preferred_audience = self._learned_preferences[original_query]
            options.extend(
                self._create_audience_options([preferred_audience], base_score=0.9)
            )

        # Generate options based on detected audiences
        if classification.ambiguity_factors.get("audience"):
            # Check for specific audiences
            if Audience.STUDENT in classification.detected_audiences:
                options.extend(
                    self._create_audience_options([Audience.STUDENT], base_score=0.7)
                )
            if Audience.FACULTY in classification.detected_audiences:
                options.extend(
                    self._create_audience_options([Audience.FACULTY], base_score=0.7)
                )
            if Audience.STAFF in classification.detected_audiences:
                options.extend(
                    self._create_audience_options([Audience.STAFF], base_score=0.7)
                )

            # If all audiences detected, create options for each
            if len(classification.detected_audiences) >= 3:
                all_options = self._create_audience_options(
                    [
                        Audience.STUDENT,
                        Audience.FACULTY,
                        Audience.STAFF,
                    ],
                    base_score=0.5,
                )
                options.extend(all_options)

        # If no options generated, create generic ones
        if not options:
            options = self._create_generic_options(classification)

        return options

    def _create_audience_options(
        self, audiences: List[Audience], base_score: float
    ) -> List[DisambiguationOption]:
        """Create disambiguation options for specific audiences."""
        options: List[DisambiguationOption] = []

        audience_labels = {
            Audience.STUDENT: "학생",
            Audience.FACULTY: "교원",
            Audience.STAFF: "직원",
        }

        for audience in audiences:
            label = audience_labels.get(audience, audience.value)
            options.append(
                DisambiguationOption(
                    label=f"{label} 관련 규정",
                    clarified_query=f"{label} 규정",
                    relevance_score=base_score,
                    audience=audience,
                    explanation=f"{label}에게 적용되는 규정을 찾습니다.",
                )
            )

        return options

    def _create_generic_options(
        self, classification: ClassificationResult
    ) -> List[DisambiguationOption]:
        """Create generic disambiguation options."""
        return [
            DisambiguationOption(
                label="규정 전체 검색",
                clarified_query="규정",
                relevance_score=0.5,
                explanation="모든 규정을 검색합니다.",
            )
        ]

    def _generate_disambiguation_message(
        self, classification: ClassificationResult
    ) -> str:
        """Generate user-facing disambiguation message."""
        if classification.level == AmbiguityLevel.HIGHLY_AMBIGUOUS:
            return "질문이 명확하지 않습니다. 아래 옵션 중에서 원하는 내용을 선택해 주세요."

        if classification.ambiguity_factors.get("audience"):
            return "대상에 따라 규정이 다릅니다. 해당되는 경우를 선택해 주세요."

        if classification.ambiguity_factors.get("regulation_type"):
            return "검색 범위를 좁혀주세요. 더 구체적인 규정 이름을 선택해 주세요."

        return "검색어를 명확히 해 주세요."

    def apply_user_selection(
        self,
        original_query: str,
        selected_option: DisambiguationOption,
    ) -> str:
        """
        Apply user's disambiguation selection to clarify query.

        Args:
            original_query: Original user query
            selected_option: User-selected disambiguation option

        Returns:
            Clarified query text
        """
        # Learn from user selection
        if selected_option.audience:
            self._learned_preferences[original_query] = selected_option.audience

        # Return clarified query from option
        return selected_option.clarified_query

    def skip_clarification(self, query: str) -> str:
        """
        Skip clarification and return original query.

        Args:
            query: Original query

        Returns:
            Original query unchanged
        """
        return query

    def learn_from_selection(self, query: str, audience: Audience) -> None:
        """
        Learn from user disambiguation selection for future classifications.

        Args:
            query: Query text
            audience: User-selected audience
        """
        self._learned_preferences[query] = audience
        logger.debug(
            f"Learned preference for query '{query}': audience={audience.value}"
        )
