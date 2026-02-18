"""
Intent Classifier for RAG System.

Classifies user query intent into categories for better response generation.
Implements SPEC-RAG-QUALITY-006: Citation & Context Relevance Enhancement.

Intent Categories:
- PROCEDURE: Queries about how to do something (어떻게, 방법, 신청)
- ELIGIBILITY: Queries about qualifications (받을 수 있어, 가능해, 자격)
- DEADLINE: Queries about time periods (언제까지, 기간, 마감)
- GENERAL: Other queries
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Intent categories for query classification."""

    PROCEDURE = "PROCEDURE"  # 절차: "어떻게", "방법", "신청"
    ELIGIBILITY = "ELIGIBILITY"  # 자격: "받을 수 있어", "가능해", "자격"
    DEADLINE = "DEADLINE"  # 기한: "언제까지", "기간", "마감"
    GENERAL = "GENERAL"  # 일반: Other queries


@dataclass
class IntentClassificationResult:
    """Result of intent classification."""

    category: IntentCategory
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)


class IntentClassifier:
    """
    Classifies user query intent based on keyword matching.

    Features:
    - Keyword-based classification with confidence scoring
    - Support for Korean colloquial expressions
    - Priority-based conflict resolution
    """

    # Keyword patterns for each category with weights
    CATEGORY_KEYWORDS = {
        IntentCategory.PROCEDURE: {
            # High-weight keywords (strong signal)
            "어떻게": 0.9,
            "방법": 0.85,
            "신청": 0.8,
            "절차": 0.85,
            "절차는": 0.9,
            "어떻게해": 0.85,
            "어떡해": 0.85,
            "어떡하": 0.85,
            "해야해": 0.7,
            "해야 되": 0.7,
            "해주세요": 0.65,
            "주세요": 0.5,
            "알려주": 0.5,
            "어디서": 0.6,
            "무엇을": 0.5,
            "뭘": 0.5,
            "어떤 서류": 0.7,
            "준비해야": 0.7,
            "제출": 0.6,
            "등록": 0.5,
            "작성": 0.5,
            "어떻게 하나": 0.85,
            "어떻게 되나": 0.7,
        },
        IntentCategory.ELIGIBILITY: {
            # High-weight keywords (strong signal)
            "받을 수 있": 0.9,
            "받을수있": 0.9,
            "가능해": 0.85,
            "가능한": 0.85,
            "가능하": 0.8,
            "자격": 0.9,
            "자격이": 0.9,
            "조건": 0.8,
            "조건이": 0.8,
            "할 수 있": 0.85,
            "할수있": 0.85,
            "을 수 있": 0.85,  # "들을 수 있", "받을 수 있" etc.
            "를 수 있": 0.85,  # "받을 수 있" alternative ending
            "할 수 있나": 0.9,
            "될까": 0.7,
            "되나요": 0.6,
            "받나요": 0.7,
            "되려면": 0.75,
            "되려면어": 0.8,
            "해당하": 0.65,
            "누가": 0.6,
            "누구": 0.55,
        },
        IntentCategory.DEADLINE: {
            # High-weight keywords (strong signal)
            "언제까지": 0.95,
            "언제": 0.75,
            "기간": 0.8,
            "기한": 0.85,
            "마감": 0.9,
            "마감일": 0.95,
            "까지": 0.6,
            "며칠": 0.65,
            "며칠간": 0.7,
            "언제부터": 0.8,
            "언제까지야": 0.9,
            "날짜": 0.7,
            "일정": 0.6,
            "학기": 0.5,  # Context for period
            "주": 0.4,  # Week context
        },
    }

    # Colloquial pattern mappings
    COLLOQUIAL_PATTERNS = [
        (r"게요$", IntentCategory.PROCEDURE),  # "~게요" ending
        (r"게 되나요", IntentCategory.PROCEDURE),  # "~게 되나요"
        (r"까요\?$", IntentCategory.ELIGIBILITY),  # "~까요?" ending
        (r"되나요\?$", IntentCategory.DEADLINE),  # timing question
    ]

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize IntentClassifier.

        Args:
            confidence_threshold: Minimum confidence to classify as non-GENERAL.
        """
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> IntentClassificationResult:
        """
        Classify the intent of a user query.

        Args:
            query: User query string.

        Returns:
            IntentClassificationResult with category, confidence, and matched keywords.
        """
        if not query or not query.strip():
            return IntentClassificationResult(
                category=IntentCategory.GENERAL,
                confidence=0.0,
                matched_keywords=[],
            )

        query_normalized = query.strip().lower()

        # Score each category
        scores: dict[IntentCategory, float] = {}
        matched_by_category: dict[IntentCategory, List[str]] = {}

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            category_score = 0.0
            category_matches: List[str] = []

            for keyword, weight in keywords.items():
                if keyword in query_normalized:
                    category_score += weight
                    category_matches.append(keyword)

            # Normalize score (cap at 1.0)
            if category_matches:
                # More matches = higher confidence, but diminishing returns
                normalized_score = min(category_score / len(category_matches), 1.0)
                # Boost for multiple matches
                if len(category_matches) > 1:
                    boost = min(len(category_matches) * 0.1, 0.2)
                    normalized_score = min(normalized_score + boost, 1.0)

                scores[category] = normalized_score
                matched_by_category[category] = category_matches

        # Check colloquial patterns
        for pattern, category in self.COLLOQUIAL_PATTERNS:
            if re.search(pattern, query):
                current_score = scores.get(category, 0.0)
                scores[category] = max(current_score, 0.6)
                if category not in matched_by_category:
                    matched_by_category[category] = [pattern]

        # Select best category
        if not scores:
            return IntentClassificationResult(
                category=IntentCategory.GENERAL,
                confidence=0.3,
                matched_keywords=[],
            )

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # If best score is below threshold, classify as GENERAL
        if best_score < self.confidence_threshold:
            return IntentClassificationResult(
                category=IntentCategory.GENERAL,
                confidence=best_score,
                matched_keywords=matched_by_category.get(best_category, []),
            )

        return IntentClassificationResult(
            category=best_category,
            confidence=best_score,
            matched_keywords=matched_by_category.get(best_category, []),
        )

    def classify_batch(self, queries: List[str]) -> List[IntentClassificationResult]:
        """
        Classify multiple queries.

        Args:
            queries: List of query strings.

        Returns:
            List of IntentClassificationResult objects.
        """
        return [self.classify(query) for query in queries]

    def get_category_keywords(self, category: IntentCategory) -> List[str]:
        """
        Get keywords for a specific category.

        Args:
            category: Intent category.

        Returns:
            List of keywords for the category.
        """
        return list(self.CATEGORY_KEYWORDS.get(category, {}).keys())
