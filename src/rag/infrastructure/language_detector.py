"""
Language Detection Utility for RAG System.

Detects the language of a query (Korean, English, or Mixed) to enable
appropriate processing for multilingual query handling.

SPEC-RAG-Q-011 Phase 4: Multilingual Optimization for International Persona.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class QueryLanguage(Enum):
    """Detected language of a query."""

    KOREAN = "korean"
    ENGLISH = "english"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Result of language detection with details."""

    language: QueryLanguage
    korean_ratio: float  # 0.0 to 1.0
    english_ratio: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    def is_korean_dominant(self) -> bool:
        """Check if Korean is the dominant language."""
        return self.korean_ratio > 0.5

    def is_english_dominant(self) -> bool:
        """Check if English is the dominant language."""
        return self.english_ratio > 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "language": self.language.value,
            "korean_ratio": round(self.korean_ratio, 3),
            "english_ratio": round(self.english_ratio, 3),
            "confidence": round(self.confidence, 3),
        }


class LanguageDetector:
    """
    Detects language in queries for multilingual handling.

    Uses simple heuristics based on character ranges to determine
    if a query is primarily Korean, English, or a mix of both.

    This is a lightweight implementation focused on Korean/English detection.
    For comprehensive multilingual support, consider integrating a proper
    language detection library.
    """

    # Korean character ranges
    # Hangul Syllables: U+AC00 to U+D7A3 (11,172 characters)
    # Hangul Jamo: U+1100 to U+11FF
    # Hangul Compatibility Jamo: U+3130 to U+318F
    KOREAN_PATTERN = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+")

    # Basic Latin letters (A-Z, a-z)
    ENGLISH_PATTERN = re.compile(r"[a-zA-Z]+")

    # Common Korean-English academic term mappings
    # Used for query expansion when English terms are detected
    KOREAN_ENGLISH_MAPPINGS = {
        # Admission/Enrollment
        "admission": "입학",
        "enrollment": "등록",
        "registration": "수강신청",
        # Academic Status
        "leave": "휴학",
        "absence": "휴학",
        "withdrawal": "자퇴",
        "expulsion": "제적",
        "dismissal": "제적",
        "return": "복학",
        "reinstatement": "복학",
        # Transfer
        "transfer": "전과",
        # Graduation
        "graduation": "졸업",
        "degree": "학위",
        "thesis": "논문",
        "dissertation": "논문",
        # Grades
        "grade": "성적",
        "gpa": "학점",
        "credit": "학점",
        "transcript": "성적증명서",
        # Scholarship
        "scholarship": "장학금",
        "tuition": "등록금",
        "fee": "수업료",
        # Faculty
        "professor": "교수",
        "faculty": "교원",
        "instructor": "강사",
        # Courses
        "course": "수강",
        "class": "수업",
        "lecture": "강의",
        "syllabus": "강의계획서",
        # International
        "visa": "비자",
        "international": "외국인",
        "foreign": "외국인",
        # Procedures
        "application": "신청",
        "deadline": "기한",
        "deadline": "마감",
        "period": "기간",
        "procedure": "절차",
        "process": "절차",
        "requirement": "요건",
        "requirements": "요건",
        "eligibility": "자격",
        "document": "서류",
        "form": "양식",
        # Regulations
        "regulation": "규정",
        "rule": "규칙",
        "policy": "정책",
        "bylaw": "학칙",
    }

    # English to Korean expansion for common query patterns
    ENGLISH_QUERY_EXPANSIONS = {
        # How-to patterns
        "how to apply": ["신청", "방법", "절차"],
        "how to get": ["받는 방법", "신청", "절차"],
        "what is": ["정의", "설명", "안내"],
        "when is": ["언제", "기간", "일정"],
        "where is": ["위치", "장소", "어디"],
        # Deadline patterns
        "due date": ["마감", "기한", "까지"],
        "deadline": ["마감", "기한", "까지"],
        "due": ["기한", "마감"],
        # Academic terms
        "leave of absence": ["휴학", "휴학신청"],
        "study abroad": ["해외교류", "교환학생"],
        "exchange program": ["교환학생", "해외교류"],
    }

    def __init__(self, korean_threshold: float = 0.3, english_threshold: float = 0.3):
        """
        Initialize the language detector.

        Args:
            korean_threshold: Minimum ratio of Korean chars to consider as Korean-dominant
            english_threshold: Minimum ratio of English chars to consider as English-dominant
        """
        self.korean_threshold = korean_threshold
        self.english_threshold = english_threshold

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the given text.

        Args:
            text: Input text to analyze

        Returns:
            LanguageDetectionResult with detected language and ratios
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                language=QueryLanguage.UNKNOWN,
                korean_ratio=0.0,
                english_ratio=0.0,
                confidence=1.0,
            )

        # Count Korean and English characters
        korean_chars = self.KOREAN_PATTERN.findall(text)
        english_chars = self.ENGLISH_PATTERN.findall(text)

        total_korean = sum(len(chars) for chars in korean_chars)
        total_english = sum(len(chars) for chars in english_chars)
        total_alpha = total_korean + total_english

        if total_alpha == 0:
            return LanguageDetectionResult(
                language=QueryLanguage.UNKNOWN,
                korean_ratio=0.0,
                english_ratio=0.0,
                confidence=0.5,
            )

        korean_ratio = total_korean / total_alpha
        english_ratio = total_english / total_alpha

        # Determine primary language
        language, confidence = self._determine_language(
            korean_ratio, english_ratio
        )

        return LanguageDetectionResult(
            language=language,
            korean_ratio=korean_ratio,
            english_ratio=english_ratio,
            confidence=confidence,
        )

    def _determine_language(
        self, korean_ratio: float, english_ratio: float
    ) -> Tuple[QueryLanguage, float]:
        """
        Determine the language based on ratios.

        Args:
            korean_ratio: Ratio of Korean characters
            english_ratio: Ratio of English characters

        Returns:
            Tuple of (language, confidence)
        """
        # Clear Korean dominance
        if korean_ratio >= 0.7:
            return QueryLanguage.KOREAN, 0.95

        # Clear English dominance
        if english_ratio >= 0.7:
            return QueryLanguage.ENGLISH, 0.95

        # Mixed language
        if korean_ratio >= self.korean_threshold and english_ratio >= self.english_threshold:
            # Both significant - it's mixed
            # Confidence based on how balanced they are
            balance = min(korean_ratio, english_ratio) / max(korean_ratio, english_ratio, 0.01)
            return QueryLanguage.MIXED, 0.7 + (balance * 0.2)

        # Slight Korean dominance
        if korean_ratio > english_ratio:
            return QueryLanguage.KOREAN, 0.7 + korean_ratio * 0.2

        # Slight English dominance
        if english_ratio > korean_ratio:
            return QueryLanguage.ENGLISH, 0.7 + english_ratio * 0.2

        # Default to unknown
        return QueryLanguage.UNKNOWN, 0.5

    def get_korean_equivalent(self, english_term: str) -> List[str]:
        """
        Get Korean equivalents for an English term.

        Args:
            english_term: English term to translate

        Returns:
            List of Korean equivalents (empty if not found)
        """
        term_lower = english_term.lower().strip()

        # Direct mapping
        if term_lower in self.KOREAN_ENGLISH_MAPPINGS:
            return [self.KOREAN_ENGLISH_MAPPINGS[term_lower]]

        # Check for multi-word patterns
        for pattern, korean_terms in self.ENGLISH_QUERY_EXPANSIONS.items():
            if pattern in term_lower:
                return korean_terms

        return []

    def expand_english_query(self, query: str) -> List[str]:
        """
        Expand an English query with Korean equivalents.

        Args:
            query: English query text

        Returns:
            List of expansion terms (Korean equivalents)
        """
        expansions = []

        # Extract English words
        english_words = self.ENGLISH_PATTERN.findall(query)

        for word in english_words:
            korean_equivs = self.get_korean_equivalent(word)
            expansions.extend(korean_equivs)

        # Check for phrase patterns
        query_lower = query.lower()
        for pattern, korean_terms in self.ENGLISH_QUERY_EXPANSIONS.items():
            if pattern in query_lower:
                expansions.extend(korean_terms)

        # Remove duplicates while preserving order
        seen = set()
        unique_expansions = []
        for term in expansions:
            if term not in seen:
                seen.add(term)
                unique_expansions.append(term)

        return unique_expansions

    def is_english_query(self, text: str) -> bool:
        """
        Quick check if a query is primarily in English.

        Args:
            text: Input text to check

        Returns:
            True if the query is primarily English
        """
        result = self.detect(text)
        return result.language in (QueryLanguage.ENGLISH, QueryLanguage.MIXED) and result.is_english_dominant()

    def is_korean_query(self, text: str) -> bool:
        """
        Quick check if a query is primarily in Korean.

        Args:
            text: Input text to check

        Returns:
            True if the query is primarily Korean
        """
        result = self.detect(text)
        return result.language in (QueryLanguage.KOREAN, QueryLanguage.MIXED) and result.is_korean_dominant()

    def get_detected_language(self, text: str) -> QueryLanguage:
        """
        Get the detected language enum value.

        Args:
            text: Input text to analyze

        Returns:
            QueryLanguage enum value
        """
        return self.detect(text).language
