"""
Regulation Title Detector for HWPX Parsing

Identifies regulation titles with multi-pattern matching and confidence scoring.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class TitleMatchResult:
    """Result of title detection with confidence scoring."""
    is_title: bool
    title: str
    confidence_score: float  # 0.0 to 1.0
    match_type: str  # 'keyword', 'pattern', 'fuzzy'


class RegulationTitleDetector:
    """
    Detect regulation titles using multi-pattern matching.

    Handles various regulation title patterns:
    - Direct keyword matches (e.g., "학칙", "규정")
    - Pattern-based matches (e.g., "XXX에 관한 규정")
    - Compound patterns (e.g., "시행세칙", "운영규정")
    """

    # Regulation title keywords (highest priority)
    TITLE_KEYWORDS = [
        '규정', '요령', '지침', '세칙', '내규', '학칙', '헌장',
        '기준', '수칙', '준칙', '요강', '운영', '정관',
        '시행세칙', '시행규칙', '운영규정', '관리규정', '처리규정',
        '위원회규정', '센터규정', '연구소규정', '부설규정',
    ]

    # High-confidence patterns (must end with keyword)
    HIGH_CONFIDENCE_PATTERNS = [
        r'(규정|요령|지침|세칙|내규|학칙|헌장|기준|수칙|준칙|요강|운영|정관)$',
        r'시행세칙$', r'시행규칙$', r'운영규정$', r'관리규정$', r'처리규정$',
    ]

    # Medium-confidence patterns (keyword + additional text)
    MEDIUM_CONFIDENCE_PATTERNS = [
        r'.+(에 관한|관련)(규정|요령|지침|세칙)',
        r'.+(규정|요령|지침|세칙)$',
    ]

    # Skip patterns (NOT regulation titles)
    SKIP_PATTERNS = [
        r'^\d+\.',  # Numbered lists
        r'^\d+(?:학|년|월|일|시|분|초)례?\b',  # Number + temporal Korean words (학년, 3학년, 2024년)
        r'^[가-힣]{1,3}\s*[\.·\)]',  # Korean letter prefixes
        r'^이\s*규정집은',  # "이 규정집은..."
        r'^제\s*\d+\s*편\s*$',  # Part markers (only if standalone)
        r'^제\s*\d+\s*장\s*$',  # Chapter markers (only if standalone)
        r'^제\s*\d+\s*절\s*$',  # Section markers (only if standalone)
        r'^제\s*\d+조',  # Article markers
        r'^동의대학교\s*규정집$',  # TOC title (exact match)
        r'^총\s*장',  # TOC elements
        r'^목\s*차',  # TOC elements
        r'^추록',  # TOC elements
        r'^부록',  # TOC elements
        r'^규정집\s*추록',
        r'^규정집\s*관리',
        r'^가제\s*정리',
        r'^비고',
        r'^규정\s*명',
        r'^소관\s*부서',
        r'^페이지',
        r'^▪',
        r'^\s*$',
        r'^학위\s*과정',
        r'^학과간\s*',
        r'^[①-⑮]\s*',
        r'^\d+년\s*\d+월',
        r'^\d+\s*회',
        # More specific university patterns - only standalone headers
        r'^[가-힣]{2}\s*대학$',  # "공과 대학" (only if exactly 2 chars + space + 대학)
        r'^제\s*\d+\s*장\s+\S+',  # Chapter markers with content
    ]

    # False positive patterns (might match keyword but aren't titles)
    FALSE_POSITIVE_PATTERNS = [
        r'이\s*규정',  # "이 규정은..." references
        r'동의대학교\s*규정집',  # Document title
        r'규정\s*관리',  # Administrative text
        r'규정\s*제정',
        r'규정\s*개정',
        r'규정\s*폐지',
        r'규정\s*개정\s*안',
    ]

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the title detector.

        Args:
            min_confidence: Minimum confidence score for positive detection.
        """
        self.min_confidence = min_confidence

        # Compile patterns for efficiency
        self.high_confidence_regex = [
            re.compile(p) for p in self.HIGH_CONFIDENCE_PATTERNS
        ]
        self.medium_confidence_regex = [
            re.compile(p) for p in self.MEDIUM_CONFIDENCE_PATTERNS
        ]
        self.skip_regex = [
            re.compile(p) for p in self.SKIP_PATTERNS
        ]
        self.false_positive_regex = [
            re.compile(p) for p in self.FALSE_POSITIVE_PATTERNS
        ]

    def detect(self, text: str) -> TitleMatchResult:
        """
        Detect if text is a regulation title.

        Args:
            text: Text to analyze.

        Returns:
            TitleMatchResult with detection status and confidence.
        """
        if not text:
            return TitleMatchResult(False, "", 0.0, "empty")

        text = text.strip()

        # Length check
        # Special case: Allow certain very short titles that are commonly used standalone
        standalone_short_titles = {'학칙', '규정', '요령', '세칙', '지침'}
        if text in standalone_short_titles:
            pass  # Continue to other checks
        elif len(text) < 4 or len(text) > 200:
            return TitleMatchResult(False, text, 0.0, "length")
        # Allow 4-character titles ending with keywords or having prefix
        elif len(text) == 4:
            # Accept if it ends with ANY keyword
            if not any(text.endswith(kw) for kw in self.TITLE_KEYWORDS):
                return TitleMatchResult(False, text, 0.0, "too_short_no_keyword")

        # Check skip patterns first
        for pattern in self.skip_regex:
            if pattern.match(text):
                return TitleMatchResult(False, text, 0.0, "skip")

        # Check false positive patterns
        for pattern in self.false_positive_regex:
            if pattern.search(text):
                return TitleMatchResult(False, text, 0.0, "false_positive")

        # High confidence patterns
        for pattern in self.high_confidence_regex:
            if pattern.search(text):
                return TitleMatchResult(True, text, 0.95, "keyword")

        # Medium confidence patterns
        for pattern in self.medium_confidence_regex:
            if pattern.search(text):
                return TitleMatchResult(True, text, 0.75, "pattern")

        # Check for any keyword presence (low confidence)
        for keyword in self.TITLE_KEYWORDS:
            if text.endswith(keyword):
                confidence = 0.6 if len(text) < 30 else 0.5
                return TitleMatchResult(
                    confidence >= self.min_confidence,
                    text,
                    confidence,
                    "keyword"
                )

        return TitleMatchResult(False, text, 0.0, "no_match")

    def is_title(self, text: str) -> bool:
        """
        Quick check if text is a regulation title.

        Args:
            text: Text to check.

        Returns:
            True if text is likely a regulation title.
        """
        result = self.detect(text)
        return result.is_title

    def extract_title(self, text: str) -> Optional[str]:
        """
        Extract clean title from text.

        Args:
            text: Text to extract title from.

        Returns:
            Cleaned title or None.
        """
        result = self.detect(text)
        if result.is_title:
            return result.title
        return None

    def get_confidence(self, text: str) -> float:
        """
        Get confidence score for text being a title.

        Args:
            text: Text to analyze.

        Returns:
            Confidence score 0.0 to 1.0.
        """
        result = self.detect(text)
        return result.confidence_score

    def batch_detect(self, texts: List[str]) -> List[TitleMatchResult]:
        """
        Detect titles in a batch of texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of TitleMatchResult objects.
        """
        return [self.detect(text) for text in texts]


# Singleton instance for convenience
_default_detector: Optional[RegulationTitleDetector] = None


def get_default_detector() -> RegulationTitleDetector:
    """Get or create the default detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = RegulationTitleDetector()
    return _default_detector


def detect_regulation_title(text: str) -> bool:
    """
    Convenience function to detect regulation titles.

    Args:
        text: Text to check.

    Returns:
        True if text is likely a regulation title.
    """
    return get_default_detector().is_title(text)
