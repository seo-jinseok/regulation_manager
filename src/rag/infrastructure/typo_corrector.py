"""
Korean Typo Correction Module for Regulation Search.

Implements a hybrid approach:
1. Rule-based correction for common colloquial patterns
2. SymSpell-based correction for general typos
3. Edit distance-based correction for regulation names
4. LLM fallback for complex cases
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from symspellpy import Verbosity

    from ..domain.repositories import ILLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TypoCorrectionResult:
    """Result of typo correction with metadata."""

    original: str
    corrected: str
    method: str  # "rule", "symspell", "edit_distance", "llm", "none"
    corrections: List[Tuple[str, str]]  # List of (original, corrected) pairs
    confidence: float  # 0.0 ~ 1.0


class TypoCorrector:
    """
    Korean typo correction for regulation search queries.

    Uses a hybrid approach:
    1. Rule-based: Fast correction for common patterns
    2. SymSpell: Syllable-based typo correction
    3. Edit Distance: For regulation/article name correction
    4. LLM Fallback: For complex cases

    Examples:
        "시퍼" -> "싶어"
        "되요" -> "돼요"
        "하가요" -> "하세요"
        "바드려면" -> "바라면"
        "교원인사극정" -> "교원인사규정"
    """

    # ==================== Rule-Based Patterns ====================

    # Common colloquial typo patterns (regex -> replacement)
    RULE_PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Desire expressions (common in student queries)
        (re.compile(r"시퍼$"), "싶어"),
        (re.compile(r"시페$"), "싶어"),
        (re.compile(r"스펴$"), "싶어"),
        # Informal speech patterns
        (re.compile(r"되요"), "돼요"),
        (re.compile(r"되어요"), "돼요"),
        (re.compile(r"하가요"), "하세요"),
        (re.compile(r"해여"), "해요"),
        (re.compile(r"뵈어요"), "뵙어요"),
        (re.compile(r"주여요"), "주세요"),
        (re.compile(r"가여요"), "가요"),
        (re.compile(r"와여요"), "와요"),
        (re.compile(r"자여요"), "자요"),
        # Common misspellings
        (re.compile(r"바드려면"), "바라면"),
        (re.compile(r"바드리"), "바라"),
        (re.compile(r"봐드려"), "봐달라"),
        (re.compile(r"줘드려"), "줄라"),
        (re.compile(r"해드려"), "해달라"),
        (re.compile(r"가드려"), "가달라"),
        # Academic term typos
        (re.compile(r"휴학원"), "휴학"),
        (re.compile(r"복학원"), "복학"),
        (re.compile(r"자퇴원"), "자퇴"),
        (re.compile(r"제적원"), "제적"),
        (re.compile(r"장학금원"), "장학금"),
        # Regulation name typos (extended list)
        (re.compile(r"극정"), "규정"),
        (re.compile(r"귀정"), "규정"),
        (re.compile(r"규정이"), "규정"),
        (re.compile(r"규정을"), "규정"),
        (re.compile(r"학칙이"), "학칙"),
        (re.compile(r"학칙을"), "학칙"),
        (re.compile(r"내규이"), "내규"),
        (re.compile(r"내규을"), "내규"),
        (re.compile(r"세칙이"), "세칙"),
        (re.compile(r"세칙을"), "세칙"),
        (re.compile(r"지침이"), "지침"),
        (re.compile(r"지침을"), "지침"),
        (re.compile(r"정관이"), "정관"),
        (re.compile(r"정관을"), "정관"),
        # Common colloquial variations
        (re.compile(r"할까요"), "할까"),
        (re.compile(r"하나요"), "하나"),
        (re.compile(r"어떻해"), "어떻게"),
        (re.compile(r"어케"), "어떻게"),
        (re.compile(r"뭔가"), "뭔가"),
        (re.compile(r"무슨"), "무엇"),
        # Particle/ending corrections
        (re.compile(r"으로서"), "으로"),
        (re.compile(r"으로써"), "으로"),
        (re.compile(r"에서는"), "에서"),
        (re.compile(r"부터는"), "부터"),
        (re.compile(r"에게는"), "에게"),
        (re.compile(r"한테는"), "한테"),
    ]

    # Common particle/ending normalization patterns
    PARTICLE_NORMALIZATION: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"(\S+)으로서"), r"\1으로"),
        (re.compile(r"(\S+)으로써"), r"\1으로"),
        (re.compile(r"(\S+)에서는"), r"\1에서"),
        (re.compile(r"(\S+)부터는"), r"\1부터"),
        (re.compile(r"(\S+)에게는"), r"\1에게"),
        (re.compile(r"(\S+)한테는"), r"\1한테"),
        # Question ending normalization
        (re.compile(r"(\S+)나요\?*$"), r"\1나"),
        (re.compile(r"(\S+)까요\?*$"), r"\1까"),
        (re.compile(r"(\S+)인가요\?*$"), r"\1인가"),
    ]

    # Regulation name suffixes for edit distance matching
    REGULATION_SUFFIXES = ("규정", "학칙", "내규", "세칙", "지침", "정관")

    def __init__(
        self,
        llm_client: Optional["ILLMClient"] = None,
        regulation_names: Optional[List[str]] = None,
        symspell_dictionary_path: Optional[str] = None,
    ):
        """
        Initialize TypoCorrector.

        Args:
            llm_client: Optional LLM client for fallback correction.
            regulation_names: List of known regulation names for edit distance matching.
            symspell_dictionary_path: Optional path to SymSpell dictionary file.
        """
        self._llm_client = llm_client
        self._regulation_names = set(regulation_names or [])
        self._symspell_checker = None
        self._cache: Dict[str, TypoCorrectionResult] = {}

        # Initialize SymSpell if dictionary path is provided
        if symspell_dictionary_path:
            try:
                from symspellpy import SymSpell

                self._symspell_checker = SymSpell(max_dictionary_edit_distance=2)
                self._symspell_checker.load_dictionary(
                    symspell_dictionary_path, term_index=0, count_index=1
                )
                logger.info(
                    f"Loaded SymSpell dictionary from {symspell_dictionary_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load SymSpell dictionary: {e}")
                self._symspell_checker = None

    def correct(
        self, query: str, use_llm_fallback: bool = True
    ) -> TypoCorrectionResult:
        """
        Correct typos in query using hybrid approach.

        Args:
            query: Input query text.
            use_llm_fallback: Whether to use LLM as final fallback.

        Returns:
            TypoCorrectionResult with corrected text and metadata.
        """
        if not query:
            return TypoCorrectionResult(
                original="", corrected="", method="none", corrections=[], confidence=1.0
            )

        # Normalize NFC
        query = unicodedata.normalize("NFC", query)

        # Check cache
        if query in self._cache:
            return self._cache[query]

        corrections: List[Tuple[str, str]] = []
        corrected = query
        method = "none"

        # Stage 1: Rule-based correction (fast, high confidence for common patterns)
        corrected, rule_corrections = self._apply_rule_based_correction(corrected)
        corrections.extend(rule_corrections)

        if rule_corrections:
            method = "rule"

        # Stage 2: SymSpell correction (syllable-based)
        if self._symspell_checker and not self._is_rule_sufficient(rule_corrections):
            corrected, symspell_corrections = self._apply_symspell_correction(corrected)
            corrections.extend(symspell_corrections)

            if symspell_corrections:
                method = "symspell" if not rule_corrections else "hybrid"

        # Stage 3: Edit distance correction for regulation names
        if self._regulation_names:
            corrected, edit_corrections = self._apply_edit_distance_correction(
                corrected
            )
            corrections.extend(edit_corrections)

            if edit_corrections:
                method = "edit_distance" if not rule_corrections else "hybrid"

        # Stage 4: LLM fallback (if enabled and previous stages insufficient)
        if (
            use_llm_fallback
            and self._llm_client
            and not self._is_correction_sufficient(corrections)
        ):
            corrected, llm_corrections = self._apply_llm_correction(corrected)
            corrections.extend(llm_corrections)

            if llm_corrections:
                method = "llm"

        # Calculate confidence based on method and number of corrections
        confidence = self._calculate_confidence(method, corrections)

        result = TypoCorrectionResult(
            original=query,
            corrected=corrected,
            method=method,
            corrections=corrections,
            confidence=confidence,
        )

        # Cache result
        self._cache[query] = result
        return result

    def _apply_rule_based_correction(
        self, text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply rule-based correction patterns."""
        corrections: List[Tuple[str, str]] = []
        corrected = text

        for pattern, replacement in self.RULE_PATTERNS:
            matches = pattern.findall(corrected)
            if matches:
                for match in matches:
                    if isinstance(match, str):
                        corrected = pattern.sub(replacement, corrected)
                        corrections.append((match, replacement))

        # Apply particle normalization
        for pattern, replacement in self.PARTICLE_NORMALIZATION:
            matches = pattern.findall(corrected)
            if matches:
                for match in matches:
                    if isinstance(match, str) and match != replacement:
                        old_text = match
                        corrected = pattern.sub(replacement, corrected)
                        if old_text != corrected:
                            corrections.append((old_text, replacement))

        return corrected, corrections

    def _apply_symspell_correction(
        self, text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply SymSpell-based correction for Korean text."""
        corrections: List[Tuple[str, str]] = []

        if not self._symspell_checker:
            return text, corrections

        # Split text into tokens for correction
        tokens = text.split()
        corrected_tokens = []

        for token in tokens:
            # Skip short tokens and regulation names
            if len(token) < 2 or token.endswith(self.REGULATION_SUFFIXES):
                corrected_tokens.append(token)
                continue

            # Lookup correction
            suggestions = self._symspell_checker.lookup(
                token, Verbosity.CLOSEST, max_edit_distance=2
            )

            if suggestions and suggestions[0].term != token:
                corrections.append((token, suggestions[0].term))
                corrected_tokens.append(suggestions[0].term)
            else:
                corrected_tokens.append(token)

        return " ".join(corrected_tokens), corrections

    def _apply_edit_distance_correction(
        self, text: str
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply edit distance correction for regulation names."""
        corrections: List[Tuple[str, str]] = []

        if not self._regulation_names:
            return text, corrections

        # Find potential regulation names in text
        tokens = text.split()
        corrected_tokens = []

        for token in tokens:
            # Check if token resembles a regulation name
            if token.endswith(self.REGULATION_SUFFIXES):
                # Find closest match using edit distance
                best_match = self._find_closest_regulation_name(token)

                if best_match and best_match != token:
                    corrections.append((token, best_match))
                    corrected_tokens.append(best_match)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)

        return " ".join(corrected_tokens), corrections

    def _find_closest_regulation_name(
        self, token: str, max_distance: int = 2
    ) -> Optional[str]:
        """Find closest regulation name using edit distance."""
        try:
            import editdistance
        except ImportError:
            # Fallback to built-in implementation
            return self._find_closest_regulation_name_fallback(token, max_distance)

        best_match = None
        best_distance = max_distance + 1

        for reg_name in self._regulation_names:
            distance = editdistance.eval(token, reg_name)
            if distance < best_distance:
                best_distance = distance
                best_match = reg_name

        return best_match

    def _find_closest_regulation_name_fallback(
        self, token: str, max_distance: int = 2
    ) -> Optional[str]:
        """Fallback edit distance implementation using Python."""

        def levenshtein_distance(s1: str, s2: str) -> int:
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        best_match = None
        best_distance = max_distance + 1

        for reg_name in self._regulation_names:
            distance = levenshtein_distance(token, reg_name)
            if distance < best_distance:
                best_distance = distance
                best_match = reg_name

        return best_match

    def _apply_llm_correction(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply LLM-based correction as fallback."""
        corrections: List[Tuple[str, str]] = []

        if not self._llm_client:
            return text, corrections

        try:
            prompt = f"""다음 문장의 오타를 교정하세요. 대학 규정 검색 시스템용이므로 규정명, 학칙 등은 정확한 용어로 교정해주세요.

입력: {text}

교정된 문장만 출력하세요. 설명이나 다른 텍스트를 추가하지 마세요."""

            response = self._llm_client.generate(
                system_prompt="당신은 한국어 오타 교정 전문가입니다.",
                user_message=prompt,
                temperature=0.0,
            )

            # Clean response
            corrected = response.strip()

            # Remove common prefixes
            for prefix in ["교정:", "결과:", "답:", "정답:"]:
                if corrected.startswith(prefix):
                    corrected = corrected[len(prefix) :].strip()

            if corrected and corrected != text:
                corrections.append((text, corrected))
                return corrected, corrections

        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")

        return text, corrections

    def _is_rule_sufficient(self, corrections: List[Tuple[str, str]]) -> bool:
        """Check if rule-based corrections are sufficient."""
        # If we made high-confidence rule corrections, no need for SymSpell
        return len(corrections) > 0

    def _is_correction_sufficient(self, corrections: List[Tuple[str, str]]) -> bool:
        """Check if corrections are sufficient or need LLM fallback."""
        # If we made any corrections, consider it sufficient
        return len(corrections) > 0

    def _calculate_confidence(
        self, method: str, corrections: List[Tuple[str, str]]
    ) -> float:
        """Calculate confidence score for correction result."""
        if method == "none":
            return 1.0  # No correction needed = high confidence

        # Base confidence by method
        method_confidence = {
            "rule": 0.95,
            "symspell": 0.85,
            "edit_distance": 0.90,
            "llm": 0.75,
            "hybrid": 0.90,
        }

        base = method_confidence.get(method, 0.8)

        # Adjust based on number of corrections
        # More corrections = lower confidence
        if len(corrections) > 3:
            base *= 0.9
        elif len(corrections) > 1:
            base *= 0.95

        return min(1.0, base)

    def set_regulation_names(self, regulation_names: List[str]) -> None:
        """Update the list of known regulation names for edit distance matching."""
        self._regulation_names = set(regulation_names)
        # Clear cache to force re-evaluation with new names
        self._cache.clear()

    def clear_cache(self) -> None:
        """Clear the correction cache."""
        self._cache.clear()
