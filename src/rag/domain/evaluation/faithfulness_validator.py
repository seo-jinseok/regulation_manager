"""
FaithfulnessValidator for validating answer groundedness in context.

Validates that answer claims are supported by retrieved context documents.
Implements claim extraction and context matching for Korean text.

Key features:
- Claim extraction (citations, numbers, contacts)
- Context matching with fuzzy matching
- Faithfulness score calculation (0.0-1.0)
- Threshold-based acceptance (default 0.6)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Pattern

logger = logging.getLogger(__name__)

# Constants for score calculation
DEFAULT_THRESHOLD = 0.6
CLAIM_WEIGHT = 0.7  # Weight for claim verification score
CONTEXT_OVERLAP_WEIGHT = 0.3  # Weight for context keyword overlap score


@dataclass
class FaithfulnessValidationResult:
    """Result of faithfulness validation."""

    score: float  # 0.0 to 1.0
    is_acceptable: bool  # True if score >= threshold
    grounded_claims: List[str] = field(default_factory=list)
    ungrounded_claims: List[str] = field(default_factory=list)
    suggestion: str = ""


class FaithfulnessValidator:
    """
    Validates answer faithfulness by checking claim grounding in context.

    This validator extracts claims from Korean text and verifies them against
    provided context documents. It supports multiple claim types including
    citations, numerical data, and contact information.

    Usage:
        validator = FaithfulnessValidator()
        result = validator.validate_answer(answer, context)
        if result.is_acceptable:
            return answer
        else:
            return result.suggestion

    Attributes:
        CITATION_PATTERNS: Regex patterns for Korean legal citations (제X조)
        DATE_PATTERN: Regex pattern for dates (YYYY년 MM월 DD일)
        PERIOD_PATTERN: Regex pattern for periods (MM월 DD일)
        PERCENTAGE_PATTERN: Regex pattern for percentages (XX%)
        DURATION_PATTERN: Regex pattern for durations (X일/주/개월/년/학기)
        PHONE_PATTERN: Regex pattern for Korean phone numbers
        EMAIL_PATTERN: Regex pattern for email addresses
    """

    # Claim extraction patterns (Korean-specific)
    CITATION_PATTERNS: List[Pattern] = [
        re.compile(r"(학칙|규정|지침|령)\s*제\d+조"),
        re.compile(r"제\d+조"),
    ]

    # Numerical patterns
    DATE_PATTERN: Pattern = re.compile(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일")
    PERIOD_PATTERN: Pattern = re.compile(r"\d{1,2}월\s*\d{1,2}일")
    PERCENTAGE_PATTERN: Pattern = re.compile(r"\d+%")
    DURATION_PATTERN: Pattern = re.compile(
        r"\d+(?:일|주|개월|년|학기)"
    )  # Non-capturing group to capture full duration

    # Contact patterns
    PHONE_PATTERN: Pattern = re.compile(r"\d{2,3}-\d{3,4}-\d{4}")
    EMAIL_PATTERN: Pattern = re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    )

    def __init__(self):
        """Initialize faithfulness validator."""
        logger.info("Initialized FaithfulnessValidator")

    def validate_answer(
        self,
        answer: str,
        context: List[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> FaithfulnessValidationResult:
        """
        Validate answer faithfulness against context.

        Args:
            answer: The answer to validate
            context: List of context documents
            threshold: Minimum acceptable score (default 0.6)

        Returns:
            FaithfulnessValidationResult with score and acceptance status
        """
        # Handle edge cases
        if not answer or not answer.strip():
            return FaithfulnessValidationResult(
                score=0.0,
                is_acceptable=False,
                grounded_claims=[],
                ungrounded_claims=[],
                suggestion="Empty answer provided. No claims to verify.",
            )

        if not context or all(not c.strip() for c in context):
            return FaithfulnessValidationResult(
                score=0.0,
                is_acceptable=False,
                grounded_claims=[],
                ungrounded_claims=[],
                suggestion="No context available for verification.",
            )

        # Extract claims from answer
        answer_claims = self._extract_claims(answer)

        # Check each claim against context
        grounded_claims = []
        ungrounded_claims = []

        for claim in answer_claims:
            if self._check_groundedness(claim, context):
                grounded_claims.append(claim)
            else:
                ungrounded_claims.append(claim)

        # Calculate faithfulness score
        score = self._calculate_score(grounded_claims, ungrounded_claims, answer, context)

        # Determine acceptance
        is_acceptable = score >= threshold

        # Generate suggestion
        suggestion = self._generate_suggestion(
            score, is_acceptable, grounded_claims, ungrounded_claims
        )

        logger.info(
            f"Faithfulness validation: score={score:.3f}, "
            f"grounded={len(grounded_claims)}, ungrounded={len(ungrounded_claims)}"
        )

        return FaithfulnessValidationResult(
            score=round(score, 3),
            is_acceptable=is_acceptable,
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            suggestion=suggestion,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract claims from text using various patterns.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims
        """
        claims = []

        # Extract citations
        for pattern in self.CITATION_PATTERNS:
            claims.extend(pattern.findall(text))

        # Extract dates
        claims.extend(self.DATE_PATTERN.findall(text))

        # Extract periods
        claims.extend(self.PERIOD_PATTERN.findall(text))

        # Extract percentages
        claims.extend(self.PERCENTAGE_PATTERN.findall(text))

        # Extract durations
        claims.extend(self.DURATION_PATTERN.findall(text))

        # Extract phone numbers
        claims.extend(self.PHONE_PATTERN.findall(text))

        # Extract emails
        claims.extend(self.EMAIL_PATTERN.findall(text))

        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)

        return unique_claims

    def _check_groundedness(self, claim: str, context: List[str]) -> bool:
        """
        Check if a claim is grounded in context.

        Args:
            claim: The claim to check
            context: List of context documents

        Returns:
            True if claim is found in context (with fuzzy matching)
        """
        context_text = " ".join(context)

        # Exact match
        if claim in context_text:
            return True

        # Fuzzy match: remove spaces for comparison
        normalized_claim = claim.replace(" ", "")
        normalized_context = context_text.replace(" ", "")

        if normalized_claim in normalized_context:
            return True

        # For citation claims, check article number presence
        # e.g., "제10조" should match if "제10조" appears in context
        article_match = re.search(r"제\d+조", claim)
        if article_match:
            article_num = article_match.group(0)
            if article_num in context_text:
                return True

        # For numerical claims, check for exact number match
        # More strict: require the full claim pattern to match
        if re.search(r"\d+", claim):
            # For duration claims like "30일", "4학기", check exact match
            if re.match(r"^\d+(일|주|개월|년|학기)$", claim):
                # Extract just the number and unit
                numbers = re.findall(r"\d+", claim)
                for num in numbers:
                    # Check if number + unit appears in context
                    if num in context_text:
                        # Verify it's in the same unit context
                        pattern = re.compile(rf"{num}(일|주|개월|년|학기)")
                        if pattern.search(context_text):
                            return True
                return False
            else:
                # For other numerical claims (dates, percentages), check number presence
                numbers = re.findall(r"\d+", claim)
                for num in numbers:
                    if num in context_text:
                        return True

        return False

    def _calculate_score(
        self,
        grounded_claims: List[str],
        ungrounded_claims: List[str],
        answer: str,
        context: List[str],
    ) -> float:
        """
        Calculate faithfulness score.

        Uses claim verification ratio and context overlap.

        Args:
            grounded_claims: List of grounded claims
            ungrounded_claims: List of ungrounded claims
            answer: Original answer
            context: Context documents

        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        total_claims = len(grounded_claims) + len(ungrounded_claims)

        if total_claims == 0:
            # No claims: use context keyword overlap
            return self._calculate_context_overlap(answer, context)

        # Claim verification ratio
        claim_score = len(grounded_claims) / total_claims

        # Context overlap score
        overlap_score = self._calculate_context_overlap(answer, context)

        # Weighted average: 70% claim verification + 30% context overlap
        # Higher weight on claims for better accuracy
        score = (claim_score * CLAIM_WEIGHT) + (overlap_score * CONTEXT_OVERLAP_WEIGHT)

        return max(0.0, min(1.0, score))

    def _calculate_context_overlap(self, answer: str, context: List[str]) -> float:
        """
        Calculate keyword overlap between answer and context.

        Args:
            answer: Answer text
            context: Context documents

        Returns:
            Overlap score between 0.0 and 1.0
        """
        # Extract keywords (2+ consecutive Korean characters)
        korean_pattern = re.compile(r"[가-힣]{2,}")

        answer_keywords = set(korean_pattern.findall(answer))
        context_text = " ".join(context)
        context_keywords = set(korean_pattern.findall(context_text))

        if not answer_keywords:
            return 1.0  # No keywords = neutral

        overlap = len(answer_keywords & context_keywords)
        return overlap / len(answer_keywords)

    def _generate_suggestion(
        self,
        score: float,
        is_acceptable: bool,
        grounded_claims: List[str],
        ungrounded_claims: List[str],
    ) -> str:
        """
        Generate improvement suggestion based on validation result.

        Args:
            score: Faithfulness score
            is_acceptable: Whether score meets threshold
            grounded_claims: List of grounded claims
            ungrounded_claims: List of ungrounded claims

        Returns:
            Human-readable suggestion
        """
        if is_acceptable:
            return f"Answer is well-grounded in context (score: {score:.3f})"

        suggestions = []

        if ungrounded_claims:
            suggestions.append(
                f"Contains {len(ungrounded_claims)} ungrounded claims: "
                f"{', '.join(ungrounded_claims[:3])}"
            )

        if score < 0.3:
            suggestions.append("Consider revising the answer to better match the provided context")
        elif score < 0.6:
            suggestions.append("Some claims need context verification")

        return ". ".join(suggestions) if suggestions else "Review answer for groundedness"
