"""
Extended Quality Metrics for RAG Evaluation.

SPEC: SPEC-RAG-EVAL-002
EARS: EARS-U-004 (Latency), EARS-U-005 (Consistency), EARS-U-006 (Citation), EARS-U-007 (Readability)
"""

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Single latency measurement."""
    query: str
    response_time_ms: float
    persona: str = ""
    difficulty_tier: str = ""


@dataclass
class LatencySummary:
    """Aggregated latency statistics."""
    total_queries: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    max_ms: float = 0.0
    slow_queries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatencyTracker:
    """Measures and aggregates response times.

    EARS-U-004: Wall-clock response time measurement with p50/p95/p99 percentiles.
    """

    SLOW_THRESHOLD_MS = 10_000  # 10 seconds

    def __init__(self):
        self._records: List[LatencyRecord] = []

    def measure(
        self,
        query_fn: Callable,
        query: str,
        persona: str = "",
        difficulty_tier: str = "",
        **kwargs,
    ) -> Any:
        """Execute a query function and record its latency.

        Args:
            query_fn: Function to execute (should accept query as first arg)
            query: The query string
            persona: Persona identifier
            difficulty_tier: Difficulty tier (L1-L5)

        Returns:
            The result of query_fn
        """
        start = time.monotonic()
        try:
            result = query_fn(query, **kwargs)
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            self._records.append(LatencyRecord(
                query=query,
                response_time_ms=elapsed_ms,
                persona=persona,
                difficulty_tier=difficulty_tier,
            ))
        return result

    def record(self, query: str, response_time_ms: float, persona: str = "", difficulty_tier: str = ""):
        """Manually record a latency measurement."""
        self._records.append(LatencyRecord(
            query=query,
            response_time_ms=response_time_ms,
            persona=persona,
            difficulty_tier=difficulty_tier,
        ))

    @property
    def records(self) -> List[LatencyRecord]:
        return list(self._records)

    def get_summary(
        self,
        persona: Optional[str] = None,
        difficulty_tier: Optional[str] = None,
    ) -> LatencySummary:
        """Calculate latency percentiles.

        Args:
            persona: Filter by persona (if provided)
            difficulty_tier: Filter by tier (if provided)

        Returns:
            LatencySummary with p50, p95, p99 percentiles
        """
        records = self._records
        if persona:
            records = [r for r in records if r.persona == persona]
        if difficulty_tier:
            records = [r for r in records if r.difficulty_tier == difficulty_tier]

        if not records:
            return LatencySummary()

        times = sorted(r.response_time_ms for r in records)
        n = len(times)

        slow = [
            {"query": r.query, "response_time_ms": r.response_time_ms}
            for r in records
            if r.response_time_ms > self.SLOW_THRESHOLD_MS
        ]

        return LatencySummary(
            total_queries=n,
            p50_ms=self._percentile(times, 50),
            p95_ms=self._percentile(times, 95),
            p99_ms=self._percentile(times, 99),
            mean_ms=sum(times) / n,
            max_ms=times[-1],
            slow_queries=slow,
        )

    @staticmethod
    def _percentile(sorted_values: List[float], pct: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_values):
            return sorted_values[-1]
        d = k - f
        return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


@dataclass
class ConsistencyResult:
    """Result of consistency check for a single query."""
    query: str
    responses: List[str]
    similarity_scores: List[float]
    min_similarity: float
    passed: bool
    diff_highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConsistencyChecker:
    """Checks answer consistency by running same query multiple times.

    EARS-U-005: Execute each query 3 times, compute similarity.
    """

    THRESHOLD = 0.85
    DEFAULT_RUNS = 3

    def __init__(self, similarity_fn: Optional[Callable] = None):
        """Initialize consistency checker.

        Args:
            similarity_fn: Optional function(str, str) -> float for computing similarity.
                          If not provided, uses simple token overlap.
        """
        self._similarity_fn = similarity_fn or self._default_similarity

    @staticmethod
    def _default_similarity(text1: str, text2: str) -> float:
        """Compute simple token overlap similarity (Jaccard-like)."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        if not tokens1 and not tokens2:
            return 1.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    def check(
        self,
        query: str,
        query_fn: Callable[[str], str],
        runs: int = 3,
    ) -> ConsistencyResult:
        """Check consistency by running the same query multiple times.

        Args:
            query: The query to test
            query_fn: Function that takes query and returns answer text
            runs: Number of times to execute (default: 3)

        Returns:
            ConsistencyResult with similarities and pass/fail
        """
        runs = max(2, runs)
        responses = []
        for _ in range(runs):
            try:
                answer = query_fn(query)
                responses.append(answer)
            except Exception as e:
                logger.warning("Consistency check query failed: %s", e)
                responses.append("")

        # Compute pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._similarity_fn(responses[i], responses[j])
                similarities.append(sim)

        min_sim = min(similarities) if similarities else 0.0

        # Generate diff highlights for low similarity
        diff_highlights = []
        if min_sim < self.THRESHOLD and len(responses) >= 2:
            tokens0 = set(responses[0].split())
            tokens1 = set(responses[1].split())
            only_in_0 = tokens0 - tokens1
            only_in_1 = tokens1 - tokens0
            if only_in_0:
                diff_highlights.append(f"Run 1 only: {' '.join(list(only_in_0)[:10])}")
            if only_in_1:
                diff_highlights.append(f"Run 2 only: {' '.join(list(only_in_1)[:10])}")

        return ConsistencyResult(
            query=query,
            responses=responses,
            similarity_scores=similarities,
            min_similarity=min_sim,
            passed=min_sim >= self.THRESHOLD,
            diff_highlights=diff_highlights,
        )


@dataclass
class CitationVerificationResult:
    """Result of citation verification for a response."""
    query: str
    total_citations: int
    verified_citations: int
    hallucinated_citations: List[str]
    citation_existence_rate: float
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CitationVerifier:
    """Verifies that cited regulation articles actually exist.

    EARS-U-006: Check cited articles against vector DB.
    """

    THRESHOLD = 0.95
    # Regex to match Korean regulation citations like 제3조, 제15조의2, 제3조 제2항
    CITATION_PATTERN = re.compile(
        r"제\s*(\d+)\s*조(?:의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
    )

    def __init__(self, db_lookup_fn: Optional[Callable[[str], bool]] = None):
        """Initialize citation verifier.

        Args:
            db_lookup_fn: Function(citation_text) -> bool.
                         Returns True if the cited article exists in DB.
        """
        self._lookup_fn = db_lookup_fn

    def extract_citations(self, text: str) -> List[str]:
        """Extract regulation article citations from text."""
        if not text:
            return []
        matches = self.CITATION_PATTERN.findall(text)
        # Return the full match strings
        full_matches = self.CITATION_PATTERN.finditer(text)
        return [m.group(0).strip() for m in full_matches]

    def verify(self, query: str, response_text: str) -> CitationVerificationResult:
        """Verify that citations in the response actually exist.

        Args:
            query: The original query
            response_text: The response text containing citations

        Returns:
            CitationVerificationResult with existence rate
        """
        citations = self.extract_citations(response_text)

        if not citations:
            return CitationVerificationResult(
                query=query,
                total_citations=0,
                verified_citations=0,
                hallucinated_citations=[],
                citation_existence_rate=1.0,  # No citations to verify
                passed=True,
            )

        verified = 0
        hallucinated = []

        for citation in citations:
            if self._lookup_fn:
                exists = self._lookup_fn(citation)
            else:
                # Without a lookup function, assume all citations exist
                exists = True

            if exists:
                verified += 1
            else:
                hallucinated.append(citation)

        rate = verified / len(citations) if citations else 1.0

        return CitationVerificationResult(
            query=query,
            total_citations=len(citations),
            verified_citations=verified,
            hallucinated_citations=hallucinated,
            citation_existence_rate=rate,
            passed=rate >= self.THRESHOLD,
        )


@dataclass
class ReadabilityResult:
    """Result of readability analysis."""
    query: str
    structure_score: float  # 0-1: has headers, bullets, numbered lists
    length_appropriateness: float  # 0-1: right length for query complexity
    language_quality: float  # 0-1: proper Korean, no encoding issues
    overall_score: float
    passed: bool
    feedback: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReadabilityScorer:
    """Evaluates response readability and structure.

    EARS-U-007: Structural organization, length appropriateness, Korean quality.
    """

    THRESHOLD = 0.70

    # Expected response lengths by query type
    MIN_LENGTH_SIMPLE = 50
    MAX_LENGTH_SIMPLE = 500
    MIN_LENGTH_COMPLEX = 100
    MAX_LENGTH_COMPLEX = 2000

    def score(
        self,
        query: str,
        response: str,
        is_complex: bool = False,
    ) -> ReadabilityResult:
        """Score response readability.

        Args:
            query: Original query
            response: Response text
            is_complex: Whether the query is complex (affects length expectations)

        Returns:
            ReadabilityResult with scores and feedback
        """
        feedback = []

        # 1. Structure score
        structure_score = self._score_structure(response, feedback)

        # 2. Length appropriateness
        length_score = self._score_length(response, is_complex, feedback)

        # 3. Language quality
        lang_score = self._score_language_quality(response, feedback)

        # Overall weighted average
        overall = 0.4 * structure_score + 0.3 * length_score + 0.3 * lang_score

        return ReadabilityResult(
            query=query,
            structure_score=structure_score,
            length_appropriateness=length_score,
            language_quality=lang_score,
            overall_score=overall,
            passed=overall >= self.THRESHOLD,
            feedback=feedback,
        )

    def _score_structure(self, response: str, feedback: List[str]) -> float:
        """Score structural organization."""
        if not response:
            feedback.append("Empty response")
            return 0.0

        score = 0.5  # Base score for having content
        lines = response.split("\n")

        # Check for headers (markdown or plain)
        has_headers = any(
            line.strip().startswith("#") or line.strip().startswith("**") and line.strip().endswith("**")
            for line in lines
        )
        if has_headers:
            score += 0.15

        # Check for bullet points or numbered lists
        has_lists = any(
            re.match(r"^\s*[-*•]\s", line) or re.match(r"^\s*\d+[.)]\s", line)
            for line in lines
        )
        if has_lists:
            score += 0.15

        # Check for paragraph breaks (multiple newlines)
        if "\n\n" in response:
            score += 0.1

        # Penalize single-line blob for long responses
        if len(response) > 200 and len(lines) <= 2:
            score -= 0.2
            feedback.append("Long response lacks paragraph breaks")

        if not has_headers and not has_lists and len(response) > 300:
            feedback.append("Consider adding headers or bullet points for better readability")

        return max(0.0, min(1.0, score))

    def _score_length(self, response: str, is_complex: bool, feedback: List[str]) -> float:
        """Score response length appropriateness."""
        length = len(response)

        if is_complex:
            min_len, max_len = self.MIN_LENGTH_COMPLEX, self.MAX_LENGTH_COMPLEX
        else:
            min_len, max_len = self.MIN_LENGTH_SIMPLE, self.MAX_LENGTH_SIMPLE

        if min_len <= length <= max_len:
            return 1.0
        elif length < min_len:
            ratio = length / min_len if min_len > 0 else 0.0
            feedback.append(f"Response too short ({length} chars, expected >= {min_len})")
            return max(0.0, ratio)
        else:
            # Gradually penalize for being too long
            excess = length - max_len
            penalty = min(0.5, excess / max_len)
            feedback.append(f"Response may be too long ({length} chars)")
            return max(0.5, 1.0 - penalty)

    def _score_language_quality(self, response: str, feedback: List[str]) -> float:
        """Score Korean language quality."""
        if not response:
            return 0.0

        score = 1.0

        # Check for encoding artifacts
        encoding_artifacts = re.findall(r"[â€™â€œâ€\x00-\x08\x0b\x0c\x0e-\x1f]", response)
        if encoding_artifacts:
            score -= 0.3
            feedback.append("Encoding artifacts detected")

        # Check for proper Korean sentence endings
        sentences = re.split(r"[.!?]\s", response)
        if sentences:
            last_sentence = sentences[-1].strip()
            if last_sentence and not re.search(r"[다요죠세까임음][\.\!\?]?$", last_sentence):
                score -= 0.1

        # Check for mixed encoding (common issue)
        if re.search(r"ã[가-힣]", response):
            score -= 0.2
            feedback.append("Mixed encoding detected")

        return max(0.0, min(1.0, score))
