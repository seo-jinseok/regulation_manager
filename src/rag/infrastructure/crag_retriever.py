"""
Corrective RAG (CRAG) Retriever - Cycle 9 Optimization.

Implements advanced retrieval evaluation and correction mechanisms:
1. Optimized relevance score calculation with multiple signals
2. T-Fix re-retrieval trigger conditions with adaptive thresholds
3. Enhanced document re-ranking logic
4. Performance metrics tracking
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..domain.entities import SearchResult

if TYPE_CHECKING:
    from ..domain.repositories import IHybridSearcher, ILLMClient
    from .query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


class RetrievalQuality(Enum):
    """Quality classification of retrieval results."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"


@dataclass
class CRAGMetrics:
    """
    Performance metrics for CRAG operations (Cycle 9).

    Tracks:
    - Relevance evaluation performance
    - T-Fix trigger rates and success
    - Re-ranking effectiveness
    - Timing breakdown
    """

    total_evaluations: int = 0
    excellent_count: int = 0
    good_count: int = 0
    adequate_count: int = 0
    poor_count: int = 0

    tfix_triggered: int = 0
    tfix_successful: int = 0
    tfix_failed: int = 0

    rerank_performed: int = 0
    rerank_improved: int = 0
    avg_rank_change: float = 0.0

    total_evaluation_time_ms: float = 0.0
    total_tfix_time_ms: float = 0.0
    total_rerank_time_ms: float = 0.0

    def record_evaluation(self, quality: RetrievalQuality, eval_time_ms: float) -> None:
        """Record a quality evaluation."""
        self.total_evaluations += 1
        self.total_evaluation_time_ms += eval_time_ms

        if quality == RetrievalQuality.EXCELLENT:
            self.excellent_count += 1
        elif quality == RetrievalQuality.GOOD:
            self.good_count += 1
        elif quality == RetrievalQuality.ADEQUATE:
            self.adequate_count += 1
        else:
            self.poor_count += 1

    def record_tfix(self, successful: bool, tfix_time_ms: float) -> None:
        """Record a T-Fix (re-retrieval) attempt."""
        self.tfix_triggered += 1
        self.total_tfix_time_ms += tfix_time_ms
        if successful:
            self.tfix_successful += 1
        else:
            self.tfix_failed += 1

    def record_rerank(self, improved: bool, rank_change: float, rerank_time_ms: float) -> None:
        """Record a re-ranking operation."""
        self.rerank_performed += 1
        self.total_rerank_time_ms += rerank_time_ms
        if improved:
            self.rerank_improved += 1
        if self.rerank_performed == 1:
            self.avg_rank_change = rank_change
        else:
            self.avg_rank_change = (
                self.avg_rank_change * (self.rerank_performed - 1) + rank_change
            ) / self.rerank_performed

    @property
    def poor_rate(self) -> float:
        """Calculate rate of poor quality retrievals."""
        if self.total_evaluations == 0:
            return 0.0
        return self.poor_count / self.total_evaluations

    @property
    def tfix_success_rate(self) -> float:
        """Calculate T-Fix success rate."""
        if self.tfix_triggered == 0:
            return 0.0
        return self.tfix_successful / self.tfix_triggered

    @property
    def avg_evaluation_time_ms(self) -> float:
        """Calculate average evaluation time."""
        if self.total_evaluations == 0:
            return 0.0
        return self.total_evaluation_time_ms / self.total_evaluations

    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "CRAG Metrics Summary (Cycle 9):",
            f"  Total evaluations: {self.total_evaluations}",
            "",
            "Quality Distribution:",
            f"  Excellent: {self.excellent_count}",
            f"  Good: {self.good_count}",
            f"  Adequate: {self.adequate_count}",
            f"  Poor: {self.poor_count} ({self.poor_rate:.1%})",
            "",
            "T-Fix (Re-retrieval):",
            f"  Triggered: {self.tfix_triggered}",
            f"  Success rate: {self.tfix_success_rate:.1%}",
            "",
            "Performance:",
            f"  Avg evaluation time: {self.avg_evaluation_time_ms:.2f}ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_evaluations": self.total_evaluations,
            "excellent_count": self.excellent_count,
            "good_count": self.good_count,
            "adequate_count": self.adequate_count,
            "poor_count": self.poor_count,
            "poor_rate": self.poor_rate,
            "tfix_triggered": self.tfix_triggered,
            "tfix_successful": self.tfix_successful,
            "tfix_failed": self.tfix_failed,
            "tfix_success_rate": self.tfix_success_rate,
            "rerank_performed": self.rerank_performed,
            "rerank_improved": self.rerank_improved,
            "avg_rank_change": self.avg_rank_change,
            "avg_evaluation_time_ms": self.avg_evaluation_time_ms,
        }


class CRAGRetriever:
    """
    Corrective RAG Retriever with optimized evaluation and correction.

    Features:
    - Multi-signal relevance scoring (top score, keyword overlap, diversity)
    - Adaptive T-Fix thresholds based on query complexity
    - Enhanced document re-ranking with quality signals
    - Comprehensive performance metrics
    """

    DEFAULT_THRESHOLDS = {
        "simple": 0.35,
        "medium": 0.45,
        "complex": 0.55,
    }

    TOP_SCORE_WEIGHT = 0.5
    KEYWORD_WEIGHT = 0.3
    DIVERSITY_WEIGHT = 0.2

    def __init__(
        self,
        hybrid_searcher: Optional["IHybridSearcher"] = None,
        query_analyzer: Optional["QueryAnalyzer"] = None,
        llm_client: Optional["ILLMClient"] = None,
        thresholds: Optional[Dict[str, float]] = None,
        enable_tfix: bool = True,
        enable_rerank: bool = True,
        max_tfix_attempts: int = 2,
    ):
        """
        Initialize CRAG Retriever.

        Args:
            hybrid_searcher: Hybrid search service for re-retrieval
            query_analyzer: Query analyzer for correction strategies
            llm_client: LLM for advanced relevance evaluation
            thresholds: Custom relevance thresholds
            enable_tfix: Enable T-Fix re-retrieval
            enable_rerank: Enable document re-ranking
            max_tfix_attempts: Maximum re-retrieval attempts
        """
        self._hybrid_searcher = hybrid_searcher
        self._query_analyzer = query_analyzer
        self._llm_client = llm_client
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.enable_tfix = enable_tfix
        self.enable_rerank = enable_rerank
        self._max_tfix_attempts = max_tfix_attempts

        self.metrics = CRAGMetrics()

    def evaluate_retrieval_quality(
        self,
        query: str,
        results: List[SearchResult],
        complexity: str = "medium",
    ) -> Tuple[RetrievalQuality, float]:
        """
        Evaluate retrieval quality with optimized scoring.

        Args:
            query: Search query
            results: Search results to evaluate
            complexity: Query complexity ("simple", "medium", "complex")

        Returns:
            Tuple of (quality_classification, relevance_score)
        """
        start_time = time.time()

        if not results:
            quality = RetrievalQuality.POOR
            score = 0.0
        else:
            score = self._calculate_relevance_score(query, results)
            threshold = self._thresholds.get(complexity, self._thresholds["medium"])

            if score >= threshold + 0.15:
                quality = RetrievalQuality.EXCELLENT
            elif score >= threshold + 0.05:
                quality = RetrievalQuality.GOOD
            elif score >= threshold:
                quality = RetrievalQuality.ADEQUATE
            else:
                quality = RetrievalQuality.POOR

        eval_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_evaluation(quality, eval_time_ms)

        logger.debug(
            f"CRAG evaluation: quality={quality.value}, score={score:.3f}, "
            f"complexity={complexity}, time={eval_time_ms:.2f}ms"
        )

        return quality, score

    def _calculate_relevance_score(
        self, query: str, results: List[SearchResult]
    ) -> float:
        """
        Calculate multi-signal relevance score.

        Combines:
        1. Top result score (primary signal)
        2. Keyword overlap (semantic matching)
        3. Result diversity (source coverage)
        """
        if not results:
            return 0.0

        top_score = results[0].score if results else 0.0
        keyword_score = self._calculate_keyword_overlap(query, results)
        diversity_score = self._calculate_diversity(results)

        final_score = (
            top_score * self.TOP_SCORE_WEIGHT
            + keyword_score * self.KEYWORD_WEIGHT
            + diversity_score * self.DIVERSITY_WEIGHT
        )

        return min(1.0, max(0.0, final_score))

    def _calculate_keyword_overlap(
        self, query: str, results: List[SearchResult]
    ) -> float:
        """Calculate keyword overlap between query and results."""
        if not results:
            return 0.0

        query_terms = set(self._tokenize(query))
        if not query_terms:
            return 0.5

        match_count = 0
        check_results = results[:3]

        for result in check_results:
            result_text = f"{result.chunk.title} {result.chunk.text[:200]}"
            result_terms = set(self._tokenize(result_text))

            if query_terms & result_terms:
                overlap_ratio = len(query_terms & result_terms) / len(query_terms)
                if overlap_ratio > 0.3:
                    match_count += 1

        return match_count / len(check_results)

    def _calculate_diversity(self, results: List[SearchResult]) -> float:
        """Calculate result diversity based on unique sources."""
        if not results:
            return 0.0

        unique_regs = set()
        unique_levels = set()

        for result in results[:5]:
            if result.chunk.rule_code:
                base_code = result.chunk.rule_code.split("-")[0]
                unique_regs.add(base_code)
            unique_levels.add(result.chunk.level.value)

        reg_diversity = min(1.0, len(unique_regs) / 3)
        level_diversity = min(1.0, len(unique_levels) / 4)

        return (reg_diversity + level_diversity) / 2

    def should_trigger_tfix(
        self,
        quality: RetrievalQuality,
        score: float,
        attempt_count: int = 0,
    ) -> bool:
        """Determine if T-Fix (re-retrieval) should be triggered."""
        if not self.enable_tfix:
            return False

        if attempt_count >= self._max_tfix_attempts:
            return False

        if quality == RetrievalQuality.POOR:
            return True

        if quality == RetrievalQuality.ADEQUATE and score < 0.4:
            return True

        return False

    async def apply_tfix(
        self,
        original_query: str,
        original_results: List[SearchResult],
        attempt_count: int = 0,
    ) -> Tuple[List[SearchResult], bool]:
        """Apply T-Fix (Transform-Fix) re-retrieval strategy."""
        start_time = time.time()

        if not self._hybrid_searcher:
            return original_results, False

        corrected_query = self._generate_corrected_query(
            original_query, original_results, attempt_count
        )

        if not corrected_query or corrected_query == original_query:
            tfix_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_tfix(False, tfix_time_ms)
            return original_results, False

        try:
            new_results = await self._hybrid_searcher.search(
                corrected_query,
                top_k=10,
            )

            tfix_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_tfix(True, tfix_time_ms)
            return new_results, True

        except Exception as e:
            logger.error(f"T-Fix re-retrieval failed: {e}")
            tfix_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_tfix(False, tfix_time_ms)
            return original_results, False

    def _generate_corrected_query(
        self,
        original_query: str,
        original_results: List[SearchResult],
        attempt_count: int,
    ) -> Optional[str]:
        """Generate corrected query for re-retrieval."""
        if not self._query_analyzer:
            return None

        try:
            if attempt_count == 0:
                expanded = self._query_analyzer.expand_query(original_query)
                if expanded != original_query:
                    return expanded

            if attempt_count == 1:
                try:
                    rewrite_result = self._query_analyzer.rewrite_query_with_info(
                        original_query
                    )
                    if rewrite_result.rewritten != original_query:
                        return rewrite_result.rewritten
                except Exception:
                    pass

            if attempt_count >= 2:
                terms = self._tokenize(original_query)
                key_terms = [t for t in terms if len(t) >= 2]
                if key_terms:
                    return " ".join(key_terms[:5])

        except Exception as e:
            logger.warning(f"Query correction failed: {e}")

        return None

    def apply_rerank(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Apply enhanced document re-ranking."""
        start_time = time.time()

        if not self.enable_rerank or not results:
            return results

        original_ranks = {r.chunk.id: r.rank for r in results}
        reranked = self._enhanced_rerank(query, results)

        rank_changes = []
        for i, result in enumerate(reranked):
            original_rank = original_ranks.get(result.chunk.id, i)
            rank_changes.append(abs(original_rank - i))

        avg_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0
        improved = avg_change > 0.5

        rerank_time_ms = (time.time() - start_time) * 1000
        self.metrics.record_rerank(improved, avg_change, rerank_time_ms)

        return reranked

    def _enhanced_rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Enhanced re-ranking with quality signals."""
        if not results:
            return results

        query_terms = set(self._tokenize(query))

        reranked_results = []
        for original_rank, result in enumerate(results):
            keyword_boost = self._calculate_keyword_boost(query_terms, result)
            position_penalty = 1.0 / (1.0 + original_rank * 0.1)

            new_score = (
                result.score * 0.7
                + keyword_boost * 0.2
                + position_penalty * 0.1
            )

            reranked_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=min(1.0, new_score),
                    rank=original_rank,
                )
            )

        reranked_results.sort(key=lambda x: x.score, reverse=True)

        for i, result in enumerate(reranked_results):
            result.rank = i

        return reranked_results

    def _calculate_keyword_boost(self, query_terms: set, result: SearchResult) -> float:
        """Calculate keyword matching boost for re-ranking."""
        if not query_terms:
            return 0.5

        result_text = f"{result.chunk.title} {result.chunk.text[:200]}"
        result_terms = set(self._tokenize(result_text))

        if not result_terms:
            return 0.0

        overlap = len(query_terms & result_terms) / len(query_terms)
        return min(1.0, overlap * 2)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for keyword processing."""
        import re

        text = text.lower()
        tokens = re.findall(r"[가-힣]+|[a-z0-9]+", text)

        stopwords = {
            "은", "는", "이", "가", "을", "를", "의", "에", "로", "으로",
            "하다", "되다", "이다", "없다", "있다", "그", "저",
        }

        return [t for t in tokens if len(t) >= 2 and t not in stopwords]


class CRAGPipeline:
    """
    Complete CRAG pipeline integrating evaluation, correction, and re-ranking.
    """

    def __init__(self, retriever: CRAGRetriever):
        """Initialize CRAG pipeline."""
        self.retriever = retriever

    async def search(
        self,
        query: str,
        initial_results: List[SearchResult],
        complexity: str = "medium",
    ) -> List[SearchResult]:
        """Execute complete CRAG pipeline."""
        logger.info(f"CRAG pipeline started for query: {query[:50]}...")

        quality, score = self.retriever.evaluate_retrieval_quality(
            query, initial_results, complexity
        )

        current_results = initial_results

        if self.retriever.should_trigger_tfix(quality, score):
            logger.info(f"T-Fix triggered: quality={quality.value}, score={score:.3f}")

            for attempt in range(self.retriever._max_tfix_attempts):
                new_results, improved = await self.retriever.apply_tfix(
                    query, current_results, attempt
                )

                if improved:
                    current_results = new_results
                    quality, score = self.retriever.evaluate_retrieval_quality(
                        query, current_results, complexity
                    )

                    if quality in [RetrievalQuality.EXCELLENT, RetrievalQuality.GOOD]:
                        logger.info(f"T-Fix successful, stopping after attempt {attempt + 1}")
                        break
                else:
                    break

        if quality in [RetrievalQuality.ADEQUATE, RetrievalQuality.GOOD]:
            current_results = self.retriever.apply_rerank(query, current_results)

        logger.info(
            f"CRAG pipeline completed: final_quality={quality.value}, "
            f"results_count={len(current_results)}"
        )

        return current_results
