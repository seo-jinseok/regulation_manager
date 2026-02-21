"""
A/B Test Framework for Korean Rerankers.

SPEC-RAG-QUALITY-009 REQ-003: Compare BGE Reranker v2-m3 vs Korean reranker models.

This module provides infrastructure for comparing reranker performance:
- BGE Reranker v2-m3 (multilingual baseline)
- Korean-specific rerankers (Dongjin-kr/kr-reranker, NLPai/ko-reranker)

Features:
- Configurable A/B test ratio
- Metrics logging for comparison
- Performance tracking
- Automatic model selection
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RerankerMetrics:
    """Metrics for a single reranker evaluation."""

    model_name: str
    query: str
    latency_ms: float
    scores: List[float]
    avg_score: float
    min_score: float
    max_score: float
    result_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "query": self.query[:100],  # Truncate for logging
            "latency_ms": round(self.latency_ms, 2),
            "avg_score": round(self.avg_score, 4),
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
            "result_count": self.result_count,
            "timestamp": self.timestamp,
        }


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""

    query: str
    model_a: str
    model_b: str
    metrics_a: RerankerMetrics
    metrics_b: RerankerMetrics
    winner: Optional[str] = None  # "a", "b", or "tie"
    score_diff: float = 0.0
    latency_diff_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query[:100],
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metrics_a": self.metrics_a.to_dict(),
            "metrics_b": self.metrics_b.to_dict(),
            "winner": self.winner,
            "score_diff": round(self.score_diff, 4),
            "latency_diff_ms": round(self.latency_diff_ms, 2),
            "timestamp": self.timestamp,
        }


class RerankerABTest:
    """
    A/B test framework for comparing reranker models.

    SPEC-RAG-QUALITY-009 REQ-003: Optimize Reranker for Korean Regulations.

    Usage:
        ab_test = RerankerABTest(
            model_a="BAAI/bge-reranker-v2-m3",
            model_b="Dongjin-kr/kr-reranker",
            ab_ratio=0.5,
        )

        # Run comparison
        result = ab_test.compare(query, documents, top_k=10)

        # Or run with automatic selection
        selected_results = ab_test.run_with_ab_test(query, documents, top_k=10)
    """

    def __init__(
        self,
        model_a: str = "BAAI/bge-reranker-v2-m3",
        model_b: str = "Dongjin-kr/kr-reranker",
        ab_ratio: float = 0.5,
        metrics_dir: str = ".metrics/reranker_ab",
        enabled: bool = True,
    ):
        """
        Initialize A/B test framework.

        Args:
            model_a: Baseline model (default: BGE v2-m3).
            model_b: Test model (default: Korean reranker).
            ab_ratio: Ratio of queries to route to model_b (0.5 = 50%).
            metrics_dir: Directory to store metrics.
            enabled: Whether A/B testing is enabled.
        """
        self.model_a = model_a
        self.model_b = model_b
        self.ab_ratio = ab_ratio
        self.metrics_dir = Path(metrics_dir)
        self.enabled = enabled

        # Reranker instances (lazy loaded)
        self._reranker_a = None
        self._reranker_b = None

        # Metrics storage
        self._metrics_history: List[ABTestResult] = []
        self._model_a_wins = 0
        self._model_b_wins = 0
        self._ties = 0

        # Ensure metrics directory exists
        if self.enabled:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Reranker A/B test initialized: {model_a} vs {model_b}, "
                f"ratio={ab_ratio}, metrics_dir={metrics_dir}"
            )

    def _get_reranker(self, model_name: str):
        """Get or create reranker instance for the given model."""
        if model_name == self.model_a and self._reranker_a is not None:
            return self._reranker_a
        if model_name == self.model_b and self._reranker_b is not None:
            return self._reranker_b

        # Create new reranker instance
        try:
            if "bge" in model_name.lower() or "bge-reranker" in model_name.lower():
                from .reranker import BGEReranker

                reranker = BGEReranker(model_name=model_name)
            else:
                # Try Korean reranker
                from .reranker import KoreanReranker

                reranker = KoreanReranker(model_name=model_name, use_ab_testing=False)

            # Cache instance
            if model_name == self.model_a:
                self._reranker_a = reranker
            else:
                self._reranker_b = reranker

            return reranker
        except Exception as e:
            logger.warning(f"Failed to load reranker {model_name}: {e}")
            # Fallback to BGE reranker
            from .reranker import BGEReranker

            return BGEReranker()

    def _compute_metrics(
        self,
        model_name: str,
        query: str,
        results: List[Tuple],
        latency_ms: float,
    ) -> RerankerMetrics:
        """Compute metrics for reranker results."""
        if not results:
            return RerankerMetrics(
                model_name=model_name,
                query=query,
                latency_ms=latency_ms,
                scores=[],
                avg_score=0.0,
                min_score=0.0,
                max_score=0.0,
                result_count=0,
            )

        scores = [r[2] for r in results if len(r) > 2]  # Extract score from tuple

        return RerankerMetrics(
            model_name=model_name,
            query=query,
            latency_ms=latency_ms,
            scores=scores,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            min_score=min(scores) if scores else 0.0,
            max_score=max(scores) if scores else 0.0,
            result_count=len(results),
        )

    def compare(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        top_k: int = 10,
    ) -> ABTestResult:
        """
        Compare both rerankers on the same query.

        This runs both models regardless of A/B ratio and returns comparison metrics.
        Useful for evaluation and model selection.

        Args:
            query: Search query.
            documents: List of (doc_id, content, metadata) tuples.
            top_k: Number of results to return.

        Returns:
            ABTestResult with metrics from both models.
        """
        if not documents:
            return ABTestResult(
                query=query,
                model_a=self.model_a,
                model_b=self.model_b,
                metrics_a=RerankerMetrics(
                    model_name=self.model_a,
                    query=query,
                    latency_ms=0,
                    scores=[],
                    avg_score=0,
                    min_score=0,
                    max_score=0,
                    result_count=0,
                ),
                metrics_b=RerankerMetrics(
                    model_name=self.model_b,
                    query=query,
                    latency_ms=0,
                    scores=[],
                    avg_score=0,
                    min_score=0,
                    max_score=0,
                    result_count=0,
                ),
                winner="tie",
            )

        # Run model A
        reranker_a = self._get_reranker(self.model_a)
        start_a = time.time()
        results_a = reranker_a.rerank(query, documents, top_k=top_k)
        latency_a = (time.time() - start_a) * 1000
        metrics_a = self._compute_metrics(self.model_a, query, results_a, latency_a)

        # Run model B
        reranker_b = self._get_reranker(self.model_b)
        start_b = time.time()
        results_b = reranker_b.rerank(query, documents, top_k=top_k)
        latency_b = (time.time() - start_b) * 1000
        metrics_b = self._compute_metrics(self.model_b, query, results_b, latency_b)

        # Determine winner based on average score
        score_diff = metrics_b.avg_score - metrics_a.avg_score
        if score_diff > 0.05:  # 5% threshold
            winner = "b"
            self._model_b_wins += 1
        elif score_diff < -0.05:
            winner = "a"
            self._model_a_wins += 1
        else:
            winner = "tie"
            self._ties += 1

        result = ABTestResult(
            query=query,
            model_a=self.model_a,
            model_b=self.model_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            winner=winner,
            score_diff=score_diff,
            latency_diff_ms=latency_b - latency_a,
        )

        # Store in history
        self._metrics_history.append(result)

        # Log result
        logger.info(
            f"A/B comparison: {self.model_a} avg={metrics_a.avg_score:.3f}ms={latency_a:.0f} "
            f"vs {self.model_b} avg={metrics_b.avg_score:.3f}ms={latency_b:.0f} "
            f"winner={winner}"
        )

        return result

    def run_with_ab_test(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        top_k: int = 10,
        force_model: Optional[str] = None,
    ) -> Tuple[List[Tuple], Optional[str]]:
        """
        Run reranking with A/B test routing.

        Routes queries to model A or B based on ab_ratio configuration.
        Use this in production for gradual rollout.

        Args:
            query: Search query.
            documents: List of (doc_id, content, metadata) tuples.
            top_k: Number of results to return.
            force_model: Force specific model ("a" or "b"), bypassing ratio.

        Returns:
            Tuple of (reranked results, model name used).
        """
        if not documents:
            return [], None

        import random

        # Determine which model to use
        if force_model == "a":
            use_model_b = False
        elif force_model == "b":
            use_model_b = True
        else:
            use_model_b = random.random() < self.ab_ratio

        model_name = self.model_b if use_model_b else self.model_a
        reranker = self._get_reranker(model_name)

        start_time = time.time()
        results = reranker.rerank(query, documents, top_k=top_k)
        latency_ms = (time.time() - start_time) * 1000

        # Log usage
        logger.debug(
            f"A/B routing: model={model_name}, latency={latency_ms:.0f}ms, "
            f"results={len(results)}"
        )

        return results, model_name

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of A/B test results.

        Returns:
            Dictionary with win counts, average metrics, and recommendations.
        """
        if not self._metrics_history:
            return {
                "total_comparisons": 0,
                "model_a_wins": 0,
                "model_b_wins": 0,
                "ties": 0,
                "recommendation": "insufficient_data",
            }

        # Calculate averages
        avg_score_a = sum(r.metrics_a.avg_score for r in self._metrics_history) / len(
            self._metrics_history
        )
        avg_score_b = sum(r.metrics_b.avg_score for r in self._metrics_history) / len(
            self._metrics_history
        )
        avg_latency_a = sum(r.metrics_a.latency_ms for r in self._metrics_history) / len(
            self._metrics_history
        )
        avg_latency_b = sum(r.metrics_b.latency_ms for r in self._metrics_history) / len(
            self._metrics_history
        )

        # Recommendation based on score (latency as tiebreaker)
        if avg_score_b > avg_score_a + 0.03:  # 3% improvement threshold
            recommendation = self.model_b
        elif avg_score_a > avg_score_b + 0.03:
            recommendation = self.model_a
        elif avg_latency_b < avg_latency_a * 0.9:  # 10% faster
            recommendation = self.model_b
        else:
            recommendation = self.model_a  # Default to baseline

        return {
            "total_comparisons": len(self._metrics_history),
            "model_a": self.model_a,
            "model_b": self.model_b,
            "model_a_wins": self._model_a_wins,
            "model_b_wins": self._model_b_wins,
            "ties": self._ties,
            "avg_score_a": round(avg_score_a, 4),
            "avg_score_b": round(avg_score_b, 4),
            "avg_latency_a": round(avg_latency_a, 2),
            "avg_latency_b": round(avg_latency_b, 2),
            "recommendation": recommendation,
        }

    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save A/B test metrics to JSON file.

        Args:
            filename: Optional custom filename.

        Returns:
            Path to saved file.
        """
        if not self._metrics_history:
            logger.warning("No metrics to save")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ab_test_{timestamp}.json"

        filepath = self.metrics_dir / filename

        data = {
            "summary": self.get_summary(),
            "comparisons": [r.to_dict() for r in self._metrics_history[-100:]],  # Last 100
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved A/B test metrics to {filepath}")
        return str(filepath)

    def load_metrics(self, filepath: str) -> None:
        """
        Load A/B test metrics from JSON file.

        Args:
            filepath: Path to metrics file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._model_a_wins = data.get("summary", {}).get("model_a_wins", 0)
        self._model_b_wins = data.get("summary", {}).get("model_b_wins", 0)
        self._ties = data.get("summary", {}).get("ties", 0)

        logger.info(
            f"Loaded A/B test metrics: a_wins={self._model_a_wins}, "
            f"b_wins={self._model_b_wins}, ties={self._ties}"
        )


# Global A/B test manager instance
_ab_test_manager: Optional[RerankerABTest] = None


def get_ab_manager(
    model_a: str = "BAAI/bge-reranker-v2-m3",
    model_b: str = "Dongjin-kr/kr-reranker",
    ab_ratio: float = 0.5,
    metrics_dir: str = ".metrics/reranker_ab",
    enabled: bool = True,
) -> RerankerABTest:
    """
    Get or create the global A/B test manager instance.

    Args:
        model_a: Baseline model name.
        model_b: Test model name.
        ab_ratio: Ratio of queries to route to model_b.
        metrics_dir: Directory for metrics storage.
        enabled: Whether A/B testing is enabled.

    Returns:
        RerankerABTest instance.
    """
    global _ab_test_manager

    if _ab_test_manager is None:
        _ab_test_manager = RerankerABTest(
            model_a=model_a,
            model_b=model_b,
            ab_ratio=ab_ratio,
            metrics_dir=metrics_dir,
            enabled=enabled,
        )

    return _ab_test_manager


def get_ab_test_summary() -> Dict[str, Any]:
    """
    Get summary of A/B test results.

    Returns:
        Dictionary with win counts and recommendations.
    """
    if _ab_test_manager is None:
        return {"error": "A/B test manager not initialized"}

    return _ab_test_manager.get_summary()
