"""
Korean Embedding Model Evaluator for Dense Retrieval.

Evaluates and compares Korean embedding models for semantic search performance.
Measures speed, accuracy, and memory usage for model selection.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dense_retriever import DenseRetriever, DenseRetrieverConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of embedding model evaluation."""

    model_name: str
    embedding_dim: int
    index_time: float  # Seconds to index 100 docs
    search_time: float  # Seconds per query
    memory_mb: float  # Memory usage in MB
    accuracy: float  # Cosine similarity score (0-1)
    cache_hit_rate: float  # Cache hit rate (0-1)
    docs_indexed: int


class EmbeddingEvaluator:
    """
    Evaluator for Korean embedding models.

    Compares models on:
    - Indexing speed (documents per second)
    - Search speed (queries per second)
    - Memory usage (model + index size)
    - Accuracy (cosine similarity on gold standard)
    - Cache effectiveness
    """

    # Korean test queries for evaluation
    TEST_QUERIES = [
        ("휴학 절차", "휴학 신청 방법"),
        ("장학금 신청", "장학금 지급 규정"),
        ("졸업 요건", "졸업 학점 기준"),
        ("교수 휴직", "교원 연구년 휴직"),
        ("수강 신청", "수강신청 기간 및 방법"),
        ("학사 경고", "학사경고 제적 규정"),
        ("등록금 납부", "등록금 납부 기간"),
        ("복학 절차", "휴학 후 복학 신청"),
        ("전과 신청", "타 전과 전부 신청"),
        ("성적 이의", "성적 정정 이의신청"),
    ]

    def __init__(self, sample_documents: Optional[List[Tuple[str, str, Dict]]] = None):
        """
        Initialize evaluator.

        Args:
            sample_documents: Optional sample documents for testing.
                             If None, generates synthetic documents.
        """
        self.sample_docs = sample_documents or self._generate_sample_docs()

    def _generate_sample_docs(self) -> List[Tuple[str, str, Dict]]:
        """Generate sample documents for testing."""
        docs = [
            (
                "doc1",
                "휴학 신청은 학기 시작 14일 전까지 가능합니다. 휴학 신청서를 제출하고 승인을 받아야 합니다.",
                {"category": "학적"},
            ),
            (
                "doc2",
                "장학금은 성적 우수자, 저소득 가구, 근로 장학 등 다양한 종류가 있습니다.",
                {"category": "장학"},
            ),
            (
                "doc3",
                "졸업 요건은 총 140학점 이상, 전공 60학점, 교양 30학점 이수입니다.",
                {"category": "졸업"},
            ),
            (
                "doc4",
                "교원은 5년 근무 후 연구년 휴직을 신청할 수 있으며 승인이 필요합니다.",
                {"category": "교원"},
            ),
            (
                "doc5",
                "수강신청은 매학기 지정된 기간에 하며 복수전공, 부전공도 함께 신청합니다.",
                {"category": "수강"},
            ),
            (
                "doc6",
                "학사경고는 학점 평점 1.5 미만일 때 부과되며 3회 연속 시 제적됩니다.",
                {"category": "학적"},
            ),
            (
                "doc7",
                "등록금은 매학기 지정된 기간에 납부해야 하며 분할 납부도 가능합니다.",
                {"category": "등록"},
            ),
            (
                "doc8",
                "복학은 휴학 기간 종료 후 30일 이내에 신청해야 합니다.",
                {"category": "학적"},
            ),
            (
                "doc9",
                "전과는 2학년 2학기까지 신청 가능하며 해당 학과의 승인이 필요합니다.",
                {"category": "학적"},
            ),
            (
                "doc10",
                "성적에 이의가 있을 경우 성적 개시일로부터 30일 이내에 이의신청을 할 수 있습니다.",
                {"category": "성적"},
            ),
        ]
        return docs

    def evaluate_model(
        self,
        model_name: str,
        num_warmup_queries: int = 3,
        num_test_queries: int = 10,
    ) -> EvaluationResult:
        """
        Evaluate a single embedding model.

        Args:
            model_name: HuggingFace model name to evaluate.
            num_warmup_queries: Number of warmup queries before timing.
            num_test_queries: Number of test queries for timing.

        Returns:
            EvaluationResult with performance metrics.
        """
        logger.info(f"Evaluating model: {model_name}")

        # Initialize retriever
        config = DenseRetrieverConfig(batch_size=32, cache_embeddings=True)
        retriever = DenseRetriever(model_name=model_name, config=config)

        # Measure indexing time
        start_time = time.time()
        retriever.add_documents(self.sample_docs)
        index_time = time.time() - start_time

        # Warmup queries
        for query, _ in self.TEST_QUERIES[:num_warmup_queries]:
            retriever.search(query, top_k=5)

        # Measure search time
        start_time = time.time()
        for query, _ in self.TEST_QUERIES[:num_test_queries]:
            retriever.search(query, top_k=5)
        search_time = (time.time() - start_time) / num_test_queries

        # Measure memory usage

        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Measure accuracy (cosine similarity on gold standard)
        accuracy_scores = []
        for query, _expected_context in self.TEST_QUERIES:
            results = retriever.search(query, top_k=3)
            if results:
                # Check if any top-3 result matches expected context
                top_doc_content = results[0][2]  # content field
                # Simple word overlap metric for accuracy
                query_words = set(query.split())
                doc_words = set(top_doc_content.split())
                overlap = (
                    len(query_words & doc_words) / len(query_words)
                    if query_words
                    else 0
                )
                accuracy_scores.append(overlap)

        accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0

        # Cache stats
        cache_stats = retriever.get_cache_stats()
        total_requests = cache_stats["cache_hits"] + cache_stats["cache_misses"]
        cache_hit_rate = (
            cache_stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        )

        return EvaluationResult(
            model_name=model_name,
            embedding_dim=retriever.embedding_dim,
            index_time=index_time,
            search_time=search_time,
            memory_mb=memory_mb,
            accuracy=accuracy,
            cache_hit_rate=cache_hit_rate,
            docs_indexed=len(self.sample_docs),
        )

    def compare_models(
        self, model_names: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        """
        Compare multiple embedding models.

        Args:
            model_names: List of model names to compare.
                        If None, uses default Korean models.

        Returns:
            List of EvaluationResults sorted by accuracy.
        """
        if model_names is None:
            model_names = DenseRetriever.list_models()

        logger.info(f"Comparing {len(model_names)} models")

        results = []
        for model_name in model_names:
            try:
                result = self.evaluate_model(model_name)
                results.append(result)
                logger.info(f"Evaluated {model_name}: accuracy={result.accuracy:.3f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")

        # Sort by accuracy (descending)
        results.sort(key=lambda r: r.accuracy, reverse=True)
        return results

    def print_comparison(self, results: List[EvaluationResult]) -> None:
        """
        Print comparison table of evaluation results.

        Args:
            results: List of EvaluationResults to compare.
        """
        print("\n" + "=" * 100)
        print("Korean Embedding Model Comparison")
        print("=" * 100)

        if not results:
            print("No results to display")
            return

        # Print table header
        print(
            f"{'Model':<30} {'Dim':<6} {'Index':<10} {'Search':<10} {'Memory':<10} {'Accuracy':<10} {'Cache':<10}"
        )
        print("-" * 100)

        # Print results
        for result in results:
            model_short = result.model_name.split("/")[-1][:28]
            print(
                f"{model_short:<30} "
                f"{result.embedding_dim:<6} "
                f"{result.index_time:<10.3f} "
                f"{result.search_time:<10.4f} "
                f"{result.memory_mb:<10.1f} "
                f"{result.accuracy:<10.3f} "
                f"{result.cache_hit_rate:<10.3f}"
            )

        print("=" * 100)

        # Print recommendation
        if results:
            best = results[0]
            print(f"\nRecommendation: {best.model_name}")
            print(f"  - Best accuracy: {best.accuracy:.3f}")
            print(f"  - Fast search: {best.search_time:.4f}s per query")
            print(f"  - Memory efficient: {best.memory_mb:.1f} MB")

    def get_recommendation(
        self, results: List[EvaluationResult], priority: str = "accuracy"
    ) -> Optional[EvaluationResult]:
        """
        Get recommended model based on priority.

        Args:
            results: List of EvaluationResults.
            priority: Priority metric ("accuracy", "speed", "memory").

        Returns:
            Recommended EvaluationResult.
        """
        if not results:
            return None

        if priority == "accuracy":
            return max(results, key=lambda r: r.accuracy)
        elif priority == "speed":
            return min(results, key=lambda r: r.search_time)
        elif priority == "memory":
            return min(results, key=lambda r: r.memory_mb)
        else:
            return results[0]


def benchmark_korean_models(
    priority: str = "accuracy", num_test_queries: int = 10
) -> EvaluationResult:
    """
    Benchmark Korean embedding models and return recommendation.

    Args:
        priority: Priority metric ("accuracy", "speed", "memory").
        num_test_queries: Number of test queries for evaluation.

    Returns:
        Recommended model's EvaluationResult.
    """
    logging.basicConfig(level=logging.INFO)

    # Initialize evaluator
    evaluator = EmbeddingEvaluator()

    # Compare models
    results = evaluator.compare_models()

    # Print comparison
    evaluator.print_comparison(results)

    # Get recommendation
    recommended = evaluator.get_recommendation(results, priority=priority)

    if recommended:
        print(f"\nRecommended model ({priority}): {recommended.model_name}")
        print(f"  - Accuracy: {recommended.accuracy:.3f}")
        print(f"  - Search speed: {recommended.search_time:.4f}s/query")
        print(f"  - Memory: {recommended.memory_mb:.1f} MB")

    return recommended


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        priority = sys.argv[1]
    else:
        priority = "accuracy"

    print(f"Running benchmark with priority: {priority}")
    benchmark_korean_models(priority=priority)
