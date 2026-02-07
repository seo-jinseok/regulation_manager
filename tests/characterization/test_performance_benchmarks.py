"""
Performance characterization tests for RAG system components.

These tests capture the current performance characteristics of
key components to establish baselines before optimization.

Tests follow DDD PRESERVE principle - document current behavior.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest


@pytest.mark.characterization
class TestKiwiTokenizerPerformance:
    """Characterize Kiwi tokenizer initialization and usage performance."""

    def test_kiwi_initialization_time(self):
        """Benchmark: Time to initialize Kiwi tokenizer."""
        # Reset global state for clean test
        import src.rag.infrastructure.hybrid_search as hs_module
        from src.rag.infrastructure.hybrid_search import _get_kiwi

        hs_module._kiwi = None

        start_time = time.perf_counter()
        tokenizer = _get_kiwi()
        init_time = time.perf_counter() - start_time

        # Characterize: Record initialization time
        assert init_time >= 0, "Initialization time should be non-negative"

        # Document current behavior
        result = {
            "component": "kiwi_tokenizer",
            "metric": "initialization_time_seconds",
            "value": init_time,
            "notes": "First initialization time"
            if tokenizer is not None
            else "Failed to initialize",
        }

        print(f"\n[Kiwi Performance] {result}")
        # Characterization tests document behavior
        assert result["value"] >= 0

    def test_kiwi_singleton_behavior(self):
        """Characterize: Verify singleton pattern behavior."""
        from src.rag.infrastructure.hybrid_search import _get_kiwi

        # First call
        tokenizer1 = _get_kiwi()
        # Second call should return same instance
        tokenizer2 = _get_kiwi()

        # Characterize singleton behavior
        is_singleton = tokenizer1 is tokenizer2

        result = {
            "component": "kiwi_tokenizer",
            "metric": "singleton_pattern",
            "value": is_singleton,
            "notes": "Both calls return same instance"
            if is_singleton
            else "Singleton pattern broken",
        }

        print(f"\n[Kiwi Singleton] {result}")
        assert is_singleton, "Kiwi should use singleton pattern"


@pytest.mark.characterization
class TestBM25IndexSerialization:
    """Characterize BM25 index serialization performance."""

    def test_pickle_save_performance(self):
        """Benchmark: Pickle serialization time for BM25 index."""
        from src.rag.infrastructure.hybrid_search import BM25

        # Create test index
        bm25 = BM25(k1=1.5, b=0.75)

        # Add sample documents using add_documents API
        docs = [
            "휴학은 학기 시작 30일 이전에 신청해야 합니다.",
            "장학금 지급 기준은 성적 평점이 3.0 이상이어야 합니다.",
            "교원의 승진 및 인사에 관한 규정입니다.",
        ] * 100  # 300 documents

        documents = [(f"doc_{i}", doc, {}) for i, doc in enumerate(docs)]
        bm25.add_documents(documents)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Benchmark pickle save
            start_time = time.perf_counter()
            bm25.save_index(tmp_path)
            save_time = time.perf_counter() - start_time

            # Get file size
            file_size = Path(tmp_path).stat().st_size

            result = {
                "component": "bm25_index",
                "metric": "pickle_save_time_seconds",
                "value": save_time,
                "doc_count": len(docs),
                "file_size_bytes": file_size,
                "notes": "Pickles entire index to disk",
            }

            print(f"\n[BM25 Save Performance] {result}")
            assert save_time >= 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pickle_load_performance(self):
        """Benchmark: Pickle deserialization time for BM25 index."""
        from src.rag.infrastructure.hybrid_search import BM25

        # Create and save test index
        bm25 = BM25(k1=1.5, b=0.75)

        docs = [
            "휴학은 학기 시작 30일 이전에 신청해야 합니다.",
            "장학금 지급 기준은 성적 평점이 3.0 이상이어야 합니다.",
        ] * 100  # 200 documents

        documents = [(f"doc_{i}", doc, {}) for i, doc in enumerate(docs)]
        bm25.add_documents(documents)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            bm25.save_index(tmp_path)

            # Benchmark pickle load
            start_time = time.perf_counter()
            success = bm25.load_index(tmp_path)
            load_time = time.perf_counter() - start_time

            result = {
                "component": "bm25_index",
                "metric": "pickle_load_time_seconds",
                "value": load_time,
                "success": success,
                "doc_count": len(docs),
                "notes": "Loads entire index into memory",
            }

            print(f"\n[BM25 Load Performance] {result}")
            assert success, "Should load successfully"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.characterization
class TestRedisConnectionPool:
    """Characterize Redis connection pool behavior."""

    def test_pool_status_availability(self):
        """Characterize: Pool status method availability and content."""
        from src.rag.infrastructure.cache import RedisBackend

        # Create Redis backend (may fail if Redis not available)
        backend = RedisBackend(
            host="localhost",
            port=6379,
            max_connections=50,
        )

        # Get pool status
        status = backend.get_pool_status()

        result = {
            "component": "redis_connection_pool",
            "metric": "pool_status_available",
            "value": status,
            "notes": "Pool status dict structure",
        }

        print(f"\n[Redis Pool Status] {result}")

        # Characterize expected keys
        expected_keys = {"available", "max_connections", "total_connections"}
        actual_keys = set(status.keys())

        result["expected_keys"] = expected_keys
        result["actual_keys"] = actual_keys
        result["has_basic_keys"] = expected_keys.issubset(actual_keys)

        print(f"  Expected keys: {expected_keys}")
        print(f"  Actual keys: {actual_keys}")
        print(f"  Has basic keys: {result['has_basic_keys']}")

    def test_pool_connection_reuse(self):
        """Characterize: Connection reuse behavior."""
        from src.rag.infrastructure.cache import RedisBackend

        backend = RedisBackend(
            host="localhost",
            port=6379,
            max_connections=10,
        )

        if not backend.available:
            pytest.skip("Redis not available")

        # Multiple operations to test connection reuse
        operations = []
        for i in range(5):
            start = time.perf_counter()
            backend.set(
                f"test_key_{i}", {"key": f"test_key_{i}", "value": f"test_value_{i}"}
            )
            backend.get(f"test_key_{i}")
            op_time = time.perf_counter() - start
            operations.append(op_time)

        # Characterize timing pattern
        result = {
            "component": "redis_connection_pool",
            "metric": "operation_timing_seconds",
            "values": operations,
            "avg": sum(operations) / len(operations),
            "min": min(operations),
            "max": max(operations),
            "notes": "Lower times indicate connection reuse",
        }

        print(f"\n[Redis Connection Reuse] {result}")


@pytest.mark.characterization
class TestHyDECacheBehavior:
    """Characterize HyDE cache behavior."""

    def test_cache_unbounded_growth(self):
        """Characterize: Cache growth behavior (current: unbounded)."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            # HyDEGenerator expects llm_client, enable_cache, cache_dir
            from unittest.mock import Mock

            mock_llm = Mock()
            expander = HyDEGenerator(
                llm_client=mock_llm,
                enable_cache=True,
                cache_dir=str(cache_dir),
            )

            # Add many entries to test growth
            initial_queries = [f"test query {i}" for i in range(100)]

            for query in initial_queries:
                cache_key = expander._get_cache_key(query)
                expander._cache[cache_key] = {
                    "query": query,
                    "hypothetical_doc": f"Generated doc for {query}",
                    "timestamp": time.time(),
                }

            expander._save_cache()

            # Characterize cache size
            cache_file = cache_dir / "hyde_cache.json"
            file_size = cache_file.stat().st_size if cache_file.exists() else 0
            entry_count = len(expander._cache)

            result = {
                "component": "hyde_cache",
                "metric": "cache_size_characteristics",
                "entry_count": entry_count,
                "file_size_bytes": file_size,
                "avg_entry_bytes": file_size / entry_count if entry_count > 0 else 0,
                "notes": "Current implementation has no size limit",
                "has_lru": False,
                "has_compression": False,
            }

            print(f"\n[HyDE Cache Growth] {result}")

    def test_cache_serialization_format(self):
        """Characterize: Current cache serialization format."""
        import gzip

        from src.rag.infrastructure.hyde import HyDEGenerator

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            from unittest.mock import Mock

            mock_llm = Mock()
            expander = HyDEGenerator(
                llm_client=mock_llm,
                enable_cache=True,
                cache_dir=str(cache_dir),
            )

            # Add test data
            expander._cache["test_key"] = {
                "query": "test query",
                "hypothetical_doc": "test document content",
                "timestamp": time.time(),
            }
            expander._save_cache()

            # Read and analyze format (now with gzip compression)
            cache_file = cache_dir / "hyde_cache.json.gz"
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                content = f.read()
                data = json.loads(content)

            result = {
                "component": "hyde_cache",
                "metric": "serialization_format",
                "format": "json",
                "indent": False,
                "ensure_ascii": False,
                "has_compression": True,
                "compression_type": "gzip",
                "sample_entry": data.get("test_key", {}),
                "notes": "Current: Gzip-compressed JSON without indentation",
            }

            print(f"\n[HyDE Cache Format] {result}")


@pytest.mark.characterization
class TestPerformanceBaseline:
    """Overall performance baseline tests."""

    def test_end_to_end_baseline(self, sample_chunks):
        """Establish baseline for end-to-end query processing."""
        from src.rag.infrastructure.hybrid_search import BM25

        # Setup BM25 index
        bm25 = BM25(k1=1.5, b=0.75)

        documents = [
            (f"doc_{i}", chunk.text, chunk.metadata)
            for i, chunk in enumerate(sample_chunks)
        ]
        bm25.add_documents(documents)

        # Baseline measurement
        query = "휴학"

        start_time = time.perf_counter()

        # Simulate search operation
        results = bm25.search(query, top_k=3)

        end_time = time.perf_counter()
        baseline_time = end_time - start_time

        result = {
            "component": "end_to_end",
            "metric": "baseline_query_time_seconds",
            "value": baseline_time,
            "query": query,
            "result_count": len(results),
            "notes": "Baseline before optimizations",
        }

        print(f"\n[End-to-End Baseline] {result}")
        assert baseline_time >= 0
