"""
Focused tests for hyde module to improve coverage from 75% toward 90%.
Tests target high-value, testable code paths.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.rag.infrastructure.hyde import (
    HyDEGenerator,
    HyDESearcher,
)


# Test cache operations (lines 100-101, 106-113)
def test_cache_load_from_file():
    """Test loading cache from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "hyde_cache.json"
        cache_data = {"abc123": "cached document"}
        cache_file.write_text(json.dumps(cache_data, ensure_ascii=False))

        generator = HyDEGenerator(cache_dir=tmpdir, enable_cache=True)
        assert generator._cache == cache_data


def test_cache_file_not_found():
    """Test cache when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Don't create cache file
        generator = HyDEGenerator(cache_dir=tmpdir, enable_cache=True)
        assert generator._cache == {}


def test_cache_invalid_json():
    """Test cache with invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "hyde_cache.json"
        cache_file.write_text("invalid json")

        generator = HyDEGenerator(cache_dir=tmpdir, enable_cache=True)
        # Should handle gracefully
        assert generator._cache == {}


def test_save_cache_writes_file():
    """Test that cache is saved to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = HyDEGenerator(cache_dir=tmpdir, enable_cache=True)
        generator._cache = {"key": "value"}
        generator._save_cache()

        cache_file = Path(tmpdir) / "hyde_cache.json"
        assert cache_file.exists()

        loaded = json.loads(cache_file.read_text())
        assert loaded == {"key": "value"}


def test_cache_disabled():
    """Test that cache disabled doesn't save."""
    generator = HyDEGenerator(enable_cache=False)
    generator._cache = {"key": "value"}
    generator._save_cache()
    # Should not raise error


def test_cache_write_error():
    """Test that write error is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = HyDEGenerator(cache_dir=tmpdir, enable_cache=True)
        generator._cache = {"key": "value"}

        # Mock open to raise error
        import builtins

        original_open = builtins.open

        def mock_open(*args, **kwargs):
            raise IOError("Permission denied")

        builtins.open = mock_open

        try:
            generator._save_cache()
        finally:
            builtins.open = original_open
            # Should not raise error


# Test generate_hypothetical_doc paths (lines 121, 146-179)
def test_generate_no_llm():
    """Test generate without LLM returns original query."""
    generator = HyDEGenerator(llm_client=None, enable_cache=False)
    result = generator.generate_hypothetical_doc("test query")

    assert result.hypothetical_doc == "test query"
    assert result.from_cache is False


def test_generate_cache_hit():
    """Test cache hit returns cached document."""
    generator = HyDEGenerator(enable_cache=True)
    cache_key = generator._get_cache_key("test query")
    generator._cache[cache_key] = "cached result"

    result = generator.generate_hypothetical_doc("test query")

    assert result.hypothetical_doc == "cached result"
    assert result.from_cache is True
    assert result.cache_key == cache_key


def test_generate_cache_miss():
    """Test cache miss with LLM generates new document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = lambda **kwargs: (
            "This is a long generated document that will be cached"
        )

        generator = HyDEGenerator(
            llm_client=mock_llm, enable_cache=True, cache_dir=tmpdir
        )
        result = generator.generate_hypothetical_doc("cache miss test query")

        assert (
            result.hypothetical_doc
            == "This is a long generated document that will be cached"
        )
        assert result.from_cache is False
        # Long document should be cached (>20 chars)
        assert (
            generator._cache[result.cache_key]
            == "This is a long generated document that will be cached"
        )


def test_generate_short_not_cached():
    """Test that short documents are not cached."""
    import uuid

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = MagicMock()
        # Return a short result that will fail validation (< 20 chars)
        mock_llm.generate.side_effect = lambda **kwargs: "short result"

        generator = HyDEGenerator(
            llm_client=mock_llm, enable_cache=True, cache_dir=tmpdir
        )
        # Use unique query to avoid cache hits from previous test runs
        unique_query = f"test query {uuid.uuid4()}"
        result = generator.generate_hypothetical_doc(unique_query)

        # When validation fails (< 20 chars), original query is used as fallback
        assert result.hypothetical_doc == unique_query
        assert result.from_cache is False
        # Should NOT be cached (only validated results are cached)
        assert result.cache_key not in generator._cache


def test_generate_llm_exception():
    """Test LLM exception returns original query."""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("LLM failed")

    generator = HyDEGenerator(llm_client=mock_llm, enable_cache=False)
    result = generator.generate_hypothetical_doc("test query")

    assert result.hypothetical_doc == "test query"
    assert result.from_cache is False


# Test should_use_hyde (line 224)
def test_should_use_hyde_complexity():
    """Test should_use_hyde with complexity parameter."""
    generator = HyDEGenerator()

    # Simple complexity
    assert generator.should_use_hyde("query", complexity="simple") is False

    # Complex complexity
    assert generator.should_use_hyde("query", complexity="complex") is True

    # Regulatory terms
    assert generator.should_use_hyde("규정 확인", complexity="medium") is False

    # Vague queries
    assert generator.should_use_hyde("학교 가기 싫어", complexity="medium") is True


# Test set_llm_client
def test_set_llm_client():
    """Test set_llm_client method."""
    generator = HyDEGenerator(llm_client=None)
    mock_llm = MagicMock()
    generator.set_llm_client(mock_llm)
    assert generator._llm_client is mock_llm


# Test HyDESearcher._merge_results (lines 306-307)
def test_merge_results():
    """Test HyDESearcher merge results functionality."""
    from src.rag.domain.entities import Chunk, ChunkLevel, SearchResult

    mock_generator = MagicMock()
    mock_store = MagicMock()

    chunks = []
    for i in range(3):
        chunk = Chunk(
            id=f"chunk{i}",
            text=f"result {i}",
            title="",
            parent_path=[],
            rule_code="",
            level=ChunkLevel.ARTICLE,
            embedding_text=f"result {i}",
            full_text=f"result {i}",
            token_count=10,
            keywords=[],
            is_searchable=True,
        )
        chunks.append(chunk)

    hyde_results = [SearchResult(chunk=chunks[0], score=0.9, rank=1)]
    orig_results = [
        SearchResult(chunk=chunks[1], score=0.7, rank=1),
        SearchResult(chunk=chunks[2], score=0.5, rank=2),
    ]

    searcher = HyDESearcher(mock_generator, mock_store)
    merged = searcher._merge_results(hyde_results, orig_results, top_k=10)

    # Should have all 3 results
    assert len(merged) == 3
    # Should be sorted by score descending
    scores = [r.score for r in merged]
    assert scores == sorted(scores, reverse=True)
    assert merged[0].chunk.id == "chunk0"  # Highest score
    assert merged[-1].chunk.id == "chunk2"  # Lowest score
