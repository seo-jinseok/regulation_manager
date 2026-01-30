"""
Regression tests for HyDE empty query validation.

Tests cover edge cases and validation to prevent issues where empty or
malformed queries cause problems in HyDE generation.

Critical Issue Fixed:
- HyDE now validates empty queries before processing to prevent
  LLM calls with empty or whitespace-only strings.
"""

from unittest.mock import MagicMock

import pytest


class FakeChunk:
    """Fake Chunk for testing."""

    def __init__(
        self,
        id: str,
        text: str,
        title: str = "",
        rule_code: str = "",
    ):
        self.id = id
        self.text = text
        self.title = title
        self.rule_code = rule_code
        self.keywords = []


class FakeSearchResult:
    """Fake SearchResult for testing."""

    def __init__(self, chunk: FakeChunk, score: float, rank: int = 1):
        self.chunk = chunk
        self.score = score
        self.rank = rank


def make_result(
    doc_id: str,
    text: str,
    score: float,
    title: str = "",
    rule_code: str = "",
    rank: int = 1,
) -> FakeSearchResult:
    """Helper to create fake search results."""
    chunk = FakeChunk(id=doc_id, text=text, title=title, rule_code=rule_code)
    return FakeSearchResult(chunk=chunk, score=score, rank=rank)


class FakeLLMClient:
    """Fake LLM client for HyDE generation."""

    def __init__(self, response: str = "가상 문서 내용"):
        self._response = response
        self.call_count = 0
        self.last_query = None

    def generate(self, **kwargs):
        self.call_count += 1
        self.last_query = kwargs.get("user_message", "")
        return self._response


class TestHyDEEmptyQueryValidation:
    """Test suite for HyDE empty query handling."""

    def test_rejects_empty_string(self):
        """Empty string should be rejected gracefully."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)
        result = generator.generate_hypothetical_doc("")

        # Should return empty hypothetical_doc for empty query
        assert result.hypothetical_doc == ""
        assert result.from_cache is False
        assert result.original_query == ""

    def test_rejects_whitespace_only(self):
        """Whitespace-only string should be rejected gracefully."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        whitespace_cases = ["   ", "\n\t", "  \n  ", "\t\t\t"]
        for whitespace in whitespace_cases:
            result = generator.generate_hypothetical_doc(whitespace)
            # Should handle gracefully - either return empty or original query
            assert result.hypothetical_doc in ["", whitespace]
            assert result.from_cache is False

    def test_handles_very_short_query(self):
        """Very short queries (1 char) should be handled gracefully."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        # Single character query
        result = generator.generate_hypothetical_doc("a")

        # Should handle gracefully - return query itself or empty
        assert result.hypothetical_doc in ["", "a"]
        assert result.from_cache is False

    def test_handles_llm_empty_response(self):
        """When LLM returns empty response, should use fallback."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        # Mock LLM that returns empty string
        mock_llm = FakeLLMClient(response="")
        generator = HyDEGenerator(llm_client=mock_llm, enable_cache=False)

        result = generator.generate_hypothetical_doc("test query")

        # Should fallback to original query when LLM returns empty
        assert result.hypothetical_doc == "test query"
        assert result.from_cache is False

    def test_handles_llm_very_short_response(self):
        """When LLM returns too short response, should fallback to original query."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        # Mock LLM that returns very short response
        mock_llm = FakeLLMClient(response="short")
        generator = HyDEGenerator(llm_client=mock_llm, enable_cache=False)

        result = generator.generate_hypothetical_doc("test query")

        # Very short responses (< 20 chars) fallback to original query
        assert result.hypothetical_doc == "test query"
        assert result.from_cache is False

    def test_detects_error_messages_in_response(self):
        """LLM error messages should trigger fallback to original query."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        # Mock LLM that returns Korean error message
        mock_llm = FakeLLMClient(response="죄송합니다, 도움을 드릴 수 없습니다")
        generator = HyDEGenerator(llm_client=mock_llm, enable_cache=False)

        result = generator.generate_hypothetical_doc("test query")

        # Error message is too short, should fallback to original query
        assert result.hypothetical_doc == "test query"
        assert result.from_cache is False

    def test_valid_query_generates_normal_result(self):
        """Valid query should generate hypothetical document normally."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        mock_llm = FakeLLMClient(
            response="교직원의 휴직은 다음 각 호의 사유에 해당하는 경우 신청할 수 있다."
        )
        generator = HyDEGenerator(llm_client=mock_llm, enable_cache=False)

        result = generator.generate_hypothetical_doc("학교에 가기 싫어")

        # Should generate normally
        assert len(result.hypothetical_doc) > 20
        assert result.from_cache is False
        assert mock_llm.call_count == 1

    def test_caching_skips_llm_for_duplicate_queries(self):
        """Cache should skip LLM call for previously generated queries."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        mock_llm = FakeLLMClient(
            response="교직원의 휴직은 다음 각 호의 사유에 해당하는 경우 신청할 수 있다."
        )
        # Use cache_dir=None to avoid file-based cache interference
        generator = HyDEGenerator(
            llm_client=mock_llm, enable_cache=True, cache_dir=None
        )

        query = (
            "unique_test_query_for_cache_12345"  # Unique query to avoid cache conflicts
        )

        # First call - should hit LLM if cache is empty
        result1 = generator.generate_hypothetical_doc(query)
        first_call_count = mock_llm.call_count
        # Note: result1 might be from cache if file cache exists

        # Second call - should use cache
        result2 = generator.generate_hypothetical_doc(query)
        assert mock_llm.call_count == first_call_count  # No additional call
        assert result2.from_cache is True
        assert result2.hypothetical_doc == result1.hypothetical_doc

    def test_no_llm_client_falls_back_to_query(self):
        """Without LLM client, should fallback to original query."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(llm_client=None, enable_cache=False)

        result = generator.generate_hypothetical_doc("test query")

        # Should use original query as fallback
        assert result.hypothetical_doc == "test query"
        assert result.from_cache is False

    def test_llm_exception_falls_back_to_query(self):
        """LLM exceptions should be caught and fallback to original query."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        # Mock LLM that raises exception
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM API error")

        generator = HyDEGenerator(llm_client=mock_llm, enable_cache=False)

        result = generator.generate_hypothetical_doc("test query")

        # Should fallback to original query
        assert result.hypothetical_doc == "test query"
        assert result.from_cache is False


class TestHyDEQualityValidation:
    """Test suite for HyDE output quality validation."""

    def test_should_use_hyde_for_vague_queries(self):
        """HyDE should be recommended for vague/emotional queries."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        vague_queries = [
            "학교에 가기 싫어",
            "쉬고 싶어",
            "어떻게 해야 하나요",
            "가능한가요?",
        ]

        for query in vague_queries:
            assert generator.should_use_hyde(query, complexity="medium"), (
                f"'{query}' should recommend HyDE"
            )

    def test_should_not_use_hyde_for_method_question(self):
        """HyDE should NOT be recommended for method questions (no vague indicator)."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        # "방법이 있을까요?" doesn't have vague indicators
        assert not generator.should_use_hyde("방법이 있을까요?", complexity="medium")

    def test_should_not_use_hyde_for_structural_queries(self):
        """HyDE should NOT be recommended for structural queries."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        structural_queries = [
            "교원인사규정 제8조",
            "학칙",
            "3-1-24",
        ]

        for query in structural_queries:
            assert not generator.should_use_hyde(query, complexity="simple"), (
                f"'{query}' should not recommend HyDE"
            )

    def test_should_not_use_hyde_with_regulatory_terms(self):
        """Queries with regulatory terms should not use HyDE."""
        from src.rag.infrastructure.hyde import HyDEGenerator

        generator = HyDEGenerator(enable_cache=False)

        regulatory_queries = [
            "규정에 따른 휴직",
            "조항 확인",
            "세칙 조회",
        ]

        for query in regulatory_queries:
            assert not generator.should_use_hyde(query, complexity="medium"), (
                f"'{query}' with regulatory terms should not recommend HyDE"
            )

    def test_hyde_result_structure(self):
        """HyDEResult should have correct structure."""
        from src.rag.infrastructure.hyde import HyDEGenerator, HyDEResult

        generator = HyDEGenerator(enable_cache=False)
        result = generator.generate_hypothetical_doc("test query")

        # Check HyDEResult structure
        assert isinstance(result, HyDEResult)
        assert hasattr(result, "original_query")
        assert hasattr(result, "hypothetical_doc")
        assert hasattr(result, "from_cache")
        assert hasattr(result, "cache_key")

        # Type checks
        assert isinstance(result.original_query, str)
        assert isinstance(result.hypothetical_doc, str)
        assert isinstance(result.from_cache, bool)


class TestHyDESearcherValidation:
    """Test suite for HyDESearcher with empty queries."""

    def test_search_with_hyde_handles_empty_query(self):
        """HyDESearcher should raise error for empty query (Query validates)."""
        from src.rag.infrastructure.hyde import HyDESearcher

        mock_generator = MagicMock()
        # Return a proper HyDEResult instead of MagicMock to avoid __format__ issues
        from src.rag.infrastructure.hyde import HyDEResult

        mock_result = HyDEResult(
            original_query="",
            hypothetical_doc="",
            from_cache=False,
            quality_score=0.0,
        )
        mock_generator.generate_hypothetical_doc.return_value = mock_result

        mock_store = MagicMock()
        mock_store.search.return_value = []

        searcher = HyDESearcher(mock_generator, mock_store)

        # Query class validates that text cannot be empty - should raise ValueError
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            searcher.search_with_hyde("", top_k=10)

    def test_search_with_hyde_merges_empty_results(self):
        """HyDESearcher should handle case when both searches return empty."""
        from src.rag.infrastructure.hyde import HyDEResult, HyDESearcher

        mock_generator = MagicMock()
        # quality_score >= 0.3 triggers both HyDE and original query searches
        hyde_result = HyDEResult(
            original_query="test",
            hypothetical_doc="hypothetical document",  # Different from query to use HyDE
            from_cache=False,
            quality_score=0.7,  # High quality to trigger both searches
        )
        mock_generator.generate_hypothetical_doc.return_value = hyde_result

        mock_store = MagicMock()
        mock_store.search.return_value = []  # Empty results

        searcher = HyDESearcher(mock_generator, mock_store)

        results = searcher.search_with_hyde("test", top_k=10)

        # Should return empty list without crashing
        assert results == []
        assert mock_store.search.call_count == 2  # Both searches attempted

    def test_merge_results_deduplicates(self):
        """HyDESearcher should deduplicate results from both searches."""
        from src.rag.infrastructure.hyde import HyDEResult, HyDESearcher

        mock_generator = MagicMock()
        mock_generator.generate_hypothetical_doc.return_value = HyDEResult(
            original_query="test", hypothetical_doc="hypothetical doc", from_cache=False
        )

        # Create duplicate results
        chunk1 = FakeChunk(id="doc1", text="content1", title="title1")
        chunk2 = FakeChunk(id="doc2", text="content2", title="title2")

        result1 = FakeSearchResult(chunk=chunk1, score=0.9, rank=1)
        result2 = FakeSearchResult(chunk=chunk2, score=0.8, rank=2)

        mock_store = MagicMock()
        # Both searches return same results
        mock_store.search.return_value = [result1, result2]

        searcher = HyDESearcher(mock_generator, mock_store)

        merged = searcher.search_with_hyde("test", top_k=10)

        # Should deduplicate - doc1 and doc2 should appear only once
        chunk_ids = [r.chunk.id for r in merged]
        assert chunk_ids.count("doc1") == 1
        assert chunk_ids.count("doc2") == 1
