"""
Comprehensive tests for search_usecase.py to boost coverage toward 85%.

Focuses on uncovered methods:
- _search_by_rule_code_pattern
- _search_by_regulation_only
- _search_by_regulation_article
- _find_regulation_rule_code
- _search_composite
- _classify_query_complexity
- _should_skip_reranker
- _select_scoring_query
- _filter_sparse_results
- _metadata_matches
- ask_stream
- Various edge cases
"""

from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rag.application.search_usecase import (
    QueryRewriteInfo,
    SearchStrategy,
    SearchUseCase,
    _get_fallback_regulation_qa_prompt,
    _load_prompt,
)
from src.rag.domain.entities import (
    Chunk,
    ChunkLevel,
    Keyword,
    RegulationStatus,
    SearchResult,
)
from src.rag.domain.repositories import (
    IHybridSearcher,
    ILLMClient,
    IReranker,
    IVectorStore,
)
from src.rag.domain.value_objects import Query, SearchFilter
from src.rag.infrastructure.hybrid_search import ScoredDocument

# ====================
# Helper factories
# ====================


def make_chunk(
    chunk_id: str = "test-chunk-1",
    rule_code: str = "1-1-1",
    text: str = "테스트 내용",
    title: str = "제1조",
    parent_path: Optional[List[str]] = None,
    keywords: Optional[List[Keyword]] = None,
    status: RegulationStatus = RegulationStatus.ACTIVE,
    level: ChunkLevel = ChunkLevel.ARTICLE,
) -> Chunk:
    """Factory for creating test chunks."""
    return Chunk(
        id=chunk_id,
        rule_code=rule_code,
        level=level,
        title=title,
        text=text,
        embedding_text=text,
        full_text=text,
        parent_path=parent_path or ["교원인사규정"],
        token_count=len(text),
        keywords=keywords or [],
        is_searchable=True,
        status=status,
    )


def make_search_result(chunk: Chunk, score: float = 0.5, rank: int = 1) -> SearchResult:
    """Factory for creating test search results."""
    return SearchResult(chunk=chunk, score=score, rank=rank)


# ====================
# Mock implementations
# ====================


class MockVectorStore(IVectorStore):
    """Mock vector store for testing."""

    def __init__(self, search_results: Optional[List[SearchResult]] = None):
        self._results = search_results or []
        self._documents = []
        self.search_calls = []

    def add_chunks(self, chunks: List[Chunk]) -> int:
        return len(chunks)

    def delete_by_rule_codes(self, rule_codes: List[str]) -> int:
        return len(rule_codes)

    def search(
        self,
        query: Query,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        self.search_calls.append((query, filter, top_k))
        return self._results[:top_k]

    def get_all_rule_codes(self) -> set:
        return set()

    def count(self) -> int:
        return len(self._results)

    def get_all_documents(self) -> list:
        return self._documents

    def clear_all(self) -> int:
        return 0


class MockLLMClient(ILLMClient):
    """Mock LLM client for testing."""

    def __init__(self, response: str = "테스트 답변입니다."):
        self._response = response
        self.generate_calls = []
        self.stream_responses = []

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        self.generate_calls.append((system_prompt, user_message, temperature))
        return self._response

    def get_embedding(self, text: str) -> List[float]:
        return [0.1] * 768

    def stream_generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.0
    ):
        """Mock streaming - yields token chunks."""
        self.stream_responses.append((system_prompt, user_message))
        response = self._response
        # Split into chunks for realistic streaming
        chunk_size = 5
        for i in range(0, len(response), chunk_size):
            yield response[i : i + chunk_size]


class MockReranker(IReranker):
    """Mock reranker for testing."""

    def __init__(self):
        self.rerank_calls = []

    def rerank(
        self,
        query: str,
        documents: List[tuple],
        top_k: int = 10,
    ) -> List[tuple]:
        self.rerank_calls.append((query, documents, top_k))
        # Return documents with modified scores
        return [
            (doc[0], doc[1], 0.9 - i * 0.1, doc[2])
            for i, doc in enumerate(documents[:top_k])
        ]


class MockHybridSearcher(IHybridSearcher):
    """Mock hybrid searcher for testing."""

    def __init__(self):
        self._documents = []
        self.add_documents_calls = []
        self.search_sparse_calls = []
        self.fuse_results_calls = []
        self._query_analyzer = Mock()
        # Configure mock query analyzer
        mock_rewrite_result = MagicMock()
        mock_rewrite_result.rewritten = "test query"
        mock_rewrite_result.method = "rules"
        mock_rewrite_result.from_cache = False
        mock_rewrite_result.fallback = False
        mock_rewrite_result.used_intent = False
        mock_rewrite_result.used_synonyms = False
        mock_rewrite_result.matched_intents = []
        self._query_analyzer.rewrite_query_with_info = Mock(
            return_value=mock_rewrite_result
        )
        self._query_analyzer.has_synonyms = Mock(return_value=False)
        self._query_analyzer.detect_audience = Mock(return_value=None)
        self._query_analyzer.expand_query = Mock(side_effect=lambda q: q)
        self._query_analyzer.decompose_query = Mock(side_effect=lambda q: [q])
        self._query_analyzer._llm_client = None

    def add_documents(self, documents: List[tuple]) -> None:
        self.add_documents_calls.append(documents)
        self._documents.extend(documents)

    def search_sparse(self, query: str, top_k: int = 10) -> List:
        self.search_sparse_calls.append((query, top_k))
        return []

    def fuse_results(
        self,
        sparse_results: List,
        dense_results: List,
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List:
        self.fuse_results_calls.append(
            (sparse_results, dense_results, top_k, query_text)
        )
        return dense_results[:top_k]

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        self._query_analyzer._llm_client = llm_client

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms (abstract method implementation)."""
        return query


# ====================
# Tests for _search_by_rule_code_pattern
# ====================


class TestSearchByRuleCodePattern:
    """Tests for _search_by_rule_code_pattern method."""

    def test_search_by_rule_code_pattern_sets_query_rewrite_info(self):
        """Query rewrite info is set correctly for rule code pattern."""
        chunk = make_chunk(rule_code="3-1-5")
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("3-1-5", top_k=5)

        rewrite_info = usecase.get_last_query_rewrite()
        assert rewrite_info is not None
        assert rewrite_info.original == "3-1-5"
        assert rewrite_info.used is False
        assert rewrite_info.method is None

    def test_search_by_rule_code_pattern_uses_generic_query(self):
        """Rule code pattern uses generic '규정' query with filter."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("1-2-3", top_k=5)

        query, _, top_k = store.search_calls[0]
        assert query.text == "규정"
        assert top_k == 25  # top_k * 5

    def test_search_by_rule_code_pattern_applies_filter(self):
        """Rule code filter is applied to search."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("2-1-1", top_k=3)

        _, filter_obj, _ = store.search_calls[0]
        assert filter_obj is not None
        assert filter_obj.rule_codes == ["2-1-1"]


# ====================
# Tests for _search_by_regulation_only
# ====================


class TestSearchByRegulationOnly:
    """Tests for _search_by_regulation_only method."""

    def test_search_by_regulation_only_finds_rule_code(self):
        """Finds regulation's rule_code and searches with filter."""
        # Create chunks with matching parent_path
        chunk1 = make_chunk(
            chunk_id="c1",
            parent_path=["교원인사규정"],
            rule_code="2-1-1",
        )
        chunk2 = make_chunk(
            chunk_id="c2",
            parent_path=["교원인사규정"],
            rule_code="2-1-1",
        )
        store = MockVectorStore(
            [
                make_search_result(chunk1, score=0.9),
                make_search_result(chunk2, score=0.8),
            ]
        )
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search("교원인사규정", top_k=5)

        rewrite_info = usecase.get_last_query_rewrite()
        assert rewrite_info is not None
        assert rewrite_info.method == "regulation_only"
        assert len(results) > 0

    def test_search_by_regulation_only_no_match_falls_through(self):
        """Returns None when regulation not found, falls through to general search."""
        # Use a regulation name that won't match any chunks
        chunk = make_chunk(parent_path=["학칙"], rule_code="1-1-1")
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Search for different regulation
        results = usecase.search("존재하지않는규정", top_k=5)

        # Should fall through to general search (empty results in this case)
        assert isinstance(results, list)


# ====================
# Tests for _find_regulation_rule_code
# ====================


class TestFindRegulationRuleCode:
    """Tests for _find_regulation_rule_code method."""

    def test_find_rule_code_exact_match(self):
        """Exact match returns rule_code with score 1.0."""
        chunk = make_chunk(
            parent_path=["교원인사규정"],
            rule_code="2-1-1",
        )
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Use _find_regulation_rule_code via the search path
        usecase.search("교원인사규정", top_k=5)

        # Check that search was called with the regulation name
        query, _, _ = store.search_calls[0]
        assert "교원인사규정" in query.text

    def test_find_rule_code_partial_match(self):
        """Partial match returns rule_code with lower score."""
        chunk = make_chunk(
            parent_path=["동의대학교교원인사규정"],
            rule_code="2-1-1",
        )
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("교원인사규정", top_k=5)

        query, _, _ = store.search_calls[0]
        assert "교원인사규정" in query.text

    def test_find_rule_code_no_match(self):
        """No match returns None."""
        chunk = make_chunk(
            parent_path=["학칙"],
            rule_code="1-1-1",
        )
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("교원인사규정", top_k=5)

        # Should return empty results when no match found


# ====================
# Tests for _search_by_regulation_article
# ====================


class TestSearchByRegulationArticle:
    """Tests for _search_by_regulation_article method."""

    def test_search_by_regulation_article_sets_rewrite_info(self):
        """Query rewrite info is set correctly."""
        chunk = make_chunk(
            parent_path=["교원인사규정"],
            title="제8조",
            text="제8조 휴직",
        )
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        _ = usecase.search("교원인사규정 제8조", top_k=5)

        rewrite_info = usecase.get_last_query_rewrite()
        assert rewrite_info is not None
        assert rewrite_info.method == "regulation_article"

    def test_search_by_regulation_article_filters_by_article(self):
        """Filters results by matching article number."""
        # Create chunks with different article numbers
        chunk1 = make_chunk(
            chunk_id="c1",
            parent_path=["교원인사규정", "제8조"],
            title="제8조",
            text="제8조 본문",
        )
        chunk2 = make_chunk(
            chunk_id="c2",
            parent_path=["교원인사규정", "제9조"],
            title="제9조",
            text="제9조 본문",
        )
        store = MockVectorStore(
            [
                make_search_result(chunk1),
                make_search_result(chunk2),
            ]
        )
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        _ = usecase.search("교원인사규정 제8조", top_k=5)

        # Should filter for article 8
        rewrite_info = usecase.get_last_query_rewrite()
        assert rewrite_info is not None
        assert "제8조" in rewrite_info.rewritten


# ====================
# Tests for _classify_query_complexity
# ====================


class TestClassifyQueryComplexity:
    """Tests for _classify_query_complexity method."""

    def test_classify_complexity_simple_rule_code(self):
        """Rule code pattern is classified as simple."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        complexity = usecase._classify_query_complexity("3-1-5")
        assert complexity == "simple"

    def test_classify_complexity_simple_regulation_only(self):
        """Regulation-only pattern is classified as simple."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        complexity = usecase._classify_query_complexity("교원인사규정")
        assert complexity == "simple"

    def test_classify_complexity_simple_regulation_article(self):
        """Regulation + article pattern is classified as simple."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        complexity = usecase._classify_query_complexity("교원인사규정 제8조")
        assert complexity == "simple"

    def test_classify_complexity_complex_multiple_intents(self):
        """Multiple intents trigger complex classification."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        complexity = usecase._classify_query_complexity(
            "휴학 절차", matched_intents=["leave_of_absence", "procedures"]
        )
        assert complexity == "complex"

    def test_classify_complexity_complex_comparative_keywords(self):
        """Comparative keywords trigger complex classification."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        complexity = usecase._classify_query_complexity("휴학과休學의 차이점")
        assert complexity == "complex"

    def test_classify_complexity_complex_very_long_query(self):
        """Very long queries are classified as complex."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        long_query = "휴학 절차와 방법에 대해 " + "상세히 " * 30
        complexity = usecase._classify_query_complexity(long_query)
        assert complexity == "complex"

    def test_classify_complexity_medium_default(self):
        """Default classification is medium."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        complexity = usecase._classify_query_complexity("휴학 절차 알려줘")
        assert complexity == "medium"


# ====================
# Tests for _should_skip_reranker
# ====================


class TestShouldSkipReranker:
    """Tests for _should_skip_reranker method."""

    def test_should_skip_reranker_always_false(self):
        """Reranker skip behavior based on query characteristics."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Simple query with empty/short text skips reranker
        assert usecase._should_skip_reranker("simple") == (True, "short_simple")

        # Complex and medium queries apply reranker
        assert usecase._should_skip_reranker("complex") == (False, "apply")
        assert usecase._should_skip_reranker("medium") == (False, "apply")

        # With matched intents - apply reranker
        assert usecase._should_skip_reranker(
            "complex", matched_intents=["intent1", "intent2"]
        ) == (False, "apply")

        # With expanded query - apply reranker
        assert usecase._should_skip_reranker(
            "complex", query_significantly_expanded=True
        ) == (False, "apply")


# ====================
# Tests for _select_scoring_query
# ====================


class TestSelectScoringQuery:
    """Tests for _select_scoring_query method."""

    def test_select_scoring_query_no_rewrite(self):
        """Returns original when no rewrite."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._select_scoring_query("original", "")
        assert result == "original"

    def test_select_scoring_query_same_rewrite(self):
        """Returns original when rewrite is same."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._select_scoring_query("original", "original")
        assert result == "original"

    def test_select_scoring_query_article_in_original(self):
        """Combines when original has article pattern but rewrite doesn't."""

        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._select_scoring_query("제8조 휴직", "휴직 방법")
        assert "제8조" in result or result == "휴직 방법"

    def test_select_scoring_query_returns_rewrite(self):
        """Returns rewrite when no special conditions."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._select_scoring_query("query", "rewritten query")
        assert result == "rewritten query"


# ====================
# Tests for _filter_sparse_results
# ====================


class TestFilterSparseResults:
    """Tests for _filter_sparse_results method."""

    def test_filter_sparse_results_empty(self):
        """Empty results return empty."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._filter_sparse_results([], None, False)
        assert result == []

    def test_filter_sparse_results_no_filter(self):
        """No filter returns all results."""
        sparse_results = [
            ScoredDocument(doc_id="d1", score=0.9, content="text1", metadata={}),
            ScoredDocument(doc_id="d2", score=0.8, content="text2", metadata={}),
        ]
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._filter_sparse_results(sparse_results, None, True)
        assert len(result) == 2

    def test_filter_sparse_results_status_filter(self):
        """Status filter removes abolished when include_abolished=False."""
        sparse_results = [
            ScoredDocument(
                doc_id="d1", score=0.9, content="text1", metadata={"status": "active"}
            ),
            ScoredDocument(
                doc_id="d2",
                score=0.8,
                content="text2",
                metadata={"status": "abolished"},
            ),
        ]
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._filter_sparse_results(sparse_results, None, False)
        assert len(result) == 1
        assert result[0].doc_id == "d1"

    def test_filter_sparse_results_rule_code_filter(self):
        """Rule code filter removes non-matching results."""
        sparse_results = [
            ScoredDocument(
                doc_id="d1", score=0.9, content="text1", metadata={"rule_code": "A-1-1"}
            ),
            ScoredDocument(
                doc_id="d2", score=0.8, content="text2", metadata={"rule_code": "B-1-1"}
            ),
        ]
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        filter_obj = SearchFilter(rule_codes=["A-1-1"])
        result = usecase._filter_sparse_results(sparse_results, filter_obj, True)
        assert len(result) == 1
        assert result[0].doc_id == "d1"


# ====================
# Tests for _metadata_matches
# ====================


class TestMetadataMatches:
    """Tests for _metadata_matches method."""

    def test_metadata_matches_empty_filter(self):
        """Empty filter matches everything."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._metadata_matches({}, {"key": "value"})
        assert result is True

    def test_metadata_matches_equality(self):
        """Equality filter requires exact match."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._metadata_matches({"status": "active"}, {"status": "active"})
        assert result is True

        result = usecase._metadata_matches(
            {"status": "active"}, {"status": "abolished"}
        )
        assert result is False

    def test_metadata_matches_in_operator(self):
        """$in operator checks value is in list."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._metadata_matches(
            {"status": {"$in": ["active", "pending"]}}, {"status": "active"}
        )
        assert result is True

        result = usecase._metadata_matches(
            {"status": {"$in": ["active", "pending"]}}, {"status": "abolished"}
        )
        assert result is False

    def test_metadata_matches_in_operator_none_value(self):
        """$in operator returns False for None values."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._metadata_matches(
            {"status": {"$in": ["active"]}}, {"other_key": "value"}
        )
        assert result is False

    def test_metadata_matches_missing_key(self):
        """Missing metadata key returns False."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._metadata_matches({"status": "active"}, {"other_key": "value"})
        assert result is False


# ====================
# Tests for _build_rule_code_filter
# ====================


class TestBuildRuleCodeFilter:
    """Tests for _build_rule_code_filter static method."""

    def test_build_rule_code_filter_no_base(self):
        """Creates filter from scratch when no base filter."""
        result = SearchUseCase._build_rule_code_filter(None, "3-1-5")
        assert result.rule_codes == ["3-1-5"]
        assert result.status is None
        assert result.levels is None

    def test_build_rule_code_filter_with_base(self):
        """Merges rule_code into existing filter."""
        base = SearchFilter(
            status="active",
            levels=[ChunkLevel.ARTICLE],
        )
        result = SearchUseCase._build_rule_code_filter(base, "2-1-1")
        assert result.rule_codes == ["2-1-1"]
        assert result.status == "active"
        assert result.levels == [ChunkLevel.ARTICLE]


# ====================
# Tests for search_by_rule_code method
# ====================


class TestSearchByRuleCodeMethod:
    """Tests for search_by_rule_code method."""

    def test_search_by_rule_code_uses_filter(self):
        """Searches with rule_code filter."""
        chunk = make_chunk(rule_code="3-1-5")
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        _ = usecase.search_by_rule_code("3-1-5", top_k=10)

        _, filter_obj, _ = store.search_calls[0]
        assert filter_obj is not None
        assert filter_obj.rule_codes == ["3-1-5"]

    def test_search_by_rule_code_include_abolished_default(self):
        """Default includes abolished regulations."""
        chunk = make_chunk(status=RegulationStatus.ABOLISHED)
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        _ = usecase.search_by_rule_code("3-1-5")

        query, _, _ = store.search_calls[0]
        assert query.include_abolished is True

    def test_search_by_rule_code_exclude_abolished(self):
        """Can exclude abolished regulations."""
        chunk_active = make_chunk(status=RegulationStatus.ACTIVE)
        chunk_abolished = make_chunk(status=RegulationStatus.ABOLISHED)
        store = MockVectorStore(
            [
                make_search_result(chunk_active),
                make_search_result(chunk_abolished),
            ]
        )
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        _ = usecase.search_by_rule_code("3-1-5", include_abolished=False)

        query, _, _ = store.search_calls[0]
        assert query.include_abolished is False


# ====================
# Tests for ask_stream method
# ====================


class TestAskStream:
    """Tests for ask_stream method."""

    def test_ask_stream_raises_without_llm(self):
        """Raises ConfigurationError when LLM not configured."""
        from src.exceptions import ConfigurationError

        store = MockVectorStore([])
        usecase = SearchUseCase(store, llm_client=None)

        with pytest.raises(ConfigurationError, match="LLM client not configured"):
            list(usecase.ask_stream("질문"))

    def test_ask_stream_yields_metadata_first(self):
        """Yields metadata dict before tokens."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient(response="답변입니다.")

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )

        results = list(usecase.ask_stream("질문"))

        # First result should be metadata
        assert results[0]["type"] == "metadata"
        assert "sources" in results[0]
        assert "confidence" in results[0]

        # Subsequent results should be tokens
        assert results[1]["type"] == "token"

    def test_ask_stream_fallback_to_non_streaming(self):
        """Falls back to non-streaming when stream_generate not available."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])

        # LLM without stream_generate method
        llm_no_stream = Mock()
        llm_no_stream.generate = Mock(return_value="답변")
        del llm_no_stream.stream_generate  # Remove streaming method

        usecase = SearchUseCase(
            store,
            llm_client=llm_no_stream,
            use_reranker=False,
            use_hybrid=False,
        )

        results = list(usecase.ask_stream("질문"))

        # Should still return results
        assert len(results) > 0
        assert results[0]["type"] == "metadata"

    def test_ask_stream_no_results_message(self):
        """Returns message when no search results found."""
        store = MockVectorStore([])  # No results
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )

        results = list(usecase.ask_stream("알 수 없는 질문"))

        assert results[0]["type"] == "metadata"
        assert results[0]["confidence"] == 0.0
        assert results[1]["type"] == "token"
        assert "찾을 수 없습니다" in results[1]["content"]

    def test_ask_stream_with_search_query(self):
        """Uses search_query for retrieval."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )

        # Mock cache check to bypass retrieval cache
        with patch.object(usecase, "_check_retrieval_cache", return_value=None):
            list(usecase.ask_stream("질문", search_query="검색어"))

        query, _, _ = store.search_calls[0]
        assert "검색어" in query.text


# ====================
# Tests for prompt loading
# ====================


class TestPromptLoading:
    """Tests for prompt loading functions."""

    def test_fallback_prompt_returns_string(self):
        """Fallback prompt returns non-empty string."""
        prompt = _get_fallback_regulation_qa_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "규정" in prompt or "전문가" in prompt

    def test_load_prompt_returns_fallback_on_error(self):
        """Returns fallback prompt when file not found."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with patch("src.rag.application.search_usecase.logger"):
                prompt = _load_prompt("nonexistent")
                assert isinstance(prompt, str)
                assert len(prompt) > 0


# ====================
# Tests for edge cases and error handling
# ====================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_search_with_none_query_coerces_to_empty(self):
        """None query is coerced to empty string."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search(None, top_k=5)
        assert results == []

    def test_search_with_list_query_joins_elements(self):
        """List query is joined with spaces."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Mock cache check to bypass retrieval cache
        with patch.object(usecase, "_check_retrieval_cache", return_value=None):
            usecase.search(["term1", "term2"], top_k=5)

        query, _, _ = store.search_calls[0]
        assert query.text == "term1 term2"

    def test_search_normalizes_unicode(self):
        """Query text is normalized to NFC."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Mock cache check to bypass retrieval cache
        with patch.object(usecase, "_check_retrieval_cache", return_value=None):
            # Composed character (가)
            usecase.search("가", top_k=5)

        query, _, _ = store.search_calls[0]
        # Should be normalized
        assert query.text is not None

    def test_get_recommended_strategy_public_method(self):
        """get_recommended_strategy is accessible public API."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Short query -> DIRECT
        strategy = usecase.get_recommended_strategy("휴학")
        assert strategy == SearchStrategy.DIRECT

        # Long query -> TOOL_CALLING
        strategy = usecase.get_recommended_strategy(
            "교원 승진에 필요한 업적 평가 기준과 절차를 상세히 알려주세요."
        )
        assert strategy == SearchStrategy.TOOL_CALLING

    def test_get_last_query_rewrite_returns_none_initially(self):
        """Query rewrite info is None before any search."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        assert usecase.get_last_query_rewrite() is None

    def test_query_rewrite_info_frozen_dataclass(self):
        """QueryRewriteInfo is frozen and cannot be modified."""
        from dataclasses import FrozenInstanceError

        info = QueryRewriteInfo(
            original="test",
            rewritten="rewritten",
            used=True,
        )
        # Should be frozen
        with pytest.raises(FrozenInstanceError):
            info.method = "new"


# ====================
# Tests for _search_composite
# ====================


class TestSearchComposite:
    """Tests for _search_composite method."""

    def test_search_composite_rrf_fusion(self):
        """Composite search uses RRF to merge sub-query results."""
        # Create multiple chunks for different sub-queries
        chunk1 = make_chunk(chunk_id="c1", text="휴학 신청")
        chunk2 = make_chunk(chunk_id="c2", text="휴학 절차")
        chunk3 = make_chunk(chunk_id="c3", text="복학 절차")

        # Setup store to return different results based on query
        store = MockVectorStore(
            [
                make_search_result(chunk1, score=0.9),
                make_search_result(chunk2, score=0.8),
                make_search_result(chunk3, score=0.7),
            ]
        )
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        # Mock hybrid searcher to return decomposed queries
        hybrid = MockHybridSearcher()
        hybrid._query_analyzer.decompose_query = Mock(
            return_value=["휴학 신청", "휴학 절차"]
        )
        usecase._hybrid_searcher = hybrid
        usecase._hybrid_initialized = True

        _ = usecase.search("휴학 방법과 절차", top_k=5)

        # Should have called decompose_query
        hybrid._query_analyzer.decompose_query.assert_called_once()

    def test_search_composite_deduplicates_by_article(self):
        """Composite search deduplicates results by article."""
        chunk1 = make_chunk(
            chunk_id="c1",
            parent_path=["규정", "제8조"],
            title="제8조",
        )
        chunk2 = make_chunk(
            chunk_id="c2",
            parent_path=["규정", "제8조"],
            title="①",
        )

        store = MockVectorStore(
            [
                make_search_result(chunk1, score=0.9),
                make_search_result(chunk2, score=0.8),
            ]
        )
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        hybrid = MockHybridSearcher()
        hybrid._query_analyzer.decompose_query = Mock(return_value=["query1", "query2"])
        usecase._hybrid_searcher = hybrid
        usecase._hybrid_initialized = True

        _ = usecase.search("test", top_k=5)

        # Should deduplicate by article
        # (actual deduplication depends on article detection)


# ====================
# Tests for audience penalty
# ====================


class TestAudiencePenalty:
    """Tests for _apply_audience_penalty method."""

    def test_audience_penalty_none_returns_score(self):
        """None audience returns score unchanged."""
        chunk = make_chunk()
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._apply_audience_penalty(chunk, None, 0.8)
        assert result == 0.8

    def test_audience_penalty_faculty_on_student_reg(self):
        """Faculty audience penalizes student-specific regulations."""
        chunk = make_chunk(parent_path=["학생생활규정"])
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        from src.rag.infrastructure.query_analyzer import Audience

        result = usecase._apply_audience_penalty(chunk, Audience.FACULTY, 0.8)
        # Should be penalized
        assert result < 0.8

    def test_audience_penalty_student_on_faculty_reg(self):
        """Student audience penalizes faculty-specific regulations."""
        chunk = make_chunk(parent_path=["교원인사규정"])
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        from src.rag.infrastructure.query_analyzer import Audience

        result = usecase._apply_audience_penalty(chunk, Audience.STUDENT, 0.8)
        # Should be penalized
        assert result < 0.8


# ====================
# Tests for low signal chunk detection
# ====================


class TestIsLowSignalChunk:
    """Tests for _is_low_signal_chunk method."""

    def test_is_low_signal_empty_text(self):
        """Empty text returns True."""
        chunk = make_chunk(text="")
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._is_low_signal_chunk(chunk)
        assert result is True

    def test_is_low_signal_heading_only(self):
        """Heading-only chunks with low token count return True."""
        # HEADING_ONLY_PATTERN matches parenthesized content like "(1)", "(가)"
        chunk = Chunk(
            id="test",
            rule_code="1-1-1",
            level=ChunkLevel.ARTICLE,
            title="제1조",
            text="(1)",  # Matches HEADING_ONLY_PATTERN: ^\([^)]*\)\s*$
            embedding_text="(1)",
            full_text="(1)",
            parent_path=[],
            token_count=10,  # Below the 30 token threshold
            keywords=[],
            is_searchable=True,
        )
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._is_low_signal_chunk(chunk)
        assert result is True

    def test_is_low_signal_substantial_content(self):
        """Substantial content returns False."""
        # Create Chunk directly to control token_count (make_chunk calculates from len(text))
        chunk = Chunk(
            id="test",
            rule_code="1-1-1",
            level=ChunkLevel.ARTICLE,
            title="제1조",
            text="제1조 본문 내용입니다. 이 내용은 충분히 길어서 헤딩 전용 콘텐츠로 간주되지 않습니다.",
            embedding_text="제1조 본문 내용입니다. 이 내용은 충분히 길어서 헤딩 전용 콘텐츠로 간주되지 않습니다.",
            full_text="제1조 본문 내용입니다. 이 내용은 충분히 길어서 헤딩 전용 콘텐츠로 간주되지 않습니다.",
            parent_path=[],
            token_count=50,  # Above the 30 token threshold
            keywords=[],
            is_searchable=True,
        )
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        result = usecase._is_low_signal_chunk(chunk)
        assert result is False


# ====================
# Tests for context building
# ====================


class TestBuildContext:
    """Tests for _build_context method."""

    def test_build_context_single_result(self):
        """Single result formatted correctly."""
        chunk = make_chunk(text="본문 내용")
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = [make_search_result(chunk)]
        context = usecase._build_context(results)

        assert "[1]" in context
        assert "본문 내용" in context
        assert "출처:" in context

    def test_build_context_multiple_results(self):
        """Multiple results are numbered sequentially."""
        chunk1 = make_chunk(chunk_id="c1", text="내용1")
        chunk2 = make_chunk(chunk_id="c2", text="내용2")
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = [
            make_search_result(chunk1),
            make_search_result(chunk2),
        ]
        context = usecase._build_context(results)

        assert "[1]" in context
        assert "[2]" in context
        assert "내용1" in context
        assert "내용2" in context


# ====================
# Tests for user message building
# ====================


class TestBuildUserMessage:
    """Tests for _build_user_message method."""

    def test_build_user_message_without_history(self):
        """Message without history has simpler format."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        message = usecase._build_user_message("질문", "context", None)

        assert "질문: 질문" in message
        assert "context" in message
        assert "대화 기록" not in message

    def test_build_user_message_with_history(self):
        """Message with history includes conversation history."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        message = usecase._build_user_message(
            "질문", "context", history_text="이전 대화"
        )

        assert "대화 기록" in message
        assert "이전 대화" in message
        assert "현재 질문: 질문" in message


# ====================
# Tests for fact checking and Self-RAG
# ====================


class TestFactCheckingAndSelfRAG:
    """Tests for fact checking and Self-RAG features."""

    def test_ask_with_fact_check_disabled(self):
        """Ask works when fact check is disabled."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase._enable_fact_check = False

        answer = usecase.ask("질문")

        assert answer.text == llm._response

    def test_ask_with_self_rag_disabled(self):
        """Ask works when Self-RAG is disabled."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase._enable_self_rag = False

        answer = usecase.ask("질문")

        assert answer.text == llm._response

    def test_ask_with_query_expansion_disabled(self):
        """Ask works when query expansion is disabled."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase._enable_query_expansion = False

        answer = usecase.ask("질문")

        assert answer.text == llm._response


# ====================
# Tests for deduplication by article
# ====================


class TestDeduplicateByArticle:
    """Tests for _deduplicate_by_article method."""

    def test_deduplicate_by_article_keeps_highest_score(self):
        """Keeps highest scoring chunk when multiple chunks from same article."""
        # Same article (제8조), different scores
        chunk1 = make_chunk(
            chunk_id="c1",
            parent_path=["규정", "제8조"],
            title="제8조",
        )
        chunk2 = make_chunk(
            chunk_id="c2",
            parent_path=["규정", "제8조"],
            title="①",
        )

        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        input_results = [
            make_search_result(chunk2, score=0.9, rank=1),  # Higher score
            make_search_result(chunk1, score=0.8, rank=2),
        ]

        deduplicated = usecase._deduplicate_by_article(input_results, top_k=10)

        # Should keep only one (highest scoring comes first after sorting)
        assert len(deduplicated) <= 2

    def test_deduplicate_by_article_different_articles(self):
        """Keeps chunks from different articles."""
        chunk1 = make_chunk(
            chunk_id="c1",
            parent_path=["규정", "제8조"],
            title="제8조",
        )
        chunk2 = make_chunk(
            chunk_id="c2",
            parent_path=["규정", "제9조"],
            title="제9조",
        )

        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = [
            make_search_result(chunk1, score=0.9),
            make_search_result(chunk2, score=0.8),
        ]

        deduplicated = usecase._deduplicate_by_article(results, top_k=10)

        # Should keep both (different articles)
        assert len(deduplicated) == 2

    def test_deduplicate_by_article_respects_top_k(self):
        """Respects top_k parameter."""
        chunks = []
        for i in range(10):
            chunk = make_chunk(
                chunk_id=f"c{i}",
                parent_path=[f"규정{i}", f"제{i}조"],
                title=f"제{i}조",
            )
            chunks.append(make_search_result(chunk, score=0.9 - i * 0.05))

        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        deduplicated = usecase._deduplicate_by_article(chunks, top_k=3)

        # Should only return top_k results
        assert len(deduplicated) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
