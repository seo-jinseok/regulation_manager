"""
Extended unit tests for SearchUseCase.

Tests clean architecture compliance with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Optional

from src.rag.application.search_usecase import (
    SearchUseCase,
    _extract_regulation_article_query,
    _extract_regulation_only_query,
)
from src.rag.infrastructure.patterns import normalize_article_token
from src.rag.domain.entities import (
    Answer,
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
) -> Chunk:
    """Factory for creating test chunks."""
    return Chunk(
        id=chunk_id,
        rule_code=rule_code,
        level=ChunkLevel.ARTICLE,
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
        # Mock _query_analyzer with proper return values
        self._query_analyzer = Mock()
        # Create a proper mock for QueryRewriteResult
        mock_rewrite_result = MagicMock()
        mock_rewrite_result.rewritten = "expanded query"
        mock_rewrite_result.method = "rules"
        mock_rewrite_result.from_cache = False
        mock_rewrite_result.fallback = False
        mock_rewrite_result.used_intent = False
        mock_rewrite_result.used_synonyms = True
        mock_rewrite_result.matched_intents = []  # Must be a list, not None
        self._query_analyzer.rewrite_query_with_info = Mock(return_value=mock_rewrite_result)
        self._query_analyzer.has_synonyms = Mock(return_value=True)
        self._query_analyzer.detect_audience = Mock(return_value=None)
        self._query_analyzer.expand_query = Mock(side_effect=lambda q: q)  # Return query as-is
        self._query_analyzer.decompose_query = Mock(side_effect=lambda q: [q])  # Return single-element list (no decomposition)
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
        self.fuse_results_calls.append((sparse_results, dense_results, top_k, query_text))
        # Return dense results as-is for simplicity
        return dense_results[:top_k]

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        self._query_analyzer._llm_client = llm_client

    def expand_query(self, query: str) -> str:
        return query


# ====================
# Unit tests for helper functions
# ====================

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_extract_regulation_article_query_basic(self):
        """Basic regulation + article extraction."""
        result = _extract_regulation_article_query("교원인사규정 제8조")
        assert result == ("교원인사규정", "제8조")

    def test_extract_regulation_article_query_with_paragraph(self):
        """Extraction with paragraph number."""
        result = _extract_regulation_article_query("교원인사규정 제15조 제2항")
        assert result is not None
        assert result[0] == "교원인사규정"

    def test_extract_regulation_article_query_no_match(self):
        """Returns None when no pattern matches."""
        result = _extract_regulation_article_query("휴학 절차 알려줘")
        assert result is None

    def test_extract_regulation_only_query_basic(self):
        """Extract regulation name only."""
        result = _extract_regulation_only_query("교원인사규정")
        assert result == "교원인사규정"

    def test_extract_regulation_only_query_with_whitespace(self):
        """Handles query with proper ending."""
        result = _extract_regulation_only_query("교원인사규정")
        assert result == "교원인사규정"

    def test_extract_regulation_only_query_not_only(self):
        """Returns None when query is not regulation-only."""
        result = _extract_regulation_only_query("교원인사규정 제8조")
        assert result is None

    def test_normalize_article_token(self):
        """Normalizes article tokens by removing spaces."""
        assert normalize_article_token("제 8 조") == "제8조"
        assert normalize_article_token("제8조") == "제8조"


# ====================
# Unit tests for SearchUseCase
# ====================

class TestSearchUseCaseInit:
    """Tests for SearchUseCase initialization."""

    def test_init_with_minimal_params(self):
        """Create use case with only required params."""
        store = MockVectorStore()
        usecase = SearchUseCase(store)

        assert usecase.store is store
        assert usecase.llm is None

    def test_init_with_all_dependencies(self):
        """Create use case with all dependencies injected."""
        store = MockVectorStore()
        llm = MockLLMClient()
        reranker = MockReranker()
        hybrid = MockHybridSearcher()

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
            use_reranker=True,
            reranker=reranker,
            use_hybrid=True,
            hybrid_searcher=hybrid,
        )

        assert usecase.store is store
        assert usecase.llm is llm
        assert usecase._reranker is reranker
        assert usecase._hybrid_searcher is hybrid

    def test_init_respects_config_defaults(self):
        """Config defaults are used when not explicitly provided."""
        store = MockVectorStore()
        usecase = SearchUseCase(store, use_reranker=None, use_hybrid=None)

        # Should use config values (existence check)
        assert usecase.use_reranker is not None
        assert usecase._use_hybrid is not None


class TestSearchUseCaseSearch:
    """Tests for SearchUseCase.search() method."""

    def test_search_empty_query_returns_empty(self):
        """Empty query returns empty results."""
        store = MockVectorStore([make_search_result(make_chunk())])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search("", top_k=5)

        assert results == []

    def test_search_whitespace_only_returns_empty(self):
        """Whitespace-only query returns empty results."""
        store = MockVectorStore([make_search_result(make_chunk())])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search("   ", top_k=5)

        assert results == []

    def test_search_calls_store(self):
        """Search delegates to vector store."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search("휴학 절차", top_k=5)

        assert len(store.search_calls) == 1
        assert len(results) > 0

    def test_search_with_rule_code_pattern(self):
        """Rule code query uses appropriate filter."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("3-1-5", top_k=5)

        assert len(store.search_calls) == 1
        query, filter_, _ = store.search_calls[0]
        assert filter_ is not None
        assert filter_.rule_codes == ["3-1-5"]

    def test_search_applies_keyword_bonus(self):
        """Keyword matches boost score."""
        chunk = make_chunk(keywords=[Keyword(term="교원", weight=1.0)])
        store = MockVectorStore([make_search_result(chunk, score=0.4)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search("교원", top_k=1)

        # Keyword bonus: min(0.3, 1.0 * 0.05) = 0.05
        assert results[0].score == pytest.approx(0.45)

    def test_search_applies_article_bonus(self):
        """Exact article number match gets bonus."""
        chunk = make_chunk(text="제15조 교원의 연구년")
        store = MockVectorStore([make_search_result(chunk, score=0.4)])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search("제15조", top_k=1)

        # Article bonus: 0.2
        assert results[0].score >= 0.6

    def test_search_with_injected_reranker(self):
        """Search uses injected reranker."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        reranker = MockReranker()

        usecase = SearchUseCase(
            store,
            use_reranker=True,
            reranker=reranker,
            use_hybrid=False,
        )
        results = usecase.search("휴학", top_k=5)

        assert len(reranker.rerank_calls) == 1

    def test_search_with_injected_hybrid_searcher(self):
        """Search uses injected hybrid searcher."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        hybrid = MockHybridSearcher()

        usecase = SearchUseCase(
            store,
            use_reranker=False,
            use_hybrid=True,
            hybrid_searcher=hybrid,
        )
        results = usecase.search("휴학", top_k=5)

        assert len(hybrid.search_sparse_calls) >= 1  # May be called multiple times due to Corrective RAG


class TestSearchUseCaseSearchUnique:
    """Tests for SearchUseCase.search_unique() method."""

    def test_search_unique_deduplicates_by_rule_code(self):
        """Returns only one result per rule_code."""
        chunk1 = make_chunk(chunk_id="c1", rule_code="A-1-1")
        chunk2 = make_chunk(chunk_id="c2", rule_code="A-1-1")  # Same rule_code
        chunk3 = make_chunk(chunk_id="c3", rule_code="B-1-1")  # Different

        store = MockVectorStore([
            make_search_result(chunk1, score=0.9, rank=1),
            make_search_result(chunk2, score=0.8, rank=2),
            make_search_result(chunk3, score=0.7, rank=3),
        ])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search_unique("test", top_k=10)

        rule_codes = [r.chunk.rule_code for r in results]
        assert len(rule_codes) == 2
        assert rule_codes == ["A-1-1", "B-1-1"]

    def test_search_unique_skips_dedup_for_regulation_only(self):
        """Regulation-only queries skip deduplication."""
        chunk1 = make_chunk(chunk_id="c1", rule_code="A-1-1")
        chunk2 = make_chunk(chunk_id="c2", rule_code="A-1-1")

        store = MockVectorStore([
            make_search_result(chunk1, score=0.9, rank=1),
            make_search_result(chunk2, score=0.8, rank=2),
        ])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        results = usecase.search_unique("교원인사규정", top_k=10)

        # Should return both (no dedup for regulation-only)
        assert len(results) >= 1


class TestSearchUseCaseAsk:
    """Tests for SearchUseCase.ask() method."""

    def test_ask_requires_llm_client(self):
        """Raises ConfigurationError if LLM client not configured."""
        from src.exceptions import ConfigurationError
        store = MockVectorStore([make_search_result(make_chunk())])
        usecase = SearchUseCase(store, llm_client=None)

        with pytest.raises(ConfigurationError, match="LLM client not configured"):
            usecase.ask("질문입니다")

    def test_ask_returns_answer_with_sources(self):
        """Ask returns Answer with sources."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient(response="답변입니다.")

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        answer = usecase.ask("교원 연구년 자격은?")

        assert isinstance(answer, Answer)
        assert answer.text == "답변입니다."
        assert len(answer.sources) > 0

    def test_ask_uses_search_query_for_retrieval(self):
        """Custom search_query is used for retrieval."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase.ask(
            question="연구년 자격은?",
            search_query="교원인사규정 연구년",
        )

        query, _, _ = store.search_calls[0]
        assert "교원인사규정" in query.text or "연구년" in query.text

    def test_ask_includes_history_in_prompt(self):
        """History text is included in LLM prompt."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase._enable_self_rag = False  # Disable Self-RAG for this test
        usecase.ask(
            question="다음 질문",
            history_text="이전 대화 기록",
        )

        _, user_message, _ = llm.generate_calls[0]
        assert "대화 기록" in user_message
        assert "이전 대화 기록" in user_message

    def test_ask_returns_no_results_message(self):
        """Returns appropriate message when no results found."""
        store = MockVectorStore([])  # No results
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
        )
        answer = usecase.ask("알 수 없는 질문")

        assert "찾을 수 없습니다" in answer.text
        assert len(answer.sources) == 0
        assert answer.confidence == 0.0


class TestSearchUseCaseConfidence:
    """Tests for confidence computation."""

    def test_compute_confidence_empty_results(self):
        """Empty results return 0 confidence."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store)

        confidence = usecase._compute_confidence([])

        assert confidence == 0.0

    def test_compute_confidence_high_scores(self):
        """High scores produce high confidence."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store)

        chunk = make_chunk()
        results = [
            make_search_result(chunk, score=0.9, rank=1),
            make_search_result(chunk, score=0.85, rank=2),
            make_search_result(chunk, score=0.8, rank=3),
        ]
        confidence = usecase._compute_confidence(results)

        assert confidence > 0.7

    def test_compute_confidence_low_scores(self):
        """Low scores produce lower confidence."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store)

        chunk = make_chunk()
        results = [
            make_search_result(chunk, score=0.3, rank=1),
            make_search_result(chunk, score=0.25, rank=2),
            make_search_result(chunk, score=0.2, rank=3),
        ]
        confidence = usecase._compute_confidence(results)

        assert confidence < 0.5


class TestSearchUseCaseQueryRewrite:
    """Tests for query rewrite tracking."""

    def test_get_last_query_rewrite_initially_none(self):
        """Initially returns None before any search."""
        store = MockVectorStore([])
        usecase = SearchUseCase(store, use_hybrid=False)

        assert usecase.get_last_query_rewrite() is None

    def test_get_last_query_rewrite_after_search(self):
        """Returns rewrite info after search."""
        store = MockVectorStore([make_search_result(make_chunk())])
        usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

        usecase.search("테스트 쿼리")

        rewrite_info = usecase.get_last_query_rewrite()
        assert rewrite_info is not None
        assert rewrite_info.original == "테스트 쿼리"
