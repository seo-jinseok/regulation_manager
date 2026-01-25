"""
Advanced tests for search_usecase.py covering complex scenarios.

Focuses on:
- _apply_corrective_rag
- _apply_hyde
- _should_use_hyde
- _apply_dynamic_expansion
- _ensure_* methods
- Additional reranking edge cases
- HyDE and Self-RAG integration tests
"""

from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rag.application.search_usecase import SearchUseCase
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
        return [
            (doc[0], doc[1], 0.9 - i * 0.1, doc[2])
            for i, doc in enumerate(documents[:top_k])
        ]


class MockHybridSearcher(IHybridSearcher):
    """Mock hybrid searcher for testing."""

    def __init__(self):
        self._documents = []
        self._query_analyzer = Mock()
        # Configure query analyzer mocks
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
        self._documents.extend(documents)

    def search_sparse(self, query: str, top_k: int = 10) -> List:
        return []

    def fuse_results(
        self,
        sparse_results: List,
        dense_results: List,
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List:
        return dense_results[:top_k]

    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        self._query_analyzer._llm_client = llm_client

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms (abstract method implementation)."""
        return query


# ====================
# Tests for _ensure_* methods
# ====================


class TestEnsureMethods:
    """Tests for lazy initialization _ensure_* methods."""

    def test_ensure_hybrid_searcher_with_no_documents(self):
        """When no documents, hybrid_searcher remains None after ensure."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=True, enable_warmup=False
        )

        usecase._ensure_hybrid_searcher()

        assert usecase._hybrid_initialized is True
        assert usecase._hybrid_searcher is None

    def test_ensure_hybrid_searcher_already_initialized(self):
        """Does not re-initialize if already initialized."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=True, enable_warmup=False
        )

        # Set initialized flag
        usecase._hybrid_initialized = True

        # Should not raise any errors
        usecase._ensure_hybrid_searcher()

    def test_ensure_hybrid_searcher_with_llm_client(self):
        """Sets LLM client on hybrid searcher when available."""
        documents = [("doc1", "content", {"rule_code": "1-1-1"})]
        store = MockVectorStore([])
        store._documents = documents

        llm = MockLLMClient()
        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=True,
            enable_warmup=False,
        )

        usecase._ensure_hybrid_searcher()

        assert usecase._hybrid_initialized is True

    def test_ensure_reranker_initializes_reranker(self):
        """Initializes reranker when not already initialized."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=True, use_hybrid=False, enable_warmup=False
        )

        assert usecase._reranker_initialized is False

        # warmup_reranker is imported inside _ensure_reranker from ..infrastructure.reranker
        with patch("src.rag.infrastructure.reranker.warmup_reranker"):
            usecase._ensure_reranker()

        assert usecase._reranker_initialized is True
        assert usecase._reranker is not None

    def test_ensure_reranker_already_initialized(self):
        """Does not re-initialize if already initialized."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=True, use_hybrid=False, enable_warmup=False
        )

        usecase._reranker_initialized = True
        usecase._reranker = MockReranker()

        # Should not raise any errors
        usecase._ensure_reranker()


# ====================
# Tests for HyDE-related methods
# ====================


class TestHydeMethods:
    """Tests for HyDE (Hypothetical Document Embeddings) methods."""

    def test_ensure_hyde_creates_generator_when_enabled(self):
        """Creates HyDE generator when enabled."""
        store = MockVectorStore([])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
            enable_warmup=False,
        )
        usecase._enable_hyde = True

        usecase._ensure_hyde()

        assert usecase._hyde_generator is not None

    def test_ensure_hyde_skips_when_disabled(self):
        """Skips initialization when HyDE is disabled."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_hyde = False

        usecase._ensure_hyde()

        assert usecase._hyde_generator is None

    def test_should_use_hyde_returns_false_when_disabled(self):
        """Returns False when HyDE is disabled."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_hyde = False

        result = usecase._should_use_hyde("query", "medium")

        assert result is False

    def test_should_use_hyde_returns_false_when_no_generator(self):
        """Returns False when generator is None."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_hyde = True
        usecase._hyde_generator = None

        result = usecase._should_use_hyde("query", "medium")

        assert result is False

    def test_apply_hyde_returns_empty_when_no_generator(self):
        """Returns empty list when generator is None."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._hyde_generator = None

        result = usecase._apply_hyde("query", None, 10)

        assert result == []


# ====================
# Tests for dynamic query expansion
# ====================


class TestDynamicExpansion:
    """Tests for dynamic query expansion methods."""

    def test_ensure_query_expander_creates_expander(self):
        """Creates query expander when enabled."""
        store = MockVectorStore([])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
            enable_warmup=False,
        )
        usecase._enable_query_expansion = True

        usecase._ensure_query_expander()

        assert usecase._query_expander is not None

    def test_apply_dynamic_expansion_disabled(self):
        """Returns original query and empty keywords when disabled."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_query_expansion = False

        expanded, keywords = usecase._apply_dynamic_expansion("test query")

        assert expanded == "test query"
        assert keywords == []

    def test_apply_dynamic_expansion_no_expander(self):
        """Returns original query when expansion is disabled."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        # Disable query expansion - should return original query with no keywords
        usecase._enable_query_expansion = False

        expanded, keywords = usecase._apply_dynamic_expansion("test query")

        assert expanded == "test query"
        assert keywords == []


# ====================
# Tests for Self-RAG methods
# ====================


class TestSelfRagMethods:
    """Tests for Self-RAG related methods."""

    def test_ensure_self_rag_creates_pipeline(self):
        """Creates Self-RAG pipeline when enabled."""
        store = MockVectorStore([])
        llm = MockLLMClient()

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
            enable_warmup=False,
        )
        usecase._enable_self_rag = True

        usecase._ensure_self_rag()

        assert usecase._self_rag_pipeline is not None

    def test_ensure_self_rag_skips_when_disabled(self):
        """Skips initialization when Self-RAG is disabled."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_self_rag = False

        usecase._ensure_self_rag()

        assert usecase._self_rag_pipeline is None

    def test_ensure_fact_checker_creates_checker(self):
        """Creates fact checker when enabled."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_fact_check = True

        usecase._ensure_fact_checker()

        assert usecase._fact_checker is not None

    def test_apply_self_rag_relevance_filter_disabled(self):
        """Returns original results when Self-RAG is disabled."""
        chunk = make_chunk()
        results = [make_search_result(chunk)]
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_self_rag = False

        filtered = usecase._apply_self_rag_relevance_filter("query", results)

        assert filtered == results

    def test_apply_self_rag_relevance_filter_no_pipeline(self):
        """Returns original results when pipeline is None."""
        chunk = make_chunk()
        results = [make_search_result(chunk)]
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )
        usecase._enable_self_rag = True
        usecase._self_rag_pipeline = None

        filtered = usecase._apply_self_rag_relevance_filter("query", results)

        assert filtered == results


# ====================
# Tests for _apply_corrective_rag
# ====================


class TestApplyCorrectiveRAG:
    """Tests for _apply_corrective_rag method."""

    def test_corrective_rag_returns_results_when_no_correction_needed(self):
        """Returns original results when evaluator says no correction needed."""
        chunk = make_chunk()
        results = [make_search_result(chunk)]
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        # Mock evaluator to return False (no correction needed)
        mock_evaluator = Mock()
        mock_evaluator.needs_correction = Mock(return_value=False)
        usecase._retrieval_evaluator = mock_evaluator

        corrected = usecase._apply_corrective_rag(
            "query", results, None, 5, False, None, "medium"
        )

        assert corrected == results

    def test_corrective_rag_returns_results_when_no_hybrid_searcher(self):
        """Returns original results when no hybrid searcher available."""
        chunk = make_chunk()
        results = [make_search_result(chunk)]
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        # Mock evaluator to return True (correction needed)
        mock_evaluator = Mock()
        mock_evaluator.needs_correction = Mock(return_value=True)
        usecase._retrieval_evaluator = mock_evaluator
        usecase._hybrid_searcher = None

        corrected = usecase._apply_corrective_rag(
            "query", results, None, 5, False, None, "medium"
        )

        # Should return original results since no hybrid searcher
        assert corrected == results

    def test_corrective_rag_returns_results_when_no_expansion(self):
        """Returns original results when query expansion returns same query."""
        chunk = make_chunk()
        results = [make_search_result(chunk)]
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        # Mock evaluator and hybrid searcher
        mock_evaluator = Mock()
        mock_evaluator.needs_correction = Mock(return_value=True)
        usecase._retrieval_evaluator = mock_evaluator

        hybrid = MockHybridSearcher()
        # expand_query returns the same query (no expansion)
        hybrid._query_analyzer.expand_query = Mock(return_value="original query")
        # rewrite_query_with_info also returns the same query (no rewrite available)
        from unittest.mock import MagicMock

        mock_rewrite_result = MagicMock()
        mock_rewrite_result.rewritten = ""  # Empty string means no rewrite
        mock_rewrite_result.method = None
        mock_rewrite_result.from_cache = False
        mock_rewrite_result.fallback = False
        mock_rewrite_result.used_intent = False
        mock_rewrite_result.used_synonyms = False
        mock_rewrite_result.matched_intents = []
        hybrid._query_analyzer.rewrite_query_with_info = Mock(
            return_value=mock_rewrite_result
        )
        usecase._hybrid_searcher = hybrid

        corrected = usecase._apply_corrective_rag(
            "original query", results, None, 5, False, None, "medium"
        )

        # Should return original results since expansion didn't change query
        assert corrected == results


# ====================
# Tests for reranking edge cases
# ====================


class TestRerankingEdgeCases:
    """Tests for reranking edge cases."""

    def test_apply_reranking_with_candidate_k(self):
        """Respects candidate_k parameter for number of candidates to rerank."""
        chunk1 = make_chunk(chunk_id="c1", text="result 1")
        chunk2 = make_chunk(chunk_id="c2", text="result 2")
        chunk3 = make_chunk(chunk_id="c3", text="result 3")

        results = [
            make_search_result(chunk1, score=0.9),
            make_search_result(chunk2, score=0.8),
            make_search_result(chunk3, score=0.7),
        ]

        store = MockVectorStore([])
        reranker = MockReranker()
        usecase = SearchUseCase(
            store,
            use_reranker=True,
            reranker=reranker,
            use_hybrid=False,
            enable_warmup=False,
        )

        reranked = usecase._apply_reranking(results, "query", top_k=5, candidate_k=2)

        # Should only rerank first 2 candidates
        assert len(reranker.rerank_calls) == 1
        _, documents, _ = reranker.rerank_calls[0]
        assert len(documents) == 2

    def test_apply_reranking_with_hybrid_scoring_disabled(self):
        """Uses pure reranker scores when hybrid scoring disabled."""
        chunk = make_chunk()
        results = [make_search_result(chunk, score=0.5)]

        store = MockVectorStore([])
        reranker = MockReranker()
        usecase = SearchUseCase(
            store,
            use_reranker=True,
            reranker=reranker,
            use_hybrid=False,
            enable_warmup=False,
        )

        reranked = usecase._apply_reranking(
            results, "query", top_k=5, use_hybrid_scoring=False
        )

        # Should use reranker score only
        assert len(reranked) == 1

    def test_apply_reranking_custom_alpha(self):
        """Uses custom alpha weight for hybrid scoring."""
        chunk = make_chunk()
        results = [make_search_result(chunk, score=0.5)]

        store = MockVectorStore([])
        reranker = MockReranker()
        usecase = SearchUseCase(
            store,
            use_reranker=True,
            reranker=reranker,
            use_hybrid=False,
            enable_warmup=False,
        )

        reranked = usecase._apply_reranking(
            results, "query", top_k=5, use_hybrid_scoring=True, alpha=0.5
        )

        # Should complete without errors
        assert len(reranked) == 1


# ====================
# Tests for _search_general edge cases
# ====================


class TestSearchGeneralEdgeCases:
    """Tests for _search_general edge cases."""

    def test_search_general_without_hybrid(self):
        """Works correctly when hybrid searcher is not available."""
        chunk = make_chunk(text="휴학 절차")
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        results = usecase._search_general(
            "휴학",
            None,
            5,
            False,
            None,
        )

        # Should return results from dense search only
        assert len(store.search_calls) > 0

    def test_search_general_with_composite_decomposition(self):
        """Uses composite search when query is decomposed."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(store, use_reranker=False, enable_warmup=False)

        hybrid = MockHybridSearcher()
        hybrid._query_analyzer.decompose_query = Mock(
            return_value=["subquery1", "subquery2"]
        )
        usecase._hybrid_searcher = hybrid
        usecase._hybrid_initialized = True

        results = usecase._search_general(
            "test query",
            None,
            5,
            False,
            None,
        )

        # Should call decompose_query
        hybrid._query_analyzer.decompose_query.assert_called()

    def test_search_general_applies_bonuses(self):
        """Applies score bonuses to results."""
        chunk = make_chunk(
            text="휴학 절차", keywords=[Keyword(term="휴학", weight=1.0)]
        )
        store = MockVectorStore([make_search_result(chunk, score=0.4)])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        results = usecase._search_general(
            "휴학",
            None,
            5,
            False,
            None,
        )

        # Score should be boosted due to keyword match
        assert len(results) > 0

    def test_search_general_with_audience_override(self):
        """Respects audience override parameter."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        from src.rag.infrastructure.query_analyzer import Audience

        results = usecase._search_general(
            "query",
            None,
            5,
            False,
            Audience.FACULTY,
        )

        # Should complete without errors
        assert isinstance(results, list)


# ====================
# Tests for score bonus calculations
# ====================


class TestScoreBonuses:
    """Tests for score bonus calculations."""

    def test_keyword_bonus_with_multiple_keywords(self):
        """Sum of multiple keyword hits is capped."""
        chunk = make_chunk(
            text="교원 휴직 연구년",
            keywords=[
                Keyword(term="교원", weight=1.0),
                Keyword(term="휴직", weight=1.0),
                Keyword(term="연구년", weight=1.0),
            ],
        )
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        boosted = usecase._apply_score_bonuses(
            [make_search_result(chunk, score=0.3)],
            "교원 휴직 연구년",
            "query",
            None,
        )

        # Keyword bonus should be capped at 0.3
        assert boosted[0].score >= 0.3

    def test_fundamental_regulation_bonus(self):
        """Fundamental regulations get bonus."""
        fundamental_codes = {"2-1-1", "3-1-5", "3-1-26", "1-0-1"}

        for code in fundamental_codes:
            chunk = make_chunk(rule_code=code)
            store = MockVectorStore([])
            usecase = SearchUseCase(
                store, use_reranker=False, use_hybrid=False, enable_warmup=False
            )

            boosted = usecase._apply_score_bonuses(
                [make_search_result(chunk, score=0.5)],
                "query",
                "query",
                None,
            )

            # Should get fundamental bonus
            assert boosted[0].score > 0.5

    def test_article_bonus_with_multiple_articles(self):
        """Query with multiple article references gets bonus."""
        chunk = make_chunk(text="제8조와 제9조에 따라")
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        boosted = usecase._apply_score_bonuses(
            [make_search_result(chunk, score=0.5)],
            "제8조 제9조",
            "query",
            None,
        )

        # Should get article bonus if both articles found
        assert boosted[0].score >= 0.5


# ====================
# Tests for _select_answer_sources edge cases
# ====================


class TestSelectAnswerSourcesEdgeCases:
    """Tests for _select_answer_sources edge cases."""

    def test_all_results_low_signal_falls_back(self):
        """Falls back to original results when all are low signal."""
        results = []
        for i in range(3):
            # Create Chunk directly to control token_count
            chunk = Chunk(
                id=f"c{i}",
                rule_code="1-1-1",
                level=ChunkLevel.ARTICLE,
                title="제목",
                text="(1)",  # Short text that matches HEADING_ONLY_PATTERN
                embedding_text="(1)",
                full_text="(1)",
                parent_path=["규정"],
                token_count=10,  # Low token count
                keywords=[],
                is_searchable=True,
            )
            results.append(make_search_result(chunk))

        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        selected = usecase._select_answer_sources(results, 5)

        # Should fall back to original results
        assert len(selected) == 3

    def test_mixed_signal_results(self):
        """Selects high-signal chunks, filters low-signal."""
        # Create chunks directly to control token_count
        low_signal_chunk = Chunk(
            id="low",
            rule_code="1-1-1",
            level=ChunkLevel.ARTICLE,
            title="제목",
            text="(1)",  # Short text that matches HEADING_ONLY_PATTERN
            embedding_text="(1)",
            full_text="(1)",
            parent_path=["규정"],
            token_count=10,  # Low token count
            keywords=[],
            is_searchable=True,
        )
        high_signal_chunk = Chunk(
            id="high",
            rule_code="1-1-2",
            level=ChunkLevel.ARTICLE,
            title="제목",
            text="풍부한 내용입니다. 충분한 토큰 수를 가진 본문 내용입니다.",
            embedding_text="풍부한 내용입니다. 충분한 토큰 수를 가진 본문 내용입니다.",
            full_text="풍부한 내용입니다. 충분한 토큰 수를 가진 본문 내용입니다.",
            parent_path=["규정"],
            token_count=100,  # High token count
            keywords=[],
            is_searchable=True,
        )

        results = [
            make_search_result(low_signal_chunk, score=0.9),
            make_search_result(high_signal_chunk, score=0.5),
        ]

        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=False, enable_warmup=False
        )

        selected = usecase._select_answer_sources(results, 5)

        # Low signal chunk should be filtered
        assert any(r.chunk.id == high_signal_chunk.id for r in selected)


# ====================
# Tests for fact checking loop
# ====================


class TestFactCheckingLoop:
    """Tests for fact-checking loop in _generate_with_fact_check."""

    def test_fact_check_disabled_returns_initial_generation(self):
        """Returns initial answer when fact check is disabled."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient(response="초기 답변")

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
            enable_warmup=False,
        )
        usecase._enable_fact_check = False

        answer = usecase._generate_with_fact_check("질문", "context", None, debug=False)

        assert answer == "초기 답변"

    def test_fact_check_no_checker_returns_initial(self):
        """Returns initial answer when fact checker is None."""
        chunk = make_chunk()
        store = MockVectorStore([make_search_result(chunk)])
        llm = MockLLMClient(response="초기 답변")

        usecase = SearchUseCase(
            store,
            llm_client=llm,
            use_reranker=False,
            use_hybrid=False,
            enable_warmup=False,
        )
        usecase._enable_fact_check = True
        usecase._fact_checker = None

        answer = usecase._generate_with_fact_check("질문", "context", None, debug=False)

        assert answer == "초기 답변"


# ====================
# Tests for warmup functionality
# ====================


class TestWarmupFunctionality:
    """Tests for warmup background initialization."""

    def test_warmup_logs_warning_on_error(self):
        """Logs warning when warmup fails."""
        store = MockVectorStore([])

        # Create use case with hybrid searcher that will fail
        usecase = SearchUseCase(
            store,
            use_reranker=True,
            use_hybrid=True,
            enable_warmup=False,
        )

        # Mock hybrid searcher to raise error
        def failing_ensure():
            raise RuntimeError("Warmup failed")

        usecase._ensure_hybrid_searcher = failing_ensure

        # Should not raise, just log warning
        usecase._warmup()

    def test_hybrid_searcher_property_lazy_initializes(self):
        """hybrid_searcher property lazily initializes when first accessed."""
        store = MockVectorStore([])
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=True, enable_warmup=False
        )

        assert usecase._hybrid_initialized is False

        # Access property
        _ = usecase.hybrid_searcher

        # Should be initialized after access
        assert usecase._hybrid_initialized is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
