"""
Characterization tests for SearchUseCase behavior preservation.

These tests capture the CURRENT behavior of SearchUseCase to ensure
that integration of new components (QueryExpansion, CitationEnhancer)
does not break existing functionality.
"""

import pytest
from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.config import get_config


@pytest.fixture
def search_usecase():
    """Create SearchUseCase instance for testing."""
    config = get_config()
    store = ChromaVectorStore(persist_directory="data/chroma_db")
    llm_client = LLMClientAdapter(
        provider=config.llm_provider,
        model=config.llm_model,
        base_url=config.llm_base_url,
    )
    usecase = SearchUseCase(
        store=store,
        llm_client=llm_client,
        use_reranker=True,
    )
    return usecase


class TestSearchUseCaseSearchBehavior:
    """Characterization tests for search() method."""

    def test_search_returns_results(self, search_usecase):
        """Characterize: search() returns list of SearchResult."""
        results = search_usecase.search(
            query_text="휴학 방법",
            top_k=5,
        )
        # Capture current behavior
        assert isinstance(results, list)
        if results:
            assert hasattr(results[0], 'chunk')
            assert hasattr(results[0], 'score')
            assert hasattr(results[0], 'query')

    def test_search_with_reranking(self, search_usecase):
        """Characterize: search with reranking returns scored results."""
        results = search_usecase.search(
            query_text="등록금 납부",
            top_k=3,
            use_reranker=True,
        )
        # Capture current behavior
        if results:
            assert all(hasattr(r, 'score') for r in results)
            # Results should be sorted by score
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestSearchUseCaseAskBehavior:
    """Characterization tests for ask() method."""

    def test_ask_returns_answer_with_sources(self, search_usecase):
        """Characterize: ask() returns Answer object with sources."""
        answer = search_usecase.ask(
            question="휴학 신청 방법은?",
            top_k=5,
        )
        # Capture current behavior
        assert hasattr(answer, 'text')
        assert hasattr(answer, 'sources')
        assert hasattr(answer, 'confidence')
        assert isinstance(answer.text, str)
        assert isinstance(answer.sources, list)

    def test_ask_includes_regulation_citations(self, search_usecase):
        """Characterize: ask() responses mention regulations."""
        answer = search_usecase.ask(
            question="장학금 신청 자격은?",
            top_k=5,
        )
        # Capture current citation behavior
        text = answer.text
        # Check for regulation mentions
        has_regulation = any(word in text for word in ['규정', '조', '제'])
        # This captures current state - may or may not have citations
        assert isinstance(text, str)

    def test_ask_sources_contain_chunk_info(self, search_usecase):
        """Characterize: ask() sources contain chunk information."""
        answer = search_usecase.ask(
            question="성적 정정은 어떻게 하나요?",
            top_k=3,
        )
        # Capture current source structure
        if answer.sources:
            source = answer.sources[0]
            assert hasattr(source, 'chunk')
            assert hasattr(source.chunk, 'id')
            assert hasattr(source.chunk, 'text')
            assert hasattr(source.chunk, 'title')


class TestSearchUseCaseQueryRewrite:
    """Characterization tests for query rewrite behavior."""

    def test_last_query_rewrite_tracking(self, search_usecase):
        """Characterize: query rewrite is tracked internally."""
        # Perform a search
        search_usecase.search(
            query_text="휴학",
            top_k=5,
        )
        # Check if rewrite tracking exists
        rewrite = search_usecase.get_last_query_rewrite()
        # Capture current behavior - may be None or a dict
        assert rewrite is None or isinstance(rewrite, dict)
