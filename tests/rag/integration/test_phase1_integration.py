"""Integration test for Phase 1 RAG pipeline components."""
import pytest
from unittest.mock import Mock, MagicMock, patch


class TestPhase1Integration:
    """Test that QueryExpansionService and CitationEnhancer are properly integrated."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store."""
        store = Mock()
        store.count.return_value = 0
        store.search.return_value = []
        return store

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        llm = Mock()
        llm.stream_generate.return_value = ["테스트", "답변"]
        llm.generate.return_value = "테스트 답변"
        return llm

    def test_query_expansion_called_in_ask_stream(self, mock_store, mock_llm_client):
        """Verify that query expansion is called in ask_stream."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Mock the query expansion method to track calls
        original_method = usecase._apply_dynamic_expansion
        usecase._apply_dynamic_expansion = Mock(return_value=("test query", ["test", "expansion"]))

        # Call ask_stream
        list(usecase.ask_stream(question="테스트 질문", top_k=3))

        # Verify query expansion was called
        usecase._apply_dynamic_expansion.assert_called_once()

    def test_citation_enhancement_called_in_ask_stream(self, mock_store, mock_llm_client):
        """Verify that citation enhancement is called in ask_stream."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Mock the citation enhancement method to track calls
        original_method = usecase._enhance_answer_citations
        usecase._enhance_answer_citations = Mock(return_value="enhanced answer")

        # Setup mock to return some results
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "test-id"
        mock_chunk.text = "test content"
        mock_chunk.rule_code = "TEST-001"
        mock_chunk.title = "Test Regulation"
        mock_chunk.parent_path = ["Test Regulation"]

        mock_result = SearchResult(chunk=mock_chunk, score=0.8, rank=1)
        mock_store.search.return_value = [mock_result]

        # Call ask_stream
        events = list(usecase.ask_stream(question="테스트 질문", top_k=3))

        # Verify citation enhancement was called
        usecase._enhance_answer_citations.assert_called_once()

    def test_query_expansion_in_non_streaming_ask(self, mock_store, mock_llm_client):
        """Verify that query expansion is called in non-streaming ask."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Mock the query expansion method
        usecase._apply_dynamic_expansion = Mock(return_value=("test query", ["test"]))

        # Setup mock to return results
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "test-id"
        mock_chunk.text = "test content"
        mock_chunk.rule_code = "TEST-001"
        mock_chunk.title = "Test Regulation"
        mock_chunk.parent_path = ["Test Regulation"]

        mock_result = SearchResult(chunk=mock_chunk, score=0.8, rank=1)
        mock_store.search.return_value = [mock_result]

        # Call ask
        usecase.ask(question="테스트 질문", top_k=3)

        # Verify query expansion was called
        usecase._apply_dynamic_expansion.assert_called_once()

    def test_citation_enhancement_in_non_streaming_ask(self, mock_store, mock_llm_client):
        """Verify that citation enhancement is called in non-streaming ask."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Mock the citation enhancement method
        usecase._enhance_answer_citations = Mock(return_value="enhanced answer")

        # Setup mock to return results
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "test-id"
        mock_chunk.text = "test content"
        mock_chunk.rule_code = "TEST-001"
        mock_chunk.title = "Test Regulation"
        mock_chunk.parent_path = ["Test Regulation"]

        mock_result = SearchResult(chunk=mock_chunk, score=0.8, rank=1)
        mock_store.search.return_value = [mock_result]

        # Call ask
        answer = usecase.ask(question="테스트 질문", top_k=3)

        # Verify citation enhancement was called
        usecase._enhance_answer_citations.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
