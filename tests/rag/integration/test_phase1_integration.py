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
        llm.stream_generate.return_value = iter(["테스트", "답변"])
        llm.generate.return_value = "테스트 답변"
        return llm

    def test_query_expansion_called_in_ask_stream(self, mock_store, mock_llm_client):
        """Verify that query expansion is called in ask_stream."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Mock all the internal methods that ask_stream calls
        with patch.object(
            usecase,
            "_apply_dynamic_expansion",
            return_value=("test query", ["test", "expansion"]),
        ) as mock_expansion:
            with patch.object(usecase, "search", return_value=[]):
                # Call ask_stream - it will return early due to empty results
                list(usecase.ask_stream(question="테스트 질문", top_k=3))

                # Verify query expansion was called
                mock_expansion.assert_called_once()

    def test_citation_enhancement_called_in_ask_stream(self, mock_store, mock_llm_client):
        """Verify that citation enhancement is called in ask_stream."""
        from src.rag.application.search_usecase import SearchUseCase
        from src.rag.domain.entities import Chunk, SearchResult

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Create a proper mock chunk with all required attributes
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "test-id"
        mock_chunk.text = "test content"
        mock_chunk.rule_code = "TEST-001"
        mock_chunk.title = "Test Regulation"
        mock_chunk.parent_path = ["Test Regulation"]

        mock_result = SearchResult(chunk=mock_chunk, score=0.8, rank=1)

        with patch.object(
            usecase,
            "_apply_dynamic_expansion",
            return_value=("test query", ["test", "expansion"]),
        ):
            with patch.object(usecase, "search", return_value=[mock_result]):
                with patch.object(
                    usecase, "_select_answer_sources", return_value=[mock_result]
                ):
                    with patch.object(
                        usecase,
                        "_build_context",
                        return_value="test context",
                    ):
                        with patch.object(
                            usecase,
                            "_build_user_message",
                            return_value="test message",
                        ):
                            with patch.object(
                                usecase,
                                "_compute_confidence",
                                return_value=0.8,
                            ):
                                with patch.object(
                                    usecase,
                                    "_verify_citations",
                                    return_value="test answer",
                                ):
                                    with patch.object(
                                        usecase,
                                        "_enhance_answer_citations",
                                        return_value="enhanced answer",
                                    ) as mock_citation:
                                        # Call ask_stream
                                        list(
                                            usecase.ask_stream(
                                                question="테스트 질문", top_k=3
                                            )
                                        )

                                        # Verify citation enhancement was called
                                        mock_citation.assert_called_once()

    def test_query_expansion_in_non_streaming_ask(self, mock_store, mock_llm_client):
        """Verify that query expansion is called in non-streaming ask."""
        from src.rag.application.search_usecase import SearchUseCase
        from src.rag.domain.entities import Chunk, SearchResult

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Create a proper mock chunk with all required attributes
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "test-id"
        mock_chunk.text = "test content"
        mock_chunk.rule_code = "TEST-001"
        mock_chunk.title = "Test Regulation"
        mock_chunk.parent_path = ["Test Regulation"]

        mock_result = SearchResult(chunk=mock_chunk, score=0.8, rank=1)

        with patch.object(
            usecase,
            "_apply_dynamic_expansion",
            return_value=("test query", ["test"]),
        ) as mock_expansion:
            with patch.object(usecase, "search", return_value=[mock_result]):
                with patch.object(
                    usecase,
                    "_apply_self_rag_relevance_filter",
                    return_value=[mock_result],
                ):
                    with patch.object(
                        usecase, "_select_answer_sources", return_value=[mock_result]
                    ):
                        with patch.object(
                            usecase,
                            "_build_context",
                            return_value="test context",
                        ):
                            with patch.object(
                                usecase,
                                "_build_user_message",
                                return_value="test message",
                            ):
                                with patch.object(
                                    usecase,
                                    "_compute_confidence",
                                    return_value=0.8,
                                ):
                                    with patch.object(
                                        usecase,
                                        "_generate_with_fact_check",
                                        return_value="test answer",
                                    ):
                                        with patch.object(
                                            usecase,
                                            "_verify_citations",
                                            return_value="test answer",
                                        ):
                                            with patch.object(
                                                usecase,
                                                "_enhance_answer_citations",
                                                return_value="enhanced answer",
                                            ):
                                                with patch.object(
                                                    usecase,
                                                    "_is_period_related_query",
                                                    return_value=False,
                                                ):
                                                    with patch.object(
                                                        usecase,
                                                        "_detect_audience",
                                                        return_value=None,
                                                    ):
                                                        # Call ask
                                                        usecase.ask(
                                                            question="테스트 질문",
                                                            top_k=3,
                                                        )

                                                        # Verify query expansion was called
                                                        mock_expansion.assert_called_once()

    def test_citation_enhancement_in_non_streaming_ask(
        self, mock_store, mock_llm_client
    ):
        """Verify that citation enhancement is called in non-streaming ask."""
        from src.rag.application.search_usecase import SearchUseCase
        from src.rag.domain.entities import Chunk, SearchResult

        usecase = SearchUseCase(mock_store, llm_client=mock_llm_client)

        # Create a proper mock chunk with all required attributes
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "test-id"
        mock_chunk.text = "test content"
        mock_chunk.rule_code = "TEST-001"
        mock_chunk.title = "Test Regulation"
        mock_chunk.parent_path = ["Test Regulation"]

        mock_result = SearchResult(chunk=mock_chunk, score=0.8, rank=1)

        with patch.object(
            usecase,
            "_apply_dynamic_expansion",
            return_value=("test query", ["test"]),
        ):
            with patch.object(usecase, "search", return_value=[mock_result]):
                with patch.object(
                    usecase,
                    "_apply_self_rag_relevance_filter",
                    return_value=[mock_result],
                ):
                    with patch.object(
                        usecase, "_select_answer_sources", return_value=[mock_result]
                    ):
                        with patch.object(
                            usecase,
                            "_build_context",
                            return_value="test context",
                        ):
                            with patch.object(
                                usecase,
                                "_build_user_message",
                                return_value="test message",
                            ):
                                with patch.object(
                                    usecase,
                                    "_compute_confidence",
                                    return_value=0.8,
                                ):
                                    with patch.object(
                                        usecase,
                                        "_generate_with_fact_check",
                                        return_value="test answer",
                                    ):
                                        with patch.object(
                                            usecase,
                                            "_verify_citations",
                                            return_value="test answer",
                                        ):
                                            with patch.object(
                                                usecase,
                                                "_enhance_answer_citations",
                                                return_value="enhanced answer",
                                            ) as mock_citation:
                                                with patch.object(
                                                    usecase,
                                                    "_is_period_related_query",
                                                    return_value=False,
                                                ):
                                                    with patch.object(
                                                        usecase,
                                                        "_detect_audience",
                                                        return_value=None,
                                                    ):
                                                        # Call ask
                                                        usecase.ask(
                                                            question="테스트 질문",
                                                            top_k=3,
                                                        )

                                                        # Verify citation enhancement was called
                                                        mock_citation.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
