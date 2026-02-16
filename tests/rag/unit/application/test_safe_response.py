"""
Unit tests for Safe Response Generator (SPEC-RAG-QUALITY-004).

Tests for the _generate_safe_response method in SearchUseCase.
Validates that safe responses are generated correctly when
faithfulness is too low.
"""

import pytest

from src.rag.domain.entities import Chunk, ChunkLevel, SearchResult


class TestSafeResponseGenerator:
    """Test _generate_safe_response method."""

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results for testing."""
        chunk1 = Chunk(
            id="test-1",
            text="학칙 제10조: 등록에 관한 규정입니다.",
            rule_code="학칙",
            level=ChunkLevel.ARTICLE,
            title="등록",
            embedding_text="학칙 제10조: 등록에 관한 규정입니다.",
            full_text="학칙 제10조: 등록에 관한 규정입니다.",
            parent_path=[],
            token_count=20,
            keywords=[],
            is_searchable=True,
            article_number="10",
        )
        chunk2 = Chunk(
            id="test-2",
            text="교원인사규정 제15조: 승진에 관한 규정입니다.",
            rule_code="교원인사규정",
            level=ChunkLevel.ARTICLE,
            title="승진",
            embedding_text="교원인사규정 제15조: 승진에 관한 규정입니다.",
            full_text="교원인사규정 제15조: 승진에 관한 규정입니다.",
            parent_path=[],
            token_count=20,
            keywords=[],
            is_searchable=True,
            article_number="15",
        )

        return [
            SearchResult(chunk=chunk1, score=0.8, rank=1),
            SearchResult(chunk=chunk2, score=0.7, rank=2),
        ]

    def test_safe_response_includes_base_message(self, mock_search_results):
        """
        WHEN safe response is generated
        THEN it should include the base apology message
        """
        from src.rag.application.search_usecase import SearchUseCase

        # Create a minimal SearchUseCase instance
        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="휴학은 어떻게 신청하나요?",
            sources=mock_search_results,
            faithfulness_score=0.15,
        )

        # Check base message is present
        assert "죄송합니다" in safe_response
        assert "찾을 수 없습니다" in safe_response

    def test_safe_response_includes_related_regulations(self, mock_search_results):
        """
        WHEN sources contain regulation names
        THEN safe response should include related regulations
        """
        from src.rag.application.search_usecase import SearchUseCase

        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="휴학은 어떻게 신청하나요?",
            sources=mock_search_results,
            faithfulness_score=0.15,
        )

        # Check regulation names are included
        assert "학칙" in safe_response
        assert "교원인사규정" in safe_response

    def test_safe_response_includes_article_references(self, mock_search_results):
        """
        WHEN sources contain article references
        THEN safe response should include article references
        """
        from src.rag.application.search_usecase import SearchUseCase

        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="휴학은 어떻게 신청하나요?",
            sources=mock_search_results,
            faithfulness_score=0.15,
        )

        # Check article references are included
        assert "제10조" in safe_response or "제15조" in safe_response

    def test_safe_response_includes_guidance(self, mock_search_results):
        """
        WHEN safe response is generated
        THEN it should include guidance for further assistance
        """
        from src.rag.application.search_usecase import SearchUseCase

        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="휴학은 어떻게 신청하나요?",
            sources=mock_search_results,
            faithfulness_score=0.15,
        )

        # Check guidance is present
        assert "다른 표현" in safe_response
        assert "부서" in safe_response

    def test_safe_response_includes_faithfulness_score(self, mock_search_results):
        """
        WHEN safe response is generated
        THEN it should include the faithfulness score
        """
        from src.rag.application.search_usecase import SearchUseCase

        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="휴학은 어떻게 신청하나요?",
            sources=mock_search_results,
            faithfulness_score=0.15,
        )

        # Check faithfulness score is shown
        assert "0.15" in safe_response
        assert "신뢰도" in safe_response

    def test_safe_response_empty_sources(self):
        """
        WHEN no sources are available
        THEN safe response should still be generated
        """
        from src.rag.application.search_usecase import SearchUseCase

        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="휴학은 어떻게 신청하나요?",
            sources=[],
            faithfulness_score=0.10,
        )

        # Should still have base message
        assert "죄송합니다" in safe_response
        # Should still have guidance
        assert "도움을 받을 수 있는 방법" in safe_response


class TestSafeResponseWithMockedUsecase:
    """Test safe response generation with mocked SearchUseCase."""

    def test_safe_response_structure(self):
        """
        WHEN safe response is generated
        THEN it should have proper structure
        """
        from src.rag.application.search_usecase import SearchUseCase

        # Create a mock chunk with all required attributes
        chunk = Chunk(
            id="test-id",
            text="테스트 텍스트",
            rule_code="테스트규정",
            level=ChunkLevel.ARTICLE,
            title="테스트",
            embedding_text="테스트 텍스트",
            full_text="테스트 텍스트",
            parent_path=[],
            token_count=10,
            keywords=[],
            is_searchable=True,
            article_number="1",
        )

        source = SearchResult(chunk=chunk, score=0.5, rank=1)

        usecase = object.__new__(SearchUseCase)
        usecase._generate_safe_response = SearchUseCase._generate_safe_response.__get__(
            usecase, SearchUseCase
        )

        safe_response = usecase._generate_safe_response(
            question="질문",
            sources=[source],
            faithfulness_score=0.20,
        )

        # Check structure
        lines = safe_response.split("\n")
        assert len(lines) >= 3  # Should have multiple lines

        # Check sections exist
        assert any("관련 규정" in line for line in lines)
        assert any("도움을 받을 수 있는 방법" in line for line in lines)
