"""
Characterization tests for Citation Verification integration in SearchUsecase.

SPEC-RAG-Q-004: These tests verify the integration of CitationVerificationService
into the search_usecase.py answer generation flow.

Tests focus on:
- TASK-012: _verify_citations() method behavior
- TASK-013: Integration in answer flow
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.entities import Chunk, SearchResult


def create_mock_chunk(
    text: str,
    regulation_name: str = "학칙",
    article: int = 25,
    paragraph: int = None,
) -> Chunk:
    """Create a mock Chunk with regulation metadata."""
    chunk = MagicMock(spec=Chunk)
    chunk.text = text
    chunk.regulation_name = regulation_name
    chunk.article = article
    chunk.paragraph = paragraph
    chunk.id = f"{regulation_name}-{article}"
    return chunk


def create_mock_search_result(
    text: str,
    regulation_name: str = "학칙",
    article: int = 25,
    paragraph: int = None,
    score: float = 0.9,
) -> SearchResult:
    """Create a mock SearchResult with a mock Chunk."""
    chunk = create_mock_chunk(text, regulation_name, article, paragraph)
    return SearchResult(chunk=chunk, score=score, rank=1)


class TestVerifyCitationsMethod:
    """Tests for _verify_citations() method (TASK-012)."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUsecase with mocked dependencies."""
        mock_store = MagicMock()
        mock_llm = MagicMock()

        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
        )
        return usecase

    def test_verify_citations_returns_original_answer_if_no_citations(self, usecase):
        """Test _verify_citations returns original if answer has no citations."""
        answer = "일반적인 답변입니다. 인용이 없습니다."
        sources = [create_mock_search_result("휴학 관련 내용")]

        result = usecase._verify_citations(answer, sources)

        assert result == answer

    def test_verify_citations_returns_original_answer_if_no_sources(self, usecase):
        """Test _verify_citations returns original if no source chunks provided."""
        answer = "「학칙」 제25조에 따르면 휴학은 가능합니다."

        result = usecase._verify_citations(answer, [])

        assert result == answer

    def test_verify_citations_preserves_verified_citation(self, usecase):
        """Test _verify_citations preserves citations that are in source chunks."""
        answer = "「학칙」 제25조에 따르면 휴학은 2년을 초과할 수 없습니다."
        sources = [
            create_mock_search_result(
                text="휴학은 2년을 초과할 수 없다.",
                regulation_name="학칙",
                article=25,
            )
        ]

        result = usecase._verify_citations(answer, sources)

        # Verified citation should be preserved
        assert "「학칙」 제25조" in result

    def test_verify_citations_sanitizes_unverifiable_citation(self, usecase):
        """Test _verify_citations sanitizes citations not in source chunks."""
        answer = "「학칙」 제99조에 따르면 특별한 규정이 있습니다."
        sources = [
            create_mock_search_result(
                text="휴학은 2년을 초과할 수 없다.",
                regulation_name="학칙",
                article=25,  # Different article - citation unverifiable
            )
        ]

        result = usecase._verify_citations(answer, sources)

        # Unverifiable citation should be replaced
        assert "「학칙」 제99조" not in result
        assert "관련 규정에 따르면" in result

    def test_verify_citations_handles_multiple_citations(self, usecase):
        """Test _verify_citations handles multiple citations correctly."""
        answer = "「학칙」 제25조와 「등록금에 관한 규정」 제4조에 따르면..."
        sources = [
            create_mock_search_result(
                text="휴학은 2년을 초과할 수 없다.",
                regulation_name="학칙",
                article=25,
            ),
            # Article 4 is not in sources, so it should be sanitized
        ]

        result = usecase._verify_citations(answer, sources)

        # First citation verified, second not
        assert "「학칙」 제25조" in result

    def test_verify_citations_empty_answer_returns_empty(self, usecase):
        """Test _verify_citations handles empty answer."""
        sources = [create_mock_search_result("내용")]

        result = usecase._verify_citations("", sources)

        assert result == ""

    def test_verify_citations_with_paragraph_in_citation(self, usecase):
        """Test _verify_citations handles citations with paragraph (항)."""
        answer = "「등록금에 관한 규정」 제4조 제2항에 따르면..."
        sources = [
            create_mock_search_result(
                text="등록금 납부 관련 내용",
                regulation_name="등록금에 관한 규정",
                article=4,
                paragraph=2,
            )
        ]

        result = usecase._verify_citations(answer, sources)

        # Citation with paragraph should be verified
        assert "「등록금에 관한 규정」 제4조" in result

    def test_verify_citations_graceful_failure_on_import_error(self, usecase):
        """Test _verify_citations returns original answer on import error."""
        answer = "「학칙」 제25조에 따르면..."
        sources = [create_mock_search_result("내용")]

        with patch.object(
            usecase,
            "_ensure_citation_verification_service",
            side_effect=ImportError("No module"),
        ):
            result = usecase._verify_citations(answer, sources)

        assert result == answer

    def test_verify_citations_graceful_failure_on_exception(self, usecase):
        """Test _verify_citations returns original answer on general exception."""
        answer = "「학칙」 제25조에 따르면..."
        sources = [create_mock_search_result("내용")]

        with patch.object(
            usecase,
            "_ensure_citation_verification_service",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = usecase._verify_citations(answer, sources)

        assert result == answer


class TestCitationVerificationServiceIntegration:
    """Tests for CitationVerificationService lazy initialization."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUsecase with mocked dependencies."""
        mock_store = MagicMock()
        mock_llm = MagicMock()

        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
        )
        return usecase

    def test_ensure_citation_verification_service_creates_instance(self, usecase):
        """Test _ensure_citation_verification_service creates service."""
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = usecase._ensure_citation_verification_service()

        assert isinstance(service, CitationVerificationService)

    def test_ensure_citation_verification_service_returns_cached_instance(self, usecase):
        """Test _ensure_citation_verification_service returns cached instance."""
        service1 = usecase._ensure_citation_verification_service()
        service2 = usecase._ensure_citation_verification_service()

        assert service1 is service2

    def test_citation_verification_service_initialized_as_none(self, usecase):
        """Test _citation_verification_service is None initially."""
        assert usecase._citation_verification_service is None


class TestVerifyCitationsWithChunkMetadata:
    """Tests for _verify_citations with various chunk metadata scenarios."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUsecase with mocked dependencies."""
        mock_store = MagicMock()
        mock_llm = MagicMock()

        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
        )
        return usecase

    def test_verify_citations_handles_chunk_without_regulation_name(self, usecase):
        """Test _verify_citations handles chunks without regulation_name."""
        answer = "「학칙」 제25조에 따르면..."
        chunk = create_mock_chunk("내용")
        chunk.regulation_name = None  # Missing metadata
        sources = [SearchResult(chunk=chunk, score=0.9, rank=1)]

        result = usecase._verify_citations(answer, sources)

        # Citation should be sanitized since no match found
        assert "「학칙」 제25조" not in result

    def test_verify_citations_handles_chunk_without_article(self, usecase):
        """Test _verify_citations handles chunks without article number."""
        answer = "「학칙」 제25조에 따르면..."
        chunk = create_mock_chunk("내용", regulation_name="학칙")
        chunk.article = None  # Missing metadata
        sources = [SearchResult(chunk=chunk, score=0.9, rank=1)]

        result = usecase._verify_citations(answer, sources)

        # Citation should be sanitized since no match found
        assert "「학칙」 제25조" not in result

    def test_verify_citations_matches_regulation_and_article(self, usecase):
        """Test _verify_citations correctly matches both regulation and article."""
        answer = "「학칙」 제25조에 따르면..."
        sources = [
            create_mock_search_result("내용1", "학칙", 25),  # Match
            create_mock_search_result("내용2", "학칙", 30),  # Different article
            create_mock_search_result("내용3", "다른규정", 25),  # Different regulation
        ]

        result = usecase._verify_citations(answer, sources)

        # Should find match in first source
        assert "「학칙」 제25조" in result


class TestVerifyCitationsComplexCases:
    """Tests for complex citation verification scenarios."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUsecase with mocked dependencies."""
        mock_store = MagicMock()
        mock_llm = MagicMock()

        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
        )
        return usecase

    def test_verify_citations_with_complex_regulation_name(self, usecase):
        """Test _verify_citations handles complex regulation names."""
        answer = "「졸업논문또는졸업실적심사규정」 제8조에 따르면..."
        sources = [
            create_mock_search_result(
                text="졸업 논문 심사 기준",
                regulation_name="졸업논문또는졸업실적심사규정",
                article=8,
            )
        ]

        result = usecase._verify_citations(answer, sources)

        assert "「졸업논문또는졸업실적심사규정」 제8조" in result

    def test_verify_citations_preserves_answer_context(self, usecase):
        """Test _verify_citations preserves surrounding text."""
        answer = "안녕하세요. 「학칙」 제99조에 따르면 특별 규정입니다. 감사합니다."
        sources = [
            create_mock_search_result("내용", "학칙", 25)
        ]

        result = usecase._verify_citations(answer, sources)

        # Surrounding text should be preserved
        assert "안녕하세요." in result
        assert "감사합니다." in result
        # Only citation should change
        assert "「학칙」 제99조" not in result

    def test_verify_citations_with_sub_article_format(self, usecase):
        """Test _verify_citations handles sub-article format (제X조의Y)."""
        answer = "「교원인사규정」 제10조의2에 따르면..."
        sources = [
            create_mock_search_result(
                text="교원 인사 관련 내용",
                regulation_name="교원인사규정",
                article=10,
            )
        ]

        result = usecase._verify_citations(answer, sources)

        # Article 10 should match (sub-article verification is lenient)
        assert "「교원인사규정」 제10조" in result or "관련 규정에 따르면" in result
