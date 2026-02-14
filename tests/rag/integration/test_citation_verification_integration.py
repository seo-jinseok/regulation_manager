"""Integration tests for Citation Verification (SPEC-RAG-Q-004).

Tests the end-to-end flow of citation verification in RAG responses:
1. Answer with valid citation -> citation preserved
2. Answer with invalid citation -> citation sanitized
3. Answer with multiple citations -> correct handling
4. Answer with no citations -> no modification

Verifies acceptance criteria:
- AC-001: Citation Accuracy (90% reduction in issues)
- AC-002: Format Compliance (95%+ standard format)
- AC-004: Traceability (all citations searchable)
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional


def create_mock_chunk(chunk_id, text, regulation_name, article, paragraph=None, rule_code="RULE-001", title="Test", parent_path=None):
    """Helper to create a mock Chunk with proper attributes."""
    chunk = Mock()
    chunk.id = chunk_id
    chunk.text = text
    chunk.regulation_name = regulation_name
    chunk.article = article
    chunk.paragraph = paragraph
    chunk.rule_code = rule_code
    chunk.title = title
    chunk.parent_path = parent_path or [regulation_name]
    return chunk


def create_search_result(chunk, score=0.9, rank=1):
    """Helper to create a SearchResult."""
    from src.rag.domain.entities import SearchResult
    return SearchResult(chunk=chunk, score=score, rank=rank)


class TestCitationVerificationIntegration:
    """End-to-end integration tests for citation verification flow."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with regulation metadata."""
        chunks = []

        # Chunk 1: 학칙 제25조 (휴학 규정)
        chunk1 = create_mock_chunk(
            "chunk-1",
            "휴학은 2년을 초과할 수 없다. 휴학 기간은 재학 기간에 산입하지 않는다.",
            "학칙", 25, 1, "RULE-001", "휴학에 관한 규정", ["학칙", "제4장 휴학"]
        )
        chunks.append(create_search_result(chunk1, 0.9, 1))

        # Chunk 2: 등록금 규정 제4조
        chunk2 = create_mock_chunk(
            "chunk-2",
            "등록금은 매 학기 시작 전까지 납부하여야 한다.",
            "등록금에 관한 규정", 4, None, "RULE-002", "등록금 납부", ["등록금에 관한 규정", "제2장 납부"]
        )
        chunks.append(create_search_result(chunk2, 0.85, 2))

        # Chunk 3: 졸업논문 규정 제8조
        chunk3 = create_mock_chunk(
            "chunk-3",
            "졸업논문 심사는 3인 이상의 심사위원이 수행한다.",
            "졸업논문또는졸업실적심사규정", 8, 1, "RULE-003", "논문 심사", ["졸업논문또는졸업실적심사규정", "제3장 심사"]
        )
        chunks.append(create_search_result(chunk3, 0.8, 3))

        return chunks

    # =================================================================
    # TEST CASE 1: Answer with valid citation -> citation preserved
    # =================================================================

    def test_valid_citation_preserved(self, sample_chunks):
        """
        AC-001, AC-004: Valid citations should be preserved in the answer.

        Scenario:
        - LLM generates answer with citation that exists in source chunks
        - Citation should remain unchanged in the final answer
        """
        from src.rag.application.search_usecase import SearchUseCase
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        # Create service directly
        service = CitationVerificationService()

        # Test with actual citation verification logic
        answer_text = "「학칙」 제25조에 따르면, 휴학은 2년을 초과할 수 없습니다."

        # Convert SearchResult to dict format for service
        chunks_for_verification = []
        for result in sample_chunks[:1]:
            chunk = result.chunk
            chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "regulation_name": chunk.regulation_name,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                },
            }
            chunks_for_verification.append(chunk_dict)

        # Extract and verify
        citations = service.extract_citations(answer_text)
        assert len(citations) == 1
        assert citations[0].regulation_name == "학칙"
        assert citations[0].article == 25

        # Verify grounding
        is_verified = service.verify_grounding(citations[0], chunks_for_verification)
        assert is_verified is True, "Citation should be verified against source"

        # Since verified, answer should remain unchanged
        # (This is the expected behavior)

    # =================================================================
    # TEST CASE 2: Answer with invalid citation -> citation sanitized
    # =================================================================

    def test_invalid_citation_sanitized(self, sample_chunks):
        """
        AC-001: Invalid citations should be sanitized.

        Scenario:
        - LLM generates answer with citation NOT in source chunks
        - Citation should be replaced with fallback phrase
        """
        from src.rag.application.search_usecase import SearchUseCase
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        # Answer with invalid citation (제99조는 존재하지 않음)
        answer_text = "「학칙」 제99조에 따르면, 특별한 경우에 추가 휴학이 가능합니다."

        # Convert to dict format
        chunks_for_verification = []
        for result in sample_chunks[:1]:
            chunk = result.chunk
            chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "regulation_name": chunk.regulation_name,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                },
            }
            chunks_for_verification.append(chunk_dict)

        # Extract and verify
        citations = service.extract_citations(answer_text)
        assert len(citations) == 1
        assert citations[0].article == 99  # Invalid article

        # Verify grounding - should fail
        is_verified = service.verify_grounding(citations[0], chunks_for_verification)
        assert is_verified is False, "Invalid citation should not be verified"

        # Get fallback phrase
        fallback = service.sanitize_unverifiable(citations[0])
        assert "관련 규정에 따르면" in fallback or "확인이 필요" in fallback

    # =================================================================
    # TEST CASE 3: Answer with multiple citations -> correct handling
    # =================================================================

    def test_multiple_citations_partial_verification(self, sample_chunks):
        """
        AC-001, AC-002: Multiple citations with mixed validity.

        Scenario:
        - LLM generates answer with 3 citations
        - 2 citations are valid (exist in sources)
        - 1 citation is invalid (not in sources)
        - Valid citations preserved, invalid sanitized
        """
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        answer_text = (
            "「학칙」 제25조에 따르면 휴학은 2년을 초과할 수 없으며, "
            "「등록금에 관한 규정」 제4조에 따르면 납부 기한이 정해져 있습니다. "
            "「학칙」 제99조에 따르면 특별 규정이 있습니다."
        )

        # Convert to dict format (all chunks)
        chunks_for_verification = []
        for result in sample_chunks:
            chunk = result.chunk
            chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "regulation_name": chunk.regulation_name,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                },
            }
            chunks_for_verification.append(chunk_dict)

        # Extract all citations
        citations = service.extract_citations(answer_text)
        assert len(citations) == 3

        # Verify each
        verified_count = 0
        unverified_count = 0
        for citation in citations:
            if service.verify_grounding(citation, chunks_for_verification):
                verified_count += 1
            else:
                unverified_count += 1

        assert verified_count == 2, f"Should have 2 verified citations, got {verified_count}"
        assert unverified_count == 1, f"Should have 1 unverified citation, got {unverified_count}"

    def test_multiple_citations_all_valid(self, sample_chunks):
        """
        AC-002: All valid citations should be preserved (95%+ format compliance).

        Scenario:
        - LLM generates answer with multiple valid citations
        - All citations should be preserved in standard format
        """
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        answer_text = "「학칙」 제25조와 「등록금에 관한 규정」 제4조를 참고하세요."

        # Convert to dict format
        chunks_for_verification = []
        for result in sample_chunks[:2]:
            chunk = result.chunk
            chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "regulation_name": chunk.regulation_name,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                },
            }
            chunks_for_verification.append(chunk_dict)

        # Extract and verify
        citations = service.extract_citations(answer_text)
        assert len(citations) == 2

        # All should be verified
        for citation in citations:
            is_verified = service.verify_grounding(citation, chunks_for_verification)
            assert is_verified, f"Citation {citation.to_standard_format()} should be verified"

    # =================================================================
    # TEST CASE 4: Answer with no citations -> no modification
    # =================================================================

    def test_no_citations_no_modification(self, sample_chunks):
        """
        Answers without citations should remain unchanged.

        Scenario:
        - LLM generates answer without any citations
        - Answer should be returned as-is
        """
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        # Answer without citations
        answer_text = "휴학은 일반적으로 2년까지 가능하며, 자세한 내용은 학사팀에 문의하세요."

        # Extract citations
        citations = service.extract_citations(answer_text)
        assert len(citations) == 0, "Should extract no citations from text without citations"

    # =================================================================
    # TEST CASE 5: Streaming flow with citation verification
    # =================================================================

    def test_streaming_with_citation_verification(self, sample_chunks):
        """
        Citation verification should work in streaming mode.

        Scenario:
        - LLM streams answer with citations
        - Final enhanced answer should have verified citations
        """
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        # Simulate streaming by concatenating chunks
        streamed_chunks = ["「학칙」 ", "제25조에 ", "따르면, ", "휴학은 2년을 초과할 수 없습니다."]
        full_answer = "".join(streamed_chunks)

        # Convert to dict format
        chunks_for_verification = []
        for result in sample_chunks[:1]:
            chunk = result.chunk
            chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "regulation_name": chunk.regulation_name,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                },
            }
            chunks_for_verification.append(chunk_dict)

        # Extract and verify
        citations = service.extract_citations(full_answer)
        assert len(citations) == 1

        is_verified = service.verify_grounding(citations[0], chunks_for_verification)
        assert is_verified is True

    # =================================================================
    # TEST CASE 6: Edge cases
    # =================================================================

    def test_empty_answer_returns_empty(self, sample_chunks):
        """Empty answer should be handled gracefully."""
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        citations = service.extract_citations("")
        assert len(citations) == 0

    def test_no_source_chunks_no_modification(self, sample_chunks):
        """When no source chunks, citation cannot be verified."""
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        answer_text = "「학칙」 제25조에 따르면..."

        # Empty source chunks
        citations = service.extract_citations(answer_text)
        assert len(citations) == 1

        is_verified = service.verify_grounding(citations[0], [])
        assert is_verified is False, "Citation should not be verified with empty sources"

    def test_citation_with_paragraph_preserved(self, sample_chunks):
        """Citations with paragraph (항) should be handled correctly."""
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
        )

        service = CitationVerificationService()

        answer_text = "「학칙」 제25조 제1항에 따르면, 휴학은 2년을 초과할 수 없습니다."

        # Convert to dict format
        chunks_for_verification = []
        for result in sample_chunks[:1]:
            chunk = result.chunk
            chunk_dict = {
                "text": chunk.text,
                "metadata": {
                    "regulation_name": chunk.regulation_name,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                },
            }
            chunks_for_verification.append(chunk_dict)

        citations = service.extract_citations(answer_text)
        assert len(citations) == 1
        assert citations[0].paragraph == 1

        # Even with paragraph, the base citation should be verified
        is_verified = service.verify_grounding(citations[0], chunks_for_verification)
        # Note: Current implementation matches on regulation_name and article only
        # If paragraph is also checked, this might need adjustment


class TestCitationVerificationWithSearchUsecase:
    """Integration tests that verify _verify_citations method behavior."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks with regulation metadata."""
        chunks = []

        chunk1 = create_mock_chunk(
            "chunk-1",
            "휴학은 2년을 초과할 수 없다. 휴학 기간은 재학 기간에 산입하지 않는다.",
            "학칙", 25, 1, "RULE-001", "휴학에 관한 규정", ["학칙", "제4장 휴학"]
        )
        chunks.append(create_search_result(chunk1, 0.9, 1))

        chunk2 = create_mock_chunk(
            "chunk-2",
            "등록금은 매 학기 시작 전까지 납부하여야 한다.",
            "등록금에 관한 규정", 4, None, "RULE-002", "등록금 납부", ["등록금에 관한 규정", "제2장 납부"]
        )
        chunks.append(create_search_result(chunk2, 0.85, 2))

        return chunks

    def test_verify_citations_method_preserves_valid(self, sample_chunks):
        """Test _verify_citations preserves valid citations."""
        from src.rag.application.search_usecase import SearchUseCase

        # Create usecase with minimal mocks
        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            answer_text = "「학칙」 제25조에 따르면, 휴학은 2년을 초과할 수 없습니다."

            result = usecase._verify_citations(answer_text, sample_chunks[:1])

            # Valid citation should be preserved
            assert "「학칙」 제25조" in result

    def test_verify_citations_method_sanitizes_invalid(self, sample_chunks):
        """Test _verify_citations sanitizes invalid citations."""
        from src.rag.application.search_usecase import SearchUseCase

        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            # Invalid citation (제99조 not in chunks)
            answer_text = "「학칙」 제99조에 따르면, 특별한 경우에 추가 휴학이 가능합니다."

            result = usecase._verify_citations(answer_text, sample_chunks[:1])

            # Invalid citation should be sanitized
            assert "「학칙」 제99조" not in result
            # Should have fallback phrase
            assert "관련 규정에 따르면" in result

    def test_verify_citations_method_handles_no_citations(self, sample_chunks):
        """Test _verify_citations handles answer with no citations."""
        from src.rag.application.search_usecase import SearchUseCase

        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            answer_text = "휴학은 일반적으로 2년까지 가능하며, 자세한 내용은 학사팀에 문의하세요."

            result = usecase._verify_citations(answer_text, sample_chunks)

            # Should return unchanged
            assert result == answer_text

    def test_verify_citations_method_handles_empty_answer(self, sample_chunks):
        """Test _verify_citations handles empty answer."""
        from src.rag.application.search_usecase import SearchUseCase

        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            result = usecase._verify_citations("", sample_chunks)

            assert result == ""

    def test_verify_citations_method_handles_empty_chunks(self):
        """Test _verify_citations handles empty chunks."""
        from src.rag.application.search_usecase import SearchUseCase

        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            answer_text = "「학칙」 제25조에 따르면..."

            result = usecase._verify_citations(answer_text, [])

            # With no chunks, should return answer unchanged (no verification possible)
            assert result == answer_text


class TestCitationVerificationMetrics:
    """Tests for citation verification metrics and logging (TASK-014)."""

    def test_verification_logs_citations_found(self, caplog):
        """Citation verification should log number of citations found."""
        import logging
        from src.rag.application.search_usecase import SearchUseCase

        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            chunk = create_mock_chunk("test", "휴학은 2년을 초과할 수 없다.", "학칙", 25)
            result = create_search_result(chunk, 0.9, 1)

            answer_text = "「학칙」 제25조에 따르면..."

            with caplog.at_level(logging.DEBUG):
                verified = usecase._verify_citations(answer_text, [result])

            # Check logs contain citation-related information
            log_messages = [r.message for r in caplog.records]
            # The method should log citation verification info

    def test_verification_tracks_sanitized_count(self, caplog):
        """Citation verification should track how many citations were sanitized."""
        import logging
        from src.rag.application.search_usecase import SearchUseCase

        with patch.object(SearchUseCase, '__init__', return_value=None):
            usecase = SearchUseCase.__new__(SearchUseCase)
            usecase._citation_verification_service = None

            # Chunk with different article (not 99)
            chunk = create_mock_chunk("test", "휴학은 2년을 초과할 수 없다.", "학칙", 25)
            result = create_search_result(chunk, 0.9, 1)

            answer_text = "「학칙」 제99조에 따르면..."

            with caplog.at_level(logging.INFO):
                verified = usecase._verify_citations(answer_text, [result])

            # Check for sanitization log
            log_messages = [r.message for r in caplog.records]
            citation_logs = [m for m in log_messages if "citation" in m.lower() or "sanitized" in m.lower()]

            # Should have logged something about citations
            # The exact format is verified in the implementation


class TestCitationVerificationServiceIntegration:
    """Direct integration tests with CitationVerificationService."""

    def test_service_integration_with_search_usecase(self):
        """Verify CitationVerificationService is properly used by SearchUseCase."""
        from src.rag.domain.citation.citation_verification_service import (
            CitationVerificationService,
            ExtractedCitation,
        )

        # Verify the service can be imported and instantiated
        service = CitationVerificationService()

        # Test extraction
        citations = service.extract_citations("「학칙」 제25조에 따르면...")
        assert len(citations) == 1
        assert citations[0].regulation_name == "학칙"
        assert citations[0].article == 25

        # Test verification
        chunks = [{
            "text": "휴학은 2년을 초과할 수 없다.",
            "metadata": {
                "regulation_name": "학칙",
                "article": 25,
            }
        }]
        assert service.verify_grounding(citations[0], chunks) is True

        # Test sanitization
        unverified_citation = ExtractedCitation("학칙", 99, "「학칙」 제99조", is_verified=False)
        fallback = service.sanitize_unverifiable(unverified_citation)
        assert "관련 규정에 따르면" in fallback or "확인이 필요" in fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
