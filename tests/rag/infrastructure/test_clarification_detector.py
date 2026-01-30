"""
Tests for ClarificationDetector.

Tests the clarification detection and generation functionality.
"""

import pytest

from src.rag.domain.entities import Chunk, ChunkLevel, SearchResult
from src.rag.infrastructure.clarification_detector import (
    ClarificationDetector,
    ClarificationRequest,
    QueryAnalysis,
)


@pytest.fixture
def detector():
    """Create a ClarificationDetector instance."""
    return ClarificationDetector()


@pytest.fixture
def sample_results():
    """Create sample search results."""
    results = []

    # Create sample chunks
    chunk1 = Chunk(
        id="1",
        text="휴학에 관한 규정입니다.",
        title="휴학",
        rule_code="RULE001",
        level=ChunkLevel.TEXT,
        embedding_text="휴학에 관한 규정입니다.",
        full_text="휴학에 관한 규정입니다.",
        parent_path=["학칙", "제2장", "휴학"],
        token_count=10,
        keywords=[],
        is_searchable=True,
    )

    chunk2 = Chunk(
        id="2",
        text="장학금 신청 방법입니다.",
        title="장학금",
        rule_code="RULE002",
        level=ChunkLevel.TEXT,
        embedding_text="장학금 신청 방법입니다.",
        full_text="장학금 신청 방법입니다.",
        parent_path=["장학금 규정", "제3장", "장학금"],
        token_count=10,
        keywords=[],
        is_searchable=True,
    )

    results.append(
        SearchResult(
            chunk=chunk1,
            score=0.85,
        )
    )

    results.append(
        SearchResult(
            chunk=chunk2,
            score=0.75,
        )
    )

    return results


class TestClarificationDetector:
    """Test ClarificationDetector functionality."""

    def test_analyze_single_word_query(self, detector):
        """Test analysis of single-word queries."""
        analysis = detector.analyze_query("휴학")

        assert analysis.is_single_word is True
        assert analysis.is_short is True
        assert analysis.word_count == 1
        assert "휴학" in analysis.key_terms

    def test_analyze_short_query(self, detector):
        """Test analysis of short queries."""
        analysis = detector.analyze_query("휴학 방법")

        assert analysis.is_single_word is False
        assert analysis.is_short is True
        assert analysis.word_count == 2

    def test_analyze_ambiguous_query(self, detector):
        """Test analysis of ambiguous queries."""
        analysis = detector.analyze_query("신청")

        assert analysis.is_ambiguous is True
        assert "신청" in analysis.key_terms

    def test_is_clarification_needed_single_word(self, detector, sample_results):
        """Test clarification needed for single-word query."""
        result = detector.is_clarification_needed("휴학", sample_results)

        assert result is True

    def test_is_clarification_needed_ambiguous(self, detector, sample_results):
        """Test clarification needed for ambiguous query."""
        result = detector.is_clarification_needed("신청", sample_results)

        assert result is True

    def test_is_clarification_needed_no_results(self, detector):
        """Test clarification needed when no results."""
        result = detector.is_clarification_needed("없는규정", [])

        assert result is True

    def test_is_clarification_needed_diverse_results(self, detector):
        """Test clarification needed when results are too diverse."""
        # Create diverse results from different regulations
        diverse_results = []
        for i in range(6):
            chunk = Chunk(
                id=str(i),
                text=f"내용 {i}",
                title=f"항목 {i}",
                rule_code=f"RULE{i:03d}",
                level=ChunkLevel.TEXT,
                embedding_text=f"내용 {i}",
                full_text=f"내용 {i}",
                parent_path=[f"규정{i}", "제1장"],
                token_count=5,
                keywords=[],
                is_searchable=True,
            )
            diverse_results.append(SearchResult(chunk=chunk, score=0.8))

        result = detector.is_clarification_needed("검색", diverse_results)

        assert result is True

    def test_is_clarification_not_needed_specific_query(self, detector, sample_results):
        """Test clarification not needed for specific query."""
        result = detector.is_clarification_needed(
            "휴학 신청 방법과 절차에 대해 알려주세요", sample_results
        )

        assert result is False

    def test_generate_clarification_single_word(self, detector, sample_results):
        """Test clarification generation for single-word query."""
        clarification = detector.generate_clarification("휴학", sample_results)

        assert clarification.needs_clarification is True
        assert len(clarification.clarification_questions) > 0
        assert len(clarification.suggested_options) > 0
        assert "휴학" in clarification.reason

    def test_generate_clarification_ambiguous(self, detector, sample_results):
        """Test clarification generation for ambiguous query."""
        clarification = detector.generate_clarification("신청", sample_results)

        assert clarification.needs_clarification is True
        assert len(clarification.suggested_options) > 0

    def test_extract_key_terms(self, detector):
        """Test key term extraction with particle removal."""
        # Use particles that can be removed by current implementation
        terms = detector._extract_key_terms("휴학은 신청을")

        assert "휴학" in terms
        assert "신청" in terms

    def test_clarification_request_dataclass(self):
        """Test ClarificationRequest dataclass."""
        request = ClarificationRequest(
            needs_clarification=True,
            reason="Test reason",
            clarification_questions=["Question 1?"],
            suggested_options=["Option 1", "Option 2"],
        )

        assert request.needs_clarification is True
        assert request.reason == "Test reason"
        assert len(request.clarification_questions) == 1
        assert len(request.suggested_options) == 2

    def test_query_analysis_dataclass(self):
        """Test QueryAnalysis dataclass."""
        analysis = QueryAnalysis(
            is_short=True,
            is_single_word=False,
            is_ambiguous=False,
            word_count=2,
            char_count=4,
            key_terms=["term1", "term2"],
        )

        assert analysis.is_short is True
        assert analysis.is_single_word is False
        assert analysis.word_count == 2


@pytest.mark.integration
class TestClarificationDetectorIntegration:
    """Integration tests for ClarificationDetector."""

    def test_full_clarification_workflow(self, detector, sample_results):
        """Test complete clarification workflow."""
        query = "신청"

        # Step 1: Check if clarification needed
        needs_clarification = detector.is_clarification_needed(query, sample_results)
        assert needs_clarification is True

        # Step 2: Generate clarification
        clarification = detector.generate_clarification(query, sample_results)

        # Step 3: Verify response structure
        assert clarification.needs_clarification is True
        assert isinstance(clarification.reason, str)
        assert isinstance(clarification.clarification_questions, list)
        assert isinstance(clarification.suggested_options, list)
        assert len(clarification.clarification_questions) > 0
        assert len(clarification.suggested_options) > 0
