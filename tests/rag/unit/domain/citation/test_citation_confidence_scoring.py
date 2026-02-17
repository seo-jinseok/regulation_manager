"""
Unit tests for Citation Confidence Scoring.

Tests verify confidence scoring functionality in CitationEnhancer.
"""

import pytest

from src.rag.domain.citation.citation_enhancer import (
    CitationEnhancer,
    EnhancedCitation,
)
from src.rag.domain.entities import ChunkLevel


class FakeChunk:
    """Fake chunk for testing."""

    def __init__(
        self,
        id="test-id",
        rule_code="test-code",
        title="",
        text="",
        parent_path=None,
        article_number=None,
    ):
        self.id = id
        self.rule_code = rule_code
        self.level = ChunkLevel.ARTICLE
        self.title = title
        self.text = text
        self.parent_path = parent_path or []
        self.article_number = article_number


class TestCitationConfidenceScoring:
    """Tests for citation confidence scoring functionality."""

    def test_high_confidence_exact_match(self):
        """High confidence when article appears in title and text."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-1",
            rule_code="직원복무규정_제26조",
            title="제26조 (직원의 구분)",
            text="제26조(직원의 구분) 직원은 일반직, 기술직으로 구분한다.",
            parent_path=["직원복무규정"],
            article_number="제26조",
        )

        confidence = enhancer.calculate_confidence(chunk, "제26조")

        # High confidence: title(0.2) + text(0.2) + article_number(0.3) + rule_code(0.15) + parent_path(0.15) = 1.0
        assert confidence >= 0.9

    def test_medium_confidence_partial_match(self):
        """Medium confidence when article only in metadata."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-2",
            rule_code="직원복무규정_제26조",
            title="제26조 (직원의 구분)",
            text="직원은 일반직, 기술직으로 구분한다.",  # No article number
            parent_path=["직원복무규정"],
            article_number="제26조",
        )

        confidence = enhancer.calculate_confidence(chunk, "제26조")

        # Medium: title(0.2) + article_number(0.3) + rule_code(0.15) + parent_path(0.15) = 0.8
        assert 0.7 <= confidence <= 0.9

    def test_low_confidence_metadata_only(self):
        """Low confidence when article only in rule_code."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-3",
            rule_code="직원복무규정_제26조",
            title="직원의 구분",  # No article number
            text="직원은 일반직, 기술직으로 구분한다.",
            parent_path=["직원복무규정"],
            article_number=None,
        )

        confidence = enhancer.calculate_confidence(chunk, "제26조")

        # Low: rule_code(0.15) + parent_path(0.15) = 0.3
        assert 0.2 <= confidence <= 0.4

    def test_confidence_with_context_relevance(self):
        """Context relevance boosts confidence."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-4",
            rule_code="직원복무규정_제26조",
            title="제26조 (직원의 구분)",
            text="제26조(직원의 구분) 직원은 일반직, 기술직으로 구분한다.",
            parent_path=["직원복무규정"],
            article_number="제26조",
        )
        query = "직원의 구분은 어떻게 되나요?"

        base_confidence = enhancer.calculate_confidence(chunk, "제26조")
        context_confidence = enhancer.calculate_confidence_with_context(
            chunk, "제26조", query
        )

        # Context-aware should be at least as high
        assert context_confidence >= base_confidence

    def test_confidence_capped_at_one(self):
        """Confidence score is capped at 1.0."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-5",
            rule_code="직원복무규정_제26조",
            title="제26조 (직원의 구분)",
            text="제26조(직원의 구분) 직원은 일반직, 기술직으로 구분한다. 직원 구분 관련.",
            parent_path=["직원복무규정"],
            article_number="제26조",
        )

        confidence = enhancer.calculate_confidence(chunk, "제26조")

        assert confidence <= 1.0


class TestCitationValidation:
    """Tests for citation validation against source."""

    def test_validate_citation_exists_in_text(self):
        """Validate returns True when citation exists in text."""
        enhancer = CitationEnhancer()
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="제26조",
            chunk_id="test-1",
            confidence=0.9,
            title="제26조 (직원의 구분)",
            text="제26조(직원의 구분) 직원은 일반직, 기술직으로 구분한다.",
        )

        is_valid = enhancer.validate_citation_in_source(citation)

        assert is_valid is True

    def test_validate_citation_not_in_text(self):
        """Validate returns False when citation not in text."""
        enhancer = CitationEnhancer()
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="제999조",  # Non-existent
            chunk_id="test-2",
            confidence=0.9,
            title="제26조 (직원의 구분)",
            text="제26조(직원의 구분) 직원은 일반직, 기술직으로 구분한다.",
        )

        is_valid = enhancer.validate_citation_in_source(citation)

        assert is_valid is False

    def test_validate_with_paragraph_in_text(self):
        """Validate paragraph citation matches circled number."""
        enhancer = CitationEnhancer()
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="제26조제1항",
            chunk_id="test-3",
            confidence=0.9,
            title="제26조 (직원의 구분)",
            text="제26조(직원의 구분) ① 직원은 일반직, 기술직으로 구분한다.",
        )

        is_valid = enhancer.validate_citation_in_source(citation)

        assert is_valid is True

    def test_validate_empty_text_returns_false(self):
        """Validate returns False for empty text."""
        enhancer = CitationEnhancer()
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="제26조",
            chunk_id="test-4",
            confidence=0.9,
            title="제26조 (직원의 구분)",
            text="",
        )

        is_valid = enhancer.validate_citation_in_source(citation)

        assert is_valid is False


class TestConfidenceFiltering:
    """Tests for filtering low-confidence citations."""

    def test_filter_low_confidence(self):
        """Filter removes citations below threshold."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-1",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제27조",
                chunk_id="test-2",
                confidence=0.3,  # Below threshold
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="test-3",
                confidence=0.7,
            ),
        ]

        filtered = enhancer.filter_low_confidence(citations)

        assert len(filtered) == 2
        assert all(c.confidence >= 0.5 for c in filtered)

    def test_filter_empty_list(self):
        """Filter handles empty list."""
        enhancer = CitationEnhancer()

        filtered = enhancer.filter_low_confidence([])

        assert filtered == []

    def test_filter_all_high_confidence(self):
        """Filter keeps all when all are high confidence."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-1",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="test-2",
                confidence=0.8,
            ),
        ]

        filtered = enhancer.filter_low_confidence(citations)

        assert len(filtered) == 2

    def test_filter_all_low_confidence(self):
        """Filter removes all when all are low confidence."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-1",
                confidence=0.3,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="test-2",
                confidence=0.2,
            ),
        ]

        filtered = enhancer.filter_low_confidence(citations)

        assert len(filtered) == 0
