"""
Unit tests for CitationEnhancer service.

Tests verify citation enhancement, formatting, and validation functionality.
"""

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


class TestEnhancedCitation:
    """Tests for EnhancedCitation dataclass."""

    def test_format_basic_citation(self):
        """Format basic article citation."""
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="제26조",
            chunk_id="test-1",
            confidence=0.9,
            title="제26조 (직원의 구분)",
        )

        formatted = citation.format()

        assert formatted == "「직원복무규정」 제26조"

    def test_format_sub_article_citation(self):
        """Format sub-article citation."""
        citation = EnhancedCitation(
            regulation="학칙",
            article_number="제10조의2",
            chunk_id="test-2",
            confidence=0.85,
        )

        formatted = citation.format()

        assert formatted == "「학칙」 제10조의2"

    def test_format_table_citation(self):
        """Format table reference without quotes."""
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="별표1",
            chunk_id="test-3",
            confidence=1.0,
            title="별표1 직원급별 봉급표",
        )

        formatted = citation.format()

        # Tables don't use regulation quotes
        assert formatted == "별표1 (별표1 직원급별 봉급표)"

    def test_format_form_citation(self):
        """Format form reference."""
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="서식1",
            chunk_id="test-4",
            confidence=1.0,
            title="서식1 휴직원부",
        )

        formatted = citation.format()

        assert formatted == "서식1 (서식1 휴직원부)"

    def test_format_table_without_title(self):
        """Format table reference without title."""
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="별표1",
            chunk_id="test-5",
            confidence=1.0,
        )

        formatted = citation.format()

        assert formatted == "별표1"

    def test_to_dict(self):
        """Convert citation to dictionary."""
        citation = EnhancedCitation(
            regulation="직원복무규정",
            article_number="제26조",
            chunk_id="test-6",
            confidence=0.9,
            title="제26조 (직원의 구분)",
            text="직원은 일반직, 기술직, 별정직으로 구분한다.",
        )

        result = citation.to_dict()

        assert result["regulation"] == "직원복무규정"
        assert result["article_number"] == "제26조"
        assert result["chunk_id"] == "test-6"
        assert result["confidence"] == 0.9
        assert "text" in result
        # Text should be truncated
        assert len(result["text"]) <= 200


class TestCitationEnhancer:
    """Tests for CitationEnhancer service."""

    def test_enhance_basic_citation(self):
        """Enhance chunk with basic article number."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-1",
            title="제26조 (직원의 구분)",
            text="직원은 일반직, 기술직으로 구분한다.",
            parent_path=["직원복무규정"],
            article_number="제26조",
        )

        result = enhancer.enhance_citation(chunk, confidence=0.95)

        assert result is not None
        assert result.regulation == "직원복무규정"
        assert result.article_number == "제26조"
        assert result.confidence == 0.95
        assert result.chunk_id == "chunk-1"

    def test_enhance_without_article_number_field(self):
        """Extract article number from title if field not set."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-2",
            title="제26조 (직원의 구분)",
            text="직원은 일반직으로 구분한다.",
            parent_path=["직원복무규정"],
            article_number=None,  # Field not set
        )

        result = enhancer.enhance_citation(chunk)

        # Should extract from title
        assert result is not None
        assert result.article_number == "제26조"

    def test_enhance_returns_none_without_regulation(self):
        """Return None if chunk has no regulation (parent_path)."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-3",
            title="제26조",
            text="Some text",
            parent_path=[],  # No regulation
            article_number="제26조",
        )

        result = enhancer.enhance_citation(chunk)

        assert result is None

    def test_enhance_returns_none_without_article_number(self):
        """Return None if chunk has no article number."""
        enhancer = CitationEnhancer()
        chunk = FakeChunk(
            id="chunk-4",
            title="일반 규정",
            text="Some text",
            parent_path=["직원복무규정"],
            article_number=None,
        )

        result = enhancer.enhance_citation(chunk)

        assert result is None

    def test_enhance_multiple_citations(self):
        """Enhance multiple chunks at once."""
        enhancer = CitationEnhancer()
        chunks = [
            FakeChunk(
                id="chunk-5",
                title="제26조",
                text="Text 1",
                parent_path=["직원복무규정"],
                article_number="제26조",
            ),
            FakeChunk(
                id="chunk-6",
                title="제15조",
                text="Text 2",
                parent_path=["학칙"],
                article_number="제15조",
            ),
        ]

        results = enhancer.enhance_citations(chunks)

        assert len(results) == 2
        assert results[0].regulation == "직원복무규정"
        assert results[1].regulation == "학칙"

    def test_enhance_with_confidence_scores(self):
        """Enhance with custom confidence scores."""
        enhancer = CitationEnhancer()
        chunks = [
            FakeChunk(
                id="chunk-7",
                title="제26조",
                text="Text",
                parent_path=["직원복무규정"],
                article_number="제26조",
            ),
            FakeChunk(
                id="chunk-8",
                title="제15조",
                text="Text",
                parent_path=["학칙"],
                article_number="제15조",
            ),
        ]
        confidences = [0.9, 0.7]

        results = enhancer.enhance_citations(chunks, confidences=confidences)

        assert len(results) == 2
        assert results[0].confidence == 0.9
        assert results[1].confidence == 0.7

    def test_format_citations_single(self):
        """Format single citation."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test",
                confidence=1.0,
            )
        ]

        formatted = enhancer.format_citations(citations)

        assert formatted == "「직원복무규정」 제26조"

    def test_format_citations_multiple(self):
        """Format multiple citations."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-1",
                confidence=1.0,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="test-2",
                confidence=1.0,
            ),
        ]

        formatted = enhancer.format_citations(citations)

        assert formatted == "「직원복무규정」 제26조, 「학칙」 제15조"

    def test_format_citations_empty(self):
        """Format empty citation list."""
        enhancer = CitationEnhancer()

        formatted = enhancer.format_citations([])

        assert formatted == ""

    def test_group_by_regulation(self):
        """Group citations by regulation name."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-1",
                confidence=1.0,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="test-2",
                confidence=1.0,
            ),
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제27조",
                chunk_id="test-3",
                confidence=1.0,
            ),
        ]

        grouped = enhancer.group_by_regulation(citations)

        assert "직원복무규정" in grouped
        assert "학칙" in grouped
        assert len(grouped["직원복무규정"]) == 2
        assert len(grouped["학칙"]) == 1

    def test_sort_by_article_number(self):
        """Sort citations by article number."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="규정",
                article_number="제100조",
                chunk_id="test-1",
                confidence=1.0,
            ),
            EnhancedCitation(
                regulation="규정",
                article_number="제26조",
                chunk_id="test-2",
                confidence=1.0,
            ),
            EnhancedCitation(
                regulation="규정",
                article_number="제5조",
                chunk_id="test-3",
                confidence=1.0,
            ),
        ]

        sorted_citations = enhancer.sort_by_article_number(citations)

        assert sorted_citations[0].article_number == "제5조"
        assert sorted_citations[1].article_number == "제26조"
        assert sorted_citations[2].article_number == "제100조"

    def test_deduplicate_citations(self):
        """Remove duplicate citations."""
        enhancer = CitationEnhancer()
        citations = [
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-1",
                confidence=1.0,
            ),
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="test-2",  # Different chunk ID
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="test-3",
                confidence=1.0,
            ),
        ]

        unique = enhancer.deduplicate_citations(citations)

        # Should keep first occurrence of duplicate
        assert len(unique) == 2
        assert unique[0].chunk_id == "test-1"
        assert unique[1].article_number == "제15조"
