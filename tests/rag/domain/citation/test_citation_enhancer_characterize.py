"""
Characterization tests for CitationEnhancer behavior preservation.

These tests capture the CURRENT behavior of the citation enhancer
to ensure that improvements don't break existing functionality.
"""

import pytest
from src.rag.domain.citation.citation_enhancer import CitationEnhancer, EnhancedCitation
from src.rag.domain.entities import Chunk


@pytest.fixture
def citation_enhancer():
    """Create a CitationEnhancer instance for testing."""
    return CitationEnhancer()


@pytest.fixture
def sample_chunk_with_article():
    """Create a sample chunk with article_number."""
    from src.rag.domain.entities import ChunkLevel
    chunk = Chunk(
        id="test-chunk-1",
        rule_code="교원인사규정_제26조",
        level=ChunkLevel.ARTICLE,
        title="제26조(휴학)",
        text="휴학은 학기 시작 전에 신청해야 합니다.",
        embedding_text="휴학은 학기 시작 전에 신청해야 합니다.",
        full_text="제26조(휴학) 휴학은 학기 시작 전에 신청해야 합니다.",
        parent_path=["교원인사규정"],
        token_count=50,
        keywords=[],
        is_searchable=True,
        article_number="제26조",
    )
    return chunk


@pytest.fixture
def sample_chunk_without_article():
    """Create a sample chunk without article_number."""
    from src.rag.domain.entities import ChunkLevel
    chunk = Chunk(
        id="test-chunk-2",
        rule_code="학칙",
        level=ChunkLevel.TEXT,
        title="휴학 절차",
        text="휴학 신청은 학기 시작 전에 해야 합니다.",
        embedding_text="휴학 신청은 학기 시작 전에 해야 합니다.",
        full_text="휴학 절차 휴학 신청은 학기 시작 전에 해야 합니다.",
        parent_path=["학칙"],
        token_count=40,
        keywords=[],
        is_searchable=True,
    )
    return chunk


@pytest.fixture
def sample_chunk_with_table():
    """Create a sample chunk for a table reference."""
    from src.rag.domain.entities import ChunkLevel
    chunk = Chunk(
        id="test-chunk-3",
        rule_code="직원복무규정_별표1",
        level=ChunkLevel.ADDENDUM,
        title="별표1 (직원급별 봉급표)",
        text="직원급별 봉급표는 다음과 같습니다.",
        embedding_text="직원급별 봉급표는 다음과 같습니다.",
        full_text="별표1 (직원급별 봉급표) 직원급별 봉급표는 다음과 같습니다.",
        parent_path=["직원복무규정"],
        token_count=30,
        keywords=[],
        is_searchable=True,
        article_number="별표1",
    )
    return chunk


class TestCitationEnhancerCharacterization:
    """Characterization tests for CitationEnhancer."""

    def test_enhance_citation_with_article_number(
        self, citation_enhancer, sample_chunk_with_article
    ):
        """CHARACTERIZE: Current behavior for chunks with article_number."""
        result = citation_enhancer.enhance_citation(sample_chunk_with_article, confidence=0.9)

        # Document current behavior
        assert result is not None
        assert result.regulation == "교원인사규정"
        assert result.article_number == "제26조"
        assert result.chunk_id == "test-chunk-1"
        assert result.confidence == 0.9
        assert result.title == "제26조(휴학)"
        assert "휴학은" in result.text

    def test_enhance_citation_without_article_number(
        self, citation_enhancer, sample_chunk_without_article
    ):
        """CHARACTERIZE: Current behavior for chunks without article_number."""
        result = citation_enhancer.enhance_citation(sample_chunk_without_article)

        # Document current behavior
        assert result is None  # Cannot create citation without article number

    def test_enhance_citation_table_format(
        self, citation_enhancer, sample_chunk_with_table
    ):
        """CHARACTERIZE: Current behavior for table citations (별표)."""
        result = citation_enhancer.enhance_citation(sample_chunk_with_table)

        # Document current behavior
        assert result is not None
        assert result.article_number == "별표1"
        # Test formatting
        formatted = result.format()
        assert "별표1" in formatted
        assert "직원급별 봉급표" in formatted

    def test_format_citations_single(self, citation_enhancer, sample_chunk_with_article):
        """CHARACTERIZE: Current formatting for single citation."""
        enhanced = citation_enhancer.enhance_citation(sample_chunk_with_article)
        formatted = citation_enhancer.format_citations([enhanced])

        # Document current format
        assert formatted == "「교원인사규정」 제26조"

    def test_format_citations_multiple(self, citation_enhancer):
        """CHARACTERIZE: Current formatting for multiple citations."""
        from src.rag.domain.entities import ChunkLevel
        chunks = [
            Chunk(
                id="chunk-1",
                rule_code="r1",
                level=ChunkLevel.ARTICLE,
                title="제26조(휴학)",
                text="Text 1",
                embedding_text="Text 1",
                full_text="제26조(휴학) Text 1",
                parent_path=["교원인사규정"],
                token_count=10,
                keywords=[],
                is_searchable=True,
                article_number="제26조",
            ),
            Chunk(
                id="chunk-2",
                rule_code="r2",
                level=ChunkLevel.ARTICLE,
                title="제15조(등록)",
                text="Text 2",
                embedding_text="Text 2",
                full_text="제15조(등록) Text 2",
                parent_path=["학칙"],
                token_count=10,
                keywords=[],
                is_searchable=True,
                article_number="제15조",
            ),
        ]

        enhanced = citation_enhancer.enhance_citations(chunks)
        formatted = citation_enhancer.format_citations(enhanced)

        # Document current format
        assert "「교원인사규정」 제26조" in formatted
        assert "「학칙」 제15조" in formatted
        assert ", " in formatted  # Citations separated by comma

    def test_deduplicate_citations(self, citation_enhancer):
        """CHARACTERIZE: Current deduplication behavior."""
        citations = [
            EnhancedCitation(
                regulation="교원인사규정",
                article_number="제26조",
                chunk_id="chunk-1",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="교원인사규정",
                article_number="제26조",
                chunk_id="chunk-2",  # Different chunk, same citation
                confidence=0.8,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="chunk-3",
                confidence=0.9,
            ),
        ]

        result = citation_enhancer.deduplicate_citations(citations)

        # Document current behavior: should keep first occurrence
        assert len(result) == 2
        assert result[0].chunk_id == "chunk-1"  # First occurrence kept
        assert any(c.article_number == "제15조" for c in result)

    def test_sort_by_article_number(self, citation_enhancer):
        """CHARACTERIZE: Current sorting behavior."""
        citations = [
            EnhancedCitation(
                regulation="학칙",
                article_number="제100조",
                chunk_id="chunk-1",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="교원인사규정",
                article_number="제15조",
                chunk_id="chunk-2",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="직원복무규정",
                article_number="제26조",
                chunk_id="chunk-3",
                confidence=0.9,
            ),
        ]

        result = citation_enhancer.sort_by_article_number(citations)

        # Document current behavior: should sort by numeric article number
        assert result[0].article_number == "제15조"
        assert result[1].article_number == "제26조"
        assert result[2].article_number == "제100조"

    def test_enhance_citations_with_confidences(self, citation_enhancer):
        """CHARACTERIZE: Current behavior with explicit confidence scores."""
        from src.rag.domain.entities import ChunkLevel
        chunks = [
            Chunk(
                id="chunk-1",
                rule_code="r1",
                level=ChunkLevel.ARTICLE,
                title="제26조",
                text="Text 1",
                embedding_text="Text 1",
                full_text="제26조 Text 1",
                parent_path=["규정1"],
                token_count=10,
                keywords=[],
                is_searchable=True,
                article_number="제26조",
            ),
            Chunk(
                id="chunk-2",
                rule_code="r2",
                level=ChunkLevel.ARTICLE,
                title="제15조",
                text="Text 2",
                embedding_text="Text 2",
                full_text="제15조 Text 2",
                parent_path=["규정2"],
                token_count=10,
                keywords=[],
                is_searchable=True,
                article_number="제15조",
            ),
        ]

        confidences = [0.95, 0.85]
        enhanced = citation_enhancer.enhance_citations(chunks, confidences)

        # Document current behavior
        assert len(enhanced) == 2
        assert enhanced[0].confidence == 0.95
        assert enhanced[1].confidence == 0.85

    def test_enhance_citations_mismatched_lengths(self, citation_enhancer):
        """CHARACTERIZE: Current behavior when chunks and confidences length mismatch."""
        from src.rag.domain.entities import ChunkLevel
        chunks = [
            Chunk(
                id="chunk-1",
                rule_code="r1",
                level=ChunkLevel.ARTICLE,
                title="제26조",
                text="Text",
                embedding_text="Text",
                full_text="제26조 Text",
                parent_path=["규정"],
                token_count=10,
                keywords=[],
                is_searchable=True,
                article_number="제26조",
            )
        ]

        confidences = [0.9, 0.8]  # More confidences than chunks

        # Should not crash, should use default confidence
        enhanced = citation_enhancer.enhance_citations(chunks, confidences)
        assert len(enhanced) == 1

    def test_group_by_regulation(self, citation_enhancer):
        """CHARACTERIZE: Current grouping behavior."""
        citations = [
            EnhancedCitation(
                regulation="교원인사규정",
                article_number="제26조",
                chunk_id="c1",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="학칙",
                article_number="제15조",
                chunk_id="c2",
                confidence=0.9,
            ),
            EnhancedCitation(
                regulation="교원인사규정",
                article_number="제30조",
                chunk_id="c3",
                confidence=0.9,
            ),
        ]

        grouped = citation_enhancer.group_by_regulation(citations)

        # Document current behavior
        assert "교원인사규정" in grouped
        assert "학칙" in grouped
        assert len(grouped["교원인사규정"]) == 2
        assert len(grouped["학칙"]) == 1
