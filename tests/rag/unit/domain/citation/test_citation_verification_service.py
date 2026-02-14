"""
Unit tests for CitationExtractor and citation pattern detection.

These tests verify citation pattern extraction, standardization,
and format verification for Korean regulation citations.
"""

import pytest

from src.rag.domain.citation.citation_verification_service import (
    CitationExtractor,
    CitationVerificationService,
    ExtractedCitation,
)
from src.rag.domain.citation.citation_patterns import (
    CitationPatterns,
    CitationFormat,
)


class TestCitationPatterns:
    """Tests for citation pattern definitions."""

    def test_standard_citation_pattern(self):
        """Test standard citation pattern matching."""
        patterns = CitationPatterns()

        # Standard format: 「규정명」 제X조
        match = patterns.match_citation("「학칙」 제25조에 따르면")
        assert match is not None
        assert match.group(1) == "학칙"
        assert match.group(2) == "25"

    def test_citation_with_paragraph_pattern(self):
        """Test citation with paragraph (항) pattern."""
        patterns = CitationPatterns()

        # Format: 「규정명」 제X조 제X항
        match = patterns.match_citation("「등록금에 관한 규정」 제4조 제2항에 따르면")
        assert match is not None
        assert match.group(1) == "등록금에 관한 규정"
        assert match.group(2) == "4"
        assert match.group(3) == "2"

    def test_no_match_for_incomplete_citation(self):
        """Test that incomplete citations are not matched."""
        patterns = CitationPatterns()

        # Incomplete: only regulation name
        match = patterns.match_citation("「학칙」에 따르면")
        assert match is None

        # Incomplete: only article number
        match = patterns.match_citation("제25조에 따르면")
        assert match is None

    def test_validate_citation_format(self):
        """Test citation format validation."""
        patterns = CitationPatterns()

        # Valid formats
        assert patterns.is_valid_format("「학칙」 제25조") is True
        assert patterns.is_valid_format("「등록금에 관한 규정」 제4조 제2항") is True
        assert patterns.is_valid_format("「교원인사규정」 제10조의2") is True

        # Invalid formats
        assert patterns.is_valid_format("「학칙」") is False
        assert patterns.is_valid_format("제25조") is False
        assert patterns.is_valid_format("학칙 제25조") is False  # Missing guillemets

    def test_standardize_citation(self):
        """Test citation standardization."""
        patterns = CitationPatterns()

        # Already standard
        result = patterns.standardize("「학칙」 제25조")
        assert result == "「학칙」 제25조"

        # With extra whitespace
        result = patterns.standardize("「학칙」  제25조")
        assert result == "「학칙」 제25조"


class TestExtractedCitation:
    """Tests for ExtractedCitation dataclass."""

    def test_create_basic_citation(self):
        """Test creating basic citation."""
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조에 따르면",
        )

        assert citation.regulation_name == "학칙"
        assert citation.article == 25
        assert citation.paragraph is None
        assert citation.is_verified is False
        assert citation.content == ""

    def test_create_citation_with_paragraph(self):
        """Test creating citation with paragraph."""
        citation = ExtractedCitation(
            regulation_name="등록금에 관한 규정",
            article=4,
            paragraph=2,
            original_text="「등록금에 관한 규정」 제4조 제2항에 따르면",
        )

        assert citation.regulation_name == "등록금에 관한 규정"
        assert citation.article == 4
        assert citation.paragraph == 2

    def test_citation_to_standard_format(self):
        """Test formatting citation to standard format."""
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조에 따르면",
        )

        result = citation.to_standard_format()
        assert result == "「학칙」 제25조"

    def test_citation_with_paragraph_to_standard_format(self):
        """Test formatting citation with paragraph to standard format."""
        citation = ExtractedCitation(
            regulation_name="등록금에 관한 규정",
            article=4,
            paragraph=2,
            original_text="「등록금에 관한 규정」 제4조 제2항에 따르면",
        )

        result = citation.to_standard_format()
        assert result == "「등록금에 관한 규정」 제4조 제2항"

    def test_citation_verification_status(self):
        """Test citation verification status."""
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
            is_verified=True,
        )

        assert citation.is_verified is True

    def test_citation_with_content(self):
        """Test citation with extracted content."""
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조에 따르면, '휴학은 2년을 초과할 수 없다'",
            content="휴학은 2년을 초과할 수 없다",
        )

        assert citation.content == "휴학은 2년을 초과할 수 없다"


class TestCitationExtractor:
    """Tests for CitationExtractor functionality."""

    def test_extract_single_citation(self):
        """Test extracting a single citation from text."""
        extractor = CitationExtractor()
        text = "「학칙」 제25조에 따르면 휴학은 2년을 초과할 수 없습니다."

        citations = extractor.extract(text)

        assert len(citations) == 1
        assert citations[0].regulation_name == "학칙"
        assert citations[0].article == 25
        assert citations[0].original_text == "「학칙」 제25조"

    def test_extract_citation_with_paragraph(self):
        """Test extracting citation with paragraph."""
        extractor = CitationExtractor()
        text = "「등록금에 관한 규정」 제4조 제2항에 따르면 등록금은..."

        citations = extractor.extract(text)

        assert len(citations) == 1
        assert citations[0].regulation_name == "등록금에 관한 규정"
        assert citations[0].article == 4
        assert citations[0].paragraph == 2

    def test_extract_multiple_citations(self):
        """Test extracting multiple citations from text."""
        extractor = CitationExtractor()
        text = "「학칙」 제25조와 「등록금에 관한 규정」 제4조에 따르면..."

        citations = extractor.extract(text)

        assert len(citations) == 2
        assert citations[0].regulation_name == "학칙"
        assert citations[0].article == 25
        assert citations[1].regulation_name == "등록금에 관한 규정"
        assert citations[1].article == 4

    def test_extract_no_citations(self):
        """Test extracting from text without citations."""
        extractor = CitationExtractor()
        text = "일반적인 학사 규정에 따르면 휴학이 가능합니다."

        citations = extractor.extract(text)

        assert len(citations) == 0

    def test_extract_handles_whitespace(self):
        """Test extraction handles various whitespace patterns."""
        extractor = CitationExtractor()
        text = "「학칙」  제25조에 따르면"  # Double space

        citations = extractor.extract(text)

        assert len(citations) == 1
        assert citations[0].to_standard_format() == "「학칙」 제25조"

    def test_standardize_format_basic(self):
        """Test standardizing basic citation format."""
        extractor = CitationExtractor()

        # Already standard
        result = extractor.standardize_format("「학칙」 제25조")
        assert result == "「학칙」 제25조"

    def test_standardize_format_with_whitespace(self):
        """Test standardizing citation with extra whitespace."""
        extractor = CitationExtractor()

        result = extractor.standardize_format("「학칙」  제25조")
        assert result == "「학칙」 제25조"

    def test_standardize_format_incomplete_returns_none(self):
        """Test standardizing incomplete citation returns None."""
        extractor = CitationExtractor()

        result = extractor.standardize_format("「학칙」")
        assert result is None

        result = extractor.standardize_format("제25조")
        assert result is None

    def test_extract_with_sub_article(self):
        """Test extracting citation with sub-article (제X조의Y)."""
        extractor = CitationExtractor()
        text = "「교원인사규정」 제10조의2에 따르면..."

        citations = extractor.extract(text)

        assert len(citations) == 1
        assert citations[0].regulation_name == "교원인사규정"
        assert citations[0].article == 10
        assert citations[0].sub_article == 2

    def test_extract_preserves_original_text(self):
        """Test that extraction preserves original citation text."""
        extractor = CitationExtractor()
        text = "「학칙」 제25조에 따르면..."

        citations = extractor.extract(text)

        assert len(citations) == 1
        assert citations[0].original_text == "「학칙」 제25조"

    def test_extract_complex_regulation_name(self):
        """Test extracting citation with complex regulation name."""
        extractor = CitationExtractor()
        text = "「졸업논문또는졸업실적심사규정」 제8조에 따르면..."

        citations = extractor.extract(text)

        assert len(citations) == 1
        assert citations[0].regulation_name == "졸업논문또는졸업실적심사규정"
        assert citations[0].article == 8

    def test_extract_edge_case_numbers(self):
        """Test extracting citations with various article numbers."""
        extractor = CitationExtractor()

        # Large article number
        citations = extractor.extract("「학칙」 제100조에 따르면")
        assert citations[0].article == 100

        # Article number 1
        citations = extractor.extract("「학칙」 제1조에 따르면")
        assert citations[0].article == 1

    def test_citation_to_dict(self):
        """Test citation serialization to dictionary."""
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            paragraph=2,
            original_text="「학칙」 제25조 제2항",
            content="휴학은 2년을 초과할 수 없다",
            is_verified=True,
        )

        result = citation.to_dict()

        assert result["regulation_name"] == "학칙"
        assert result["article"] == 25
        assert result["paragraph"] == 2
        assert result["original_text"] == "「학칙」 제25조 제2항"
        assert result["standard_format"] == "「학칙」 제25조 제2항"
        assert result["content"] == "휴학은 2년을 초과할 수 없다"
        assert result["is_verified"] is True


class TestCitationFormat:
    """Tests for CitationFormat enum."""

    def test_format_types(self):
        """Test citation format types."""
        assert CitationFormat.STANDARD.value == "standard"
        assert CitationFormat.WITH_PARAGRAPH.value == "with_paragraph"
        assert CitationFormat.WITH_SUB_ARTICLE.value == "with_sub_article"
        assert CitationFormat.INCOMPLETE.value == "incomplete"


class TestCitationVerificationServiceVerifyGrounding:
    """Tests for verify_grounding method (TASK-006)."""

    def test_verify_grounding_returns_true_when_citation_found(self):
        """Test verify_grounding returns True when citation exists in source chunks."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {
                "text": "휴학은 2년을 초과할 수 없다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            },
            {
                "text": "다른 내용",
                "metadata": {"regulation_name": "등록금에 관한 규정", "article": 4},
            },
        ]

        result = service.verify_grounding(citation, source_chunks)

        assert result is True

    def test_verify_grounding_returns_false_when_citation_not_found(self):
        """Test verify_grounding returns False when citation does not exist."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=99,
            original_text="「학칙」 제99조",
        )
        source_chunks = [
            {
                "text": "휴학은 2년을 초과할 수 없다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            },
        ]

        result = service.verify_grounding(citation, source_chunks)

        assert result is False

    def test_verify_grounding_returns_false_when_regulation_not_found(self):
        """Test verify_grounding returns False when regulation name does not match."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {
                "text": "다른 규정 내용",
                "metadata": {"regulation_name": "등록금에 관한 규정", "article": 25},
            },
        ]

        result = service.verify_grounding(citation, source_chunks)

        assert result is False

    def test_verify_grounding_with_empty_source_chunks(self):
        """Test verify_grounding returns False with empty source chunks."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )

        result = service.verify_grounding(citation, [])

        assert result is False

    def test_verify_grounding_with_paragraph_match(self):
        """Test verify_grounding matches citation with paragraph."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="등록금에 관한 규정",
            article=4,
            paragraph=2,
            original_text="「등록금에 관한 규정」 제4조 제2항",
        )
        source_chunks = [
            {
                "text": "등록금 납부 관련 내용",
                "metadata": {
                    "regulation_name": "등록금에 관한 규정",
                    "article": 4,
                    "paragraph": 2,
                },
            },
        ]

        result = service.verify_grounding(citation, source_chunks)

        assert result is True

    def test_verify_grounding_without_paragraph_still_matches(self):
        """Test citation without paragraph matches chunk with same article."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {
                "text": "휴학 관련 내용",
                "metadata": {
                    "regulation_name": "학칙",
                    "article": 25,
                    "paragraph": 1,  # Chunk has paragraph but citation doesn't require it
                },
            },
        ]

        result = service.verify_grounding(citation, source_chunks)

        assert result is True


class TestCitationVerificationServiceIncludeContent:
    """Tests for include_content method (TASK-007)."""

    def test_include_content_returns_citation_with_content(self):
        """Test include_content adds content from matching chunk."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {
                "text": "휴학은 2년을 초과할 수 없다. 추가 내용이 있다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            },
        ]

        result = service.include_content(citation, source_chunks)

        assert result.content != ""
        assert "휴학" in result.content

    def test_include_content_returns_original_citation_if_not_found(self):
        """Test include_content returns original citation if no match found."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=99,
            original_text="「학칙」 제99조",
        )
        source_chunks = [
            {
                "text": "다른 내용",
                "metadata": {"regulation_name": "학칙", "article": 25},
            },
        ]

        result = service.include_content(citation, source_chunks)

        assert result.content == ""

    def test_include_content_with_empty_source_chunks(self):
        """Test include_content handles empty source chunks."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )

        result = service.include_content(citation, [])

        assert result.content == ""

    def test_include_content_extracts_first_sentence(self):
        """Test include_content extracts first sentence as key content."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {
                "text": "첫 번째 문장이다. 두 번째 문장이다. 세 번째 문장이다.",
                "metadata": {"regulation_name": "학칙", "article": 25},
            },
        ]

        result = service.include_content(citation, source_chunks)

        assert "첫 번째 문장" in result.content

    def test_include_content_preserves_other_citation_fields(self):
        """Test include_content preserves other citation fields."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            paragraph=2,
            original_text="「학칙」 제25조 제2항",
        )
        source_chunks = [
            {
                "text": "내용",
                "metadata": {"regulation_name": "학칙", "article": 25, "paragraph": 2},
            },
        ]

        result = service.include_content(citation, source_chunks)

        assert result.regulation_name == "학칙"
        assert result.article == 25
        assert result.paragraph == 2


class TestCitationVerificationServiceSanitizeUnverifiable:
    """Tests for sanitize_unverifiable method (TASK-008)."""

    def test_sanitize_unverifiable_returns_generalized_phrase(self):
        """Test sanitize_unverifiable returns generalized phrase for unverified citation."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
            is_verified=False,
        )

        result = service.sanitize_unverifiable(citation)

        assert result in ["관련 규정에 따르면", "해당 규정의 구체적 조항 확인이 필요합니다"]

    def test_sanitize_unverifiable_handles_verified_citation(self):
        """Test sanitize_unverifiable handles already verified citation."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
            content="휴학은 2년을 초과할 수 없다",
            is_verified=True,
        )

        result = service.sanitize_unverifiable(citation)

        # For verified citations, return the original citation or content
        assert result != ""

    def test_sanitize_unverifiable_with_missing_regulation_name(self):
        """Test sanitize_unverifiable handles citation with missing regulation name."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="",
            article=25,
            original_text="제25조",
            is_verified=False,
        )

        result = service.sanitize_unverifiable(citation)

        assert result != ""

    def test_sanitize_unverifiable_with_missing_article(self):
        """Test sanitize_unverifiable handles citation with missing article."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=0,
            original_text="「학칙」",
            is_verified=False,
        )

        result = service.sanitize_unverifiable(citation)

        assert result != ""

    def test_sanitize_unverifiable_returns_meaningful_fallback(self):
        """Test sanitize_unverifiable returns meaningful fallback message."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="존재하지않는규정",
            article=999,
            original_text="「존재하지않는규정」 제999조",
            is_verified=False,
        )

        result = service.sanitize_unverifiable(citation)

        # Should return a fallback that indicates verification is needed
        assert "확인" in result or "규정" in result


class TestCitationVerificationServiceEdgeCases:
    """Additional tests for edge cases to improve coverage."""

    def test_extract_citation_with_sub_article_format(self):
        """Test to_standard_format with sub_article (line 52)."""
        citation = ExtractedCitation(
            regulation_name="교원인사규정",
            article=10,
            sub_article=2,
            original_text="「교원인사규정」 제10조의2",
        )

        result = citation.to_standard_format()

        assert result == "「교원인사규정」 제10조의2"

    def test_extract_citations_empty_text(self):
        """Test extract_citations with empty text (line 121)."""
        service = CitationVerificationService()

        result = service.extract_citations("")

        assert result == []

    def test_extract_citations_whitespace_text(self):
        """Test extract_citations with whitespace-only text."""
        service = CitationVerificationService()

        result = service.extract_citations("   ")

        assert result == []

    def test_include_content_with_empty_text_in_chunk(self):
        """Test include_content when chunk has empty text (line 315)."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {
                "text": "",  # Empty text
                "metadata": {"regulation_name": "학칙", "article": 25},
            },
        ]

        result = service.include_content(citation, source_chunks)

        # Should return original citation with empty content
        assert result.content == ""

    def test_extract_first_sentence_empty_text(self):
        """Test _extract_first_sentence with empty text (line 355)."""
        service = CitationVerificationService()

        result = service._extract_first_sentence("")

        assert result == ""

    def test_extract_first_sentence_long_text_no_ending(self):
        """Test _extract_first_sentence with long text no sentence ending (line 375)."""
        service = CitationVerificationService()
        # Create text longer than 100 characters with no sentence ending
        long_text = "이것은 매우 긴 텍스트입니다 문장 끝이 없이 계속 이어지는 긴 텍스트입니다 백 자를 넘으면 말줄임표로 잘라야 합니다 추가로 더 긴 내용을 넣어서 확실히 백 자를 넘게 만들어야 합니다 그래서 이렇게 더 추가합니다"

        result = service._extract_first_sentence(long_text)

        assert "..." in result
        assert len(result) <= 105  # 100 + "..."

    def test_extract_first_sentence_short_text_no_ending(self):
        """Test _extract_first_sentence with short text no sentence ending (line 372)."""
        service = CitationVerificationService()
        short_text = "짧은 텍스트"

        result = service._extract_first_sentence(short_text)

        assert result == "짧은 텍스트"

    def test_extract_first_sentence_with_korean_period(self):
        """Test _extract_first_sentence with Korean period."""
        service = CitationVerificationService()
        text = "첫 번째 문장이다。두 번째 문장이다。"

        result = service._extract_first_sentence(text)

        assert result == "첫 번째 문장이다。"

    def test_extract_first_sentence_with_exclamation(self):
        """Test _extract_first_sentence with exclamation mark."""
        service = CitationVerificationService()
        text = "중요한 내용입니다! 추가 설명입니다."

        result = service._extract_first_sentence(text)

        assert result == "중요한 내용입니다!"

    def test_extract_first_sentence_with_question(self):
        """Test _extract_first_sentence with question mark."""
        service = CitationVerificationService()
        text = "이것이 맞습니까? 확인이 필요합니다."

        result = service._extract_first_sentence(text)

        assert result == "이것이 맞습니까?"

    def test_extract_first_sentence_returns_earliest_ending(self):
        """Test _extract_first_sentence returns the earliest sentence ending."""
        service = CitationVerificationService()
        text = "첫 문장! 두 번째 문장."

        result = service._extract_first_sentence(text)

        assert result == "첫 문장!"

    def test_sanitize_unverifiable_verified_but_no_content(self):
        """Test sanitize_unverifiable when verified but no content."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
            content="",  # Empty content
            is_verified=True,
        )

        result = service.sanitize_unverifiable(citation)

        # Should return fallback since content is empty
        assert result == "관련 규정에 따르면"

    def test_verify_grounding_chunk_without_metadata(self):
        """Test verify_grounding handles chunk without metadata."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {"text": "내용만 있는 청크"},  # No metadata
        ]

        result = service.verify_grounding(citation, source_chunks)

        assert result is False

    def test_include_content_chunk_without_metadata(self):
        """Test include_content handles chunk without metadata."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {"text": "내용만 있는 청크"},  # No metadata
        ]

        result = service.include_content(citation, source_chunks)

        assert result.content == ""

    def test_include_content_chunk_without_text(self):
        """Test include_content handles chunk without text field."""
        service = CitationVerificationService()
        citation = ExtractedCitation(
            regulation_name="학칙",
            article=25,
            original_text="「학칙」 제25조",
        )
        source_chunks = [
            {"metadata": {"regulation_name": "학칙", "article": 25}},  # No text
        ]

        result = service.include_content(citation, source_chunks)

        assert result.content == ""
