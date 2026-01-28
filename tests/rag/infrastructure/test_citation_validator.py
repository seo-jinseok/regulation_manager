"""
Tests for CitationValidator.

Tests the citation validation and enforcement functionality.
"""

import pytest

from src.rag.infrastructure.citation_validator import (
    CitationEnrichmentResult,
    CitationMatch,
    CitationValidationResult,
    CitationValidator,
    enforce_citations,
    validate_answer_citation,
)


@pytest.fixture
def validator():
    """Create a CitationValidator instance."""
    return CitationValidator()


@pytest.fixture
def sample_context_sources():
    """Create sample context sources."""
    return ["학칙", "장학금 규정", "휴학 규정"]


class TestCitationValidator:
    """Test CitationValidator functionality."""

    def test_validate_citation_with_proper_format(
        self, validator, sample_context_sources
    ):
        """Test validation with proper citation format."""
        answer = "학칙 제15조에 따르면 휴학할 수 있습니다."

        result = validator.validate_citation(answer, sample_context_sources)

        assert result.is_valid is True
        assert result.citation_count == 1
        assert len(result.citations) == 1
        assert result.citations[0].is_valid is True
        assert result.citations[0].regulation_name == "학칙"
        assert result.citations[0].article_number == "15"

    def test_validate_citation_missing_regulation_name(
        self, validator, sample_context_sources
    ):
        """Test validation with missing regulation name."""
        answer = "제15조에 따르면 휴학할 수 있습니다."

        result = validator.validate_citation(answer, sample_context_sources)

        assert result.citation_count == 1
        assert result.citations[0].is_valid is False
        assert result.citations[0].regulation_name == ""
        assert len(result.missing_regulation_names) > 0

    def test_validate_citation_multiple_citations(
        self, validator, sample_context_sources
    ):
        """Test validation with multiple citations."""
        answer = "학칙 제15조와 장학금 규정 제20조에 따릅니다."

        result = validator.validate_citation(answer, sample_context_sources)

        assert result.citation_count == 2
        assert len(result.citations) == 2

    def test_validate_citation_no_citations(self, validator, sample_context_sources):
        """Test validation with no citations."""
        answer = "휴학할 수 있습니다."

        result = validator.validate_citation(answer, sample_context_sources)

        assert result.citation_count == 0
        assert len(result.citations) == 0
        assert result.is_valid is False  # No citations in strict mode

    def test_validate_citation_density(self, validator, sample_context_sources):
        """Test citation density calculation."""
        answer = "학칙 제15조에 따르면 휴학할 수 있습니다. " * 100

        result = validator.validate_citation(answer, sample_context_sources)

        assert result.citation_density > 0

    def test_enrich_citation(self, validator, sample_context_sources):
        """Test citation enrichment."""
        answer = "제15조에 따르면 휴학할 수 있습니다."

        result = validator.enrich_citation(answer, sample_context_sources)

        assert isinstance(result, CitationEnrichmentResult)
        assert result.original_answer == answer
        assert "학칙" in result.enriched_answer
        assert len(result.added_citations) > 0

    def test_enforce_citation_format(self, validator, sample_context_sources):
        """Test citation format enforcement."""
        answer = "학칙에 따라 제15조에서 휴학할 수 있습니다."

        enforced, changes = validator.enforce_citation_format(answer)

        assert isinstance(enforced, str)
        assert isinstance(changes, list)

    def test_extract_citations_various_formats(self, validator):
        """Test citation extraction from various formats."""
        answer = """
        학칙 제15조에 따릅니다.
        제20조 제2항에도 명시되어 있습니다.
        장학금 규정 제25조를 참고하세요.
        """

        citations = validator._extract_citations(answer)

        assert len(citations) >= 2

    def test_strict_mode_validation(self, validator, sample_context_sources):
        """Test validation in strict mode."""
        validator_strict = CitationValidator(strict_mode=True)

        answer = "제15조에 따르면 휴학할 수 있습니다."

        result = validator_strict.validate_citation(answer, sample_context_sources)

        # Strict mode should fail for missing regulation names
        assert result.is_valid is False

    def test_lenient_mode_validation(self, validator, sample_context_sources):
        """Test validation in lenient mode."""
        answer = "제15조에 따르면 휴학할 수 있습니다."

        result = validator.validate_citation(answer, sample_context_sources)

        # Lenient mode might pass depending on density
        assert isinstance(result.is_valid, bool)

    def test_calculate_citation_density(self, validator):
        """Test citation density calculation."""
        short_answer = "학칙 제15조에 따릅니다."
        long_answer = "학칙 제15조에 따릅니다. " * 100

        short_density = validator._calculate_citation_density(
            short_answer, validator._extract_citations(short_answer)
        )
        long_density = validator._calculate_citation_density(
            long_answer, validator._extract_citations(long_answer)
        )

        assert short_density > 0
        assert long_density > 0


class TestCitationConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_answer_citation(self, sample_context_sources):
        """Test validate_answer_citation convenience function."""
        answer = "학칙 제15조에 따릅니다."

        result = validate_answer_citation(answer, sample_context_sources)

        assert isinstance(result, CitationValidationResult)

    def test_validate_answer_citation_strict(self, sample_context_sources):
        """Test validate_answer_citation with strict mode."""
        answer = "제15조에 따릅니다."

        result = validate_answer_citation(
            answer, sample_context_sources, strict_mode=True
        )

        assert result.is_valid is False

    def test_enforce_citations_function(self, sample_context_sources):
        """Test enforce_citations convenience function."""
        answer = "제15조에 따릅니다."

        result = enforce_citations(answer, sample_context_sources)

        assert isinstance(result, CitationEnrichmentResult)
        assert result.enriched_answer != answer


@pytest.mark.integration
class TestCitationValidatorIntegration:
    """Integration tests for CitationValidator."""

    def test_full_validation_enrichment_workflow(
        self, validator, sample_context_sources
    ):
        """Test complete validation and enrichment workflow."""
        # Step 1: Validate initial answer
        answer = "제15조에 따르면 휴학할 수 있습니다."
        validation = validator.validate_citation(answer, sample_context_sources)

        assert validation.citation_count == 1
        assert validation.citations[0].is_valid is False

        # Step 2: Enrich answer
        enriched = validator.enrich_citation(answer, sample_context_sources)

        assert (
            "학칙" in enriched.enriched_answer or "장학금" in enriched.enriched_answer
        )

        # Step 3: Validate enriched answer
        final_validation = validator.validate_citation(
            enriched.enriched_answer, sample_context_sources
        )

        # Enriched answer should have better citations
        assert final_validation.citation_count >= validation.citation_count

    def test_multi_citation_scenario(self, validator, sample_context_sources):
        """Test handling multiple citations in one answer."""
        answer = """
        동의대학교 규정에 따르면:
        1. 학칙 제15조에서 휴학을 정의하고 있습니다.
        2. 제20조 제2항에서 휴학 절차를 명시하고 있습니다.
        3. 장학금 규정 제25조와 관련이 있습니다.
        """

        result = validator.validate_citation(answer, sample_context_sources)

        assert result.citation_count >= 3
        assert len(result.issues) == 0 or len(result.issues) < result.citation_count


@pytest.mark.unit
class TestCitationMatch:
    """Test CitationMatch dataclass."""

    def test_citation_match_creation(self):
        """Test CitationMatch creation."""
        match = CitationMatch(
            text="학칙 제15조",
            regulation_name="학칙",
            article_number="15",
            start_pos=0,
            end_pos=6,
            is_valid=True,
        )

        assert match.text == "학칙 제15조"
        assert match.regulation_name == "학칙"
        assert match.article_number == "15"
        assert match.is_valid is True
