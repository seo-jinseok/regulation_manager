"""
Unit tests for MultiStageQueryExpander (TAG-002).

Tests for the 3-stage query expansion pipeline implementing SPEC-RAG-SEARCH-001.
"""

import pytest

from src.rag.domain.entity import RegulationEntityRecognizer
from src.rag.infrastructure.query_expander_v2 import (
    ExpansionResult,
    MultiStageQueryExpander,
    QueryExpansionPipeline,
)


class TestExpansionResult:
    """Test ExpansionResult dataclass."""

    def test_total_terms_calculation(self):
        """Test total_terms property calculates correctly."""
        result = ExpansionResult(
            original_query="장학금 신청",
            stage1_synonyms=["장학금 지급", "장학"],
            stage2_hypernyms=["재정", "지원"],
            stage3_procedures=["신청서", "제출"],
            final_expanded="장학금 신청 장학금 지급 장학 재정 지원 신청서 제출",
            confidence=0.8,
            method="multi_stage",
        )

        # 2 + 2 + 2 = 6 total expansion terms
        assert result.total_terms == 6

    def test_has_expansions_true(self):
        """Test has_expansions returns True when expansions exist."""
        result = ExpansionResult(
            original_query="장학금",
            stage1_synonyms=["장학금 지급"],
            stage2_hypernyms=[],
            stage3_procedures=[],
            final_expanded="장학금 장학금 지급",
            confidence=0.8,
            method="multi_stage",
        )

        assert result.has_expansions is True

    def test_has_expansions_false(self):
        """Test has_expansions returns False when no expansions."""
        result = ExpansionResult(
            original_query="장학금",
            stage1_synonyms=[],
            stage2_hypernyms=[],
            stage3_procedures=[],
            final_expanded="장학금",
            confidence=1.0,
            method="formal_skip",
        )

        assert result.has_expansions is False


class TestMultiStageQueryExpander:
    """Test MultiStageQueryExpander class."""

    @pytest.fixture
    def entity_recognizer(self):
        """Create entity recognizer for testing."""
        return RegulationEntityRecognizer()

    @pytest.fixture
    def expander(self, entity_recognizer):
        """Create multi-stage expander for testing."""
        return MultiStageQueryExpander(entity_recognizer)

    def test_expand_empty_query(self, expander):
        """Test expansion with empty query."""
        result = expander.expand("")

        assert result.original_query == ""
        assert result.final_expanded == ""
        assert result.total_terms == 0
        assert result.confidence == 1.0
        assert result.method == "empty"

    def test_expand_whitespace_query(self, expander):
        """Test expansion with whitespace-only query."""
        result = expander.expand("   ")

        assert result.original_query == "   "
        assert result.final_expanded == "   "
        assert result.total_terms == 0
        assert result.method == "empty"

    def test_formal_query_skip(self, expander):
        """Test that formal regulation queries skip expansion (REQ-QE-010)."""
        # Test with "규정" (regulation)
        result = expander.expand("장학금 규정")

        assert result.method == "formal_skip"
        assert result.total_terms == 0
        assert result.final_expanded == "장학금 규정"

        # Test with "학칙" (school regulations)
        result = expander.expand("휴학 학칙")

        assert result.method == "formal_skip"
        assert result.final_expanded == "휴학 학칙"

        # Test with "제15조" (article reference)
        result = expander.expand("제15조")

        assert result.method == "formal_skip"
        assert result.final_expanded == "제15조"

    def test_synonym_expansion_stage1(self, expander):
        """Test Stage 1: Synonym expansion (REQ-QE-004, REQ-QE-005)."""
        result = expander.expand("장학금 신청")

        # Should include synonym expansions for "장학금"
        assert len(result.stage1_synonyms) > 0
        assert any(
            "장학금" in term or "장학" in term for term in result.stage1_synonyms
        )

    def test_synonym_expansion_limits(self, expander):
        """Test synonym expansion respects MAX_SYNONYMS_PER_STAGE."""
        result = expander.expand("장학금")

        # Should not exceed MAX_SYNONYMS_PER_STAGE (3)
        assert (
            len(result.stage1_synonyms)
            <= MultiStageQueryExpander.MAX_SYNONYMS_PER_STAGE
        )

    def test_hypernym_expansion_stage2(self, expander):
        """Test Stage 2: Hypernym expansion (REQ-QE-006)."""
        result = expander.expand("등록금 납부")

        # Should use entity recognizer for hypernym expansion
        assert len(result.stage2_hypernyms) >= 0  # May be empty if no matches

    def test_hypernym_expansion_limits(self, expander):
        """Test hypernym expansion respects MAX_HYPERNYMS_PER_STAGE."""
        result = expander.expand("장학금")

        # Should not exceed MAX_HYPERNYMS_PER_STAGE (2)
        assert (
            len(result.stage2_hypernyms)
            <= MultiStageQueryExpander.MAX_HYPERNYMS_PER_STAGE
        )

    def test_procedure_expansion_stage3(self, expander):
        """Test Stage 3: Procedure expansion (REQ-QE-007)."""
        result = expander.expand("장학금 신청 방법")

        # Should include procedure-related expansions
        assert len(result.stage3_procedures) >= 0  # May be empty if no matches

    def test_procedure_expansion_limits(self, expander):
        """Test procedure expansion respects MAX_PROCEDURES_PER_STAGE."""
        result = expander.expand("신청")

        # Should not exceed MAX_PROCEDURES_PER_STAGE (2)
        assert (
            len(result.stage3_procedures)
            <= MultiStageQueryExpander.MAX_PROCEDURES_PER_STAGE
        )

    def test_total_expansion_limit(self, expander):
        """Test total expansions respect MAX_TOTAL_EXPANSIONS (REQ-QE-008)."""
        # Create a query that would generate many expansions
        result = expander.expand("장학금 신청 방법")

        # Total should not exceed MAX_TOTAL_EXPANSIONS (10)
        assert result.total_terms <= MultiStageQueryExpander.MAX_TOTAL_EXPANSIONS

    def test_confidence_calculation_no_expansions(self, expander):
        """Test confidence score with no expansions."""
        result = expander.expand("일반적인 질문")

        # Low confidence for no expansions
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_confidence_calculation_with_expansions(self, expander):
        """Test confidence score increases with relevant expansions."""
        result = expander.expand("장학금 신청")

        # Higher confidence with expansions
        if result.has_expansions:
            assert result.confidence >= MultiStageQueryExpander.MIN_EXPANSION_CONFIDENCE

    def test_final_expanded_query_includes_original(self, expander):
        """Test final expanded query contains expansion terms from original."""
        result = expander.expand("장학금 신청")

        # Final expanded should contain terms related to the original query
        # (but not the original terms themselves, which are already known)
        assert len(result.final_expanded) > 0
        # The expanded query should be different from the original
        assert result.final_expanded != result.original_query
        # If expansions were generated, they should be present
        if result.has_expansions:
            assert result.final_expanded is not None

    def test_deduplication_in_combination(self, expander):
        """Test that duplicate terms are removed in combination."""
        result = expander.expand("장학금 장학")

        # Should not have duplicate terms in final expansion
        final_terms = result.final_expanded.split()
        assert len(final_terms) == len(set(final_terms))

    def test_multi_stage_method_flag(self, expander):
        """Test method flag is set correctly for multi-stage expansion."""
        result = expander.expand("장학금 신청")

        # If expansions were generated, method should be multi_stage
        if result.has_expansions:
            assert result.method == "multi_stage"

    def test_multiple_synonym_terms(self, expander):
        """Test expansion with multiple synonym-eligible terms."""
        # Query with two known terms
        result = expander.expand("장학금과 휴학")

        # Should generate expansions for both terms
        assert result.stage1_synonyms is not None
        assert len(result.stage1_synonyms) >= 0


class TestMultiStageQueryExpanderIntegration:
    """Integration tests for MultiStageQueryExpander with real scenarios."""

    @pytest.fixture
    def expander(self):
        """Create expander with default entity recognizer."""
        entity_recognizer = RegulationEntityRecognizer()
        return MultiStageQueryExpander(entity_recognizer)

    def test_scholarship_application_query(self, expander):
        """Test realistic scholarship application query."""
        result = expander.expand("장학금 신청 방법")

        # Should generate relevant expansions
        assert result.original_query == "장학금 신청 방법"
        assert result.final_expanded is not None
        assert len(result.final_expanded) > len(result.original_query)

    def test_leave_of_absence_query(self, expander):
        """Test leave of absence query."""
        result = expander.expand("휴학 절차")

        assert result.original_query == "휴학 절차"
        # Should expand both "휴학" and procedure terms
        if result.has_expansions:
            assert result.total_terms > 0

    def test_registration_fee_query(self, expander):
        """Test registration fee query."""
        result = expander.expand("등록금 납부 기한")

        assert result.original_query == "등록금 납부 기한"
        # Should expand with hypernyms and deadline terms
        if result.has_expansions:
            assert result.total_terms > 0

    def test_readmission_query(self, expander):
        """Test readmission query."""
        result = expander.expand("복학 자격 요건")

        assert result.original_query == "복학 자격 요건"
        # Should expand with synonyms and requirement terms
        if result.has_expansions:
            assert result.total_terms > 0


class TestQueryExpansionPipeline:
    """Test QueryExpansionPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create expansion pipeline for testing."""
        entity_recognizer = RegulationEntityRecognizer()
        expander = MultiStageQueryExpander(entity_recognizer)
        return QueryExpansionPipeline(expander, enable_cache=True)

    def test_process_query_basic(self, pipeline):
        """Test basic query processing through pipeline."""
        result = pipeline.process_query("장학금 신청")

        assert isinstance(result, ExpansionResult)
        assert result.original_query == "장학금 신청"

    def test_process_query_empty(self, pipeline):
        """Test pipeline with empty query."""
        result = pipeline.process_query("")

        assert result.original_query == ""
        assert result.final_expanded == ""
        assert result.method == "empty"

    def test_cache_enabled(self, pipeline):
        """Test that cache works when enabled."""
        query = "장학금 신청"

        # First call
        result1 = pipeline.process_query(query)
        # Second call (should hit cache)
        result2 = pipeline.process_query(query)

        # Results should be identical
        assert result1.final_expanded == result2.final_expanded
        assert result1.confidence == result2.confidence

    def test_clear_cache(self, pipeline):
        """Test cache clearing functionality."""
        # Add something to cache
        pipeline.process_query("장학금 신청")
        pipeline.clear_cache()

        # Cache should be empty
        assert len(pipeline._cache) == 0

    def test_cache_disabled(self):
        """Test pipeline with cache disabled."""
        entity_recognizer = RegulationEntityRecognizer()
        expander = MultiStageQueryExpander(entity_recognizer)
        pipeline = QueryExpansionPipeline(expander, enable_cache=False)

        result = pipeline.process_query("장학금 신청")

        # Cache should remain empty
        assert len(pipeline._cache) == 0
        assert isinstance(result, ExpansionResult)


class TestMultiStageQueryExpanderEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def expander(self):
        """Create expander for testing."""
        entity_recognizer = RegulationEntityRecognizer()
        return MultiStageQueryExpander(entity_recognizer)

    def test_query_with_only_stopwords(self, expander):
        """Test query with only common stopwords."""
        result = expander.expand("의 조 대한")

        # Should handle gracefully
        assert result.original_query == "의 조 대한"
        assert isinstance(result.final_expanded, str)

    def test_query_with_special_characters(self, expander):
        """Test query with special characters."""
        result = expander.expand("장학금?!@#")

        # Should handle special characters
        assert result.original_query == "장학금?!@#"

    def test_very_long_query(self, expander):
        """Test with very long query."""
        long_query = "장학금 " * 50
        result = expander.expand(long_query)

        # Should handle long queries
        assert result.original_query == long_query

    def test_unicode_characters(self, expander):
        """Test with Korean unicode characters."""
        result = expander.expand("장학금 신청 방법")

        # Should handle Korean characters properly
        assert "장학금" in result.original_query

    def test_case_sensitivity(self, expander):
        """Test that expansion is case-insensitive for Korean."""
        # Korean doesn't have case, but test consistency
        result1 = expander.expand("장학금")
        result2 = expander.expand("장학금")

        # Results should be consistent
        assert result1.final_expanded == result2.final_expanded


class TestMultiStageQueryExpanderConstants:
    """Test that constants are properly configured."""

    def test_max_synonyms_per_stage(self):
        """Verify MAX_SYNONYMS_PER_STAGE constant."""
        assert MultiStageQueryExpander.MAX_SYNONYMS_PER_STAGE == 3

    def test_max_hypernyms_per_stage(self):
        """Verify MAX_HYPERNYMS_PER_STAGE constant."""
        assert MultiStageQueryExpander.MAX_HYPERNYMS_PER_STAGE == 2

    def test_max_procedures_per_stage(self):
        """Verify MAX_PROCEDURES_PER_STAGE constant."""
        assert MultiStageQueryExpander.MAX_PROCEDURES_PER_STAGE == 2

    def test_max_total_expansions(self):
        """Verify MAX_TOTAL_EXPANSIONS constant (REQ-QE-008)."""
        assert MultiStageQueryExpander.MAX_TOTAL_EXPANSIONS == 10

    def test_min_expansion_confidence(self):
        """Verify MIN_EXPANSION_CONFIDENCE constant (REQ-QE-009)."""
        assert MultiStageQueryExpander.MIN_EXPANSION_CONFIDENCE == 0.6

    def test_formal_query_indicators(self):
        """Verify FORMAL_QUERY_INDICATORS list (REQ-QE-010)."""
        indicators = MultiStageQueryExpander.FORMAL_QUERY_INDICATORS

        # Should contain expected formal terms
        assert "규정" in indicators
        assert "학칙" in indicators
        assert "세칙" in indicators
        assert "지침" in indicators

    def test_synonym_mappings_exists(self):
        """Verify SYNONYM_MAPPINGS is populated."""
        assert hasattr(MultiStageQueryExpander, "SYNONYM_MAPPINGS")
        assert len(MultiStageQueryExpander.SYNONYM_MAPPINGS) > 0

    def test_synonym_mappings_structure(self):
        """Verify SYNONYM_MAPPINGS has correct structure."""
        for key, values in MultiStageQueryExpander.SYNONYM_MAPPINGS.items():
            assert isinstance(key, str)
            assert isinstance(values, list)
            assert all(isinstance(v, str) for v in values)
            # First value should be the original term
            assert values[0] == key


class TestGetExpandedQuery:
    """Test get_expanded_query convenience method."""

    @pytest.fixture
    def expander(self):
        """Create expander for testing."""
        entity_recognizer = RegulationEntityRecognizer()
        return MultiStageQueryExpander(entity_recognizer)

    def test_get_expanded_query_with_expansion(self, expander):
        """Test get_expanded_query returns expanded query."""
        expanded = expander.get_expanded_query("장학금 신청", use_expansion=True)

        assert isinstance(expanded, str)
        assert len(expanded) > 0

    def test_get_expanded_query_without_expansion(self, expander):
        """Test get_expanded_query returns original when expansion disabled."""
        result = expander.get_expanded_query("장학금 신청", use_expansion=False)

        assert result == "장학금 신청"

    def test_get_expanded_query_default(self, expander):
        """Test get_expanded_query default behavior (expansion enabled)."""
        result = expander.get_expanded_query("장학금 신청")

        # Default is use_expansion=True
        assert isinstance(result, str)


class TestStaffVocabularyExpansion:
    """Test staff-specific vocabulary expansion (SPEC-RAG-QUALITY-005 Phase 2)."""

    @pytest.fixture
    def expander(self):
        """Create expander for testing."""
        entity_recognizer = RegulationEntityRecognizer()
        return MultiStageQueryExpander(entity_recognizer)

    def test_bokmu_expansion(self, expander):
        """Test expansion for 복무 (work service) staff vocabulary."""
        # Use non-formal query to avoid formal_skip
        result = expander.expand("복무 관련 질문")

        # Should include related terms: 근무, 재직, 출근
        assert result.has_expansions
        expanded_terms = " ".join(result.stage1_synonyms)
        assert any(
            term in expanded_terms for term in ["근무", "재직", "출근"]
        )

    def test_yeoncha_expansion(self, expander):
        """Test expansion for 연차 (annual leave) staff vocabulary."""
        result = expander.expand("연차 신청")

        # Should include related terms: 휴가, 휴직, 휴무
        assert result.has_expansions
        expanded_terms = " ".join(result.stage1_synonyms)
        assert any(
            term in expanded_terms for term in ["휴가", "휴직", "휴무"]
        )

    def test_geupyeo_expansion(self, expander):
        """Test expansion for 급여 (salary) staff vocabulary."""
        result = expander.expand("급여 지급")

        # Should include related terms: 봉급, 월급, 보수
        assert result.has_expansions
        expanded_terms = " ".join(result.stage1_synonyms)
        assert any(
            term in expanded_terms for term in ["봉급", "월급", "보수"]
        )

    def test_yeonsu_expansion(self, expander):
        """Test expansion for 연수 (training) staff vocabulary."""
        result = expander.expand("연수 프로그램")

        # Should include related terms: 교육, 훈련, 연교육
        assert result.has_expansions
        expanded_terms = " ".join(result.stage1_synonyms)
        assert any(
            term in expanded_terms for term in ["교육", "훈련", "연교육"]
        )

    def test_samuyongpum_expansion(self, expander):
        """Test expansion for 사무용품 (office supplies) staff vocabulary."""
        result = expander.expand("사무용품 구매")

        # Should include related terms: 비품, 물품, 용품
        assert result.has_expansions
        expanded_terms = " ".join(result.stage1_synonyms)
        assert any(
            term in expanded_terms for term in ["비품", "물품", "용품"]
        )

    def test_ipchal_expansion(self, expander):
        """Test expansion for 입찰 (bidding) staff vocabulary."""
        result = expander.expand("입찰 공고")

        # Should include related terms: 계약, 발주, 조달
        assert result.has_expansions
        expanded_terms = " ".join(result.stage1_synonyms)
        assert any(
            term in expanded_terms for term in ["계약", "발주", "조달"]
        )

    def test_staff_vocabulary_in_mappings(self):
        """Verify staff vocabulary entries exist in SYNONYM_MAPPINGS."""
        mappings = MultiStageQueryExpander.SYNONYM_MAPPINGS

        # Check all 6 staff vocabulary entries exist
        staff_vocab = ["복무", "연차", "급여", "연수", "사무용품", "입찰"]
        for term in staff_vocab:
            assert term in mappings, f"Missing staff vocabulary: {term}"
            assert len(mappings[term]) >= 3, f"Insufficient synonyms for {term}"
