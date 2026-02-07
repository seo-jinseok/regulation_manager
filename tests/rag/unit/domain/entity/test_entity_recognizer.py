"""
Unit tests for RegulationEntityRecognizer.

Tests for SPEC-RAG-SEARCH-001 TAG-001: Enhanced Entity Recognition.

Validates:
- 6 new entity types are recognized correctly
- Confidence scores are appropriate
- Expansion terms are generated correctly
- Edge cases are handled properly
"""

import pytest

from src.rag.domain.entity import (
    EntityMatch,
    EntityRecognitionResult,
    EntityType,
    RegulationEntityRecognizer,
)


class TestRegulationEntityRecognizer:
    """Unit tests for RegulationEntityRecognizer."""

    @pytest.fixture
    def recognizer(self) -> RegulationEntityRecognizer:
        """Fresh recognizer instance for each test."""
        return RegulationEntityRecognizer()

    # ============ SECTION Recognition Tests ============

    def test_recognize_section_제N조(self, recognizer: RegulationEntityRecognizer):
        """Test recognition of '제N조' pattern (REQ-ER-001)."""
        result = recognizer.recognize("제15조의 휴학 절차")

        assert result.has_entities
        section_matches = [
            m for m in result.matches if m.entity_type == EntityType.SECTION
        ]

        assert len(section_matches) >= 1
        assert any(m.text == "제15조" for m in section_matches)

        # SECTION should have high confidence
        section_match = section_matches[0]
        assert section_match.confidence >= 0.90
        assert section_match.is_high_confidence

    def test_recognize_section_multiple_sections(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test recognition of multiple section references."""
        result = recognizer.recognize("제15조와 제20조의 관계")

        section_matches = [
            m for m in result.matches if m.entity_type == EntityType.SECTION
        ]
        assert len(section_matches) >= 2

    def test_recognize_section_with_spaces(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test recognition of '제 N 조' with spaces."""
        result = recognizer.recognize("제 15 조의 휴학")

        section_matches = [
            m for m in result.matches if m.entity_type == EntityType.SECTION
        ]
        assert len(section_matches) >= 1

    # ============ PROCEDURE Recognition Tests ============

    def test_recognize_procedure_신청(self, recognizer: RegulationEntityRecognizer):
        """Test recognition of '신청' keyword (REQ-ER-002)."""
        result = recognizer.recognize("장학금 신청 방법")

        procedure_matches = [
            m for m in result.matches if m.entity_type == EntityType.PROCEDURE
        ]

        assert any(m.text == "신청" for m in procedure_matches)

        # Check expansion terms
        신청_match = [m for m in procedure_matches if m.text == "신청"][0]
        assert "신청서" in 신청_match.expanded_terms
        assert "제출" in 신청_match.expanded_terms

    def test_recognize_procedure_multiple_keywords(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test recognition of multiple procedure keywords."""
        result = recognizer.recognize("신청 절차 방법")

        procedure_matches = [
            m for m in result.matches if m.entity_type == EntityType.PROCEDURE
        ]

        assert len(procedure_matches) >= 3
        keywords_found = [m.text for m in procedure_matches]
        assert "신청" in keywords_found
        assert "절차" in keywords_found
        assert "방법" in keywords_found

    def test_recognize_procedure_expansion_quality(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test that procedure expansions are relevant."""
        test_cases = [
            ("신청", ["신청서", "제출", "등록"]),
            ("절차", ["방법", "과정", "단계"]),
            ("방법", ["절차", "방식"]),
        ]

        for keyword, expected_terms in test_cases:
            result = recognizer.recognize(f"휴학 {keyword}")
            matches = [m for m in result.matches if m.text == keyword]

            assert len(matches) >= 1, f"No match found for '{keyword}'"
            expanded = matches[0].expanded_terms

            for term in expected_terms:
                assert term in expanded, (
                    f"Expected '{term}' in expansion of '{keyword}': {expanded}"
                )

    # ============ REQUIREMENT Recognition Tests ============

    def test_recognize_requirement_자격(self, recognizer: RegulationEntityRecognizer):
        """Test recognition of '자격' keyword (REQ-ER-003)."""
        result = recognizer.recognize("연구년 자격 요건")

        requirement_matches = [
            m for m in result.matches if m.entity_type == EntityType.REQUIREMENT
        ]

        assert any(m.text == "자격" for m in requirement_matches)
        assert any(m.text == "요건" for m in requirement_matches)

        # Check expansion for 자격
        자격_match = [m for m in requirement_matches if m.text == "자격"]
        if 자격_match:
            assert "자격요건" in 자격_match[0].expanded_terms

    def test_recognize_requirement_multiple_keywords(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test recognition of multiple requirement keywords."""
        result = recognizer.recognize("자격 요건 조건 기준")

        requirement_matches = [
            m for m in result.matches if m.entity_type == EntityType.REQUIREMENT
        ]

        assert len(requirement_matches) >= 3

    # ============ BENEFIT Recognition Tests ============

    def test_recognize_benefit_장학금(self, recognizer: RegulationEntityRecognizer):
        """Test recognition of '장학금' keyword (REQ-ER-004)."""
        result = recognizer.recognize("장학금 혜택 지급")

        benefit_matches = [
            m for m in result.matches if m.entity_type == EntityType.BENEFIT
        ]

        assert any(m.text == "장학금" for m in benefit_matches)
        assert any(m.text == "혜택" for m in benefit_matches)
        assert any(m.text == "지급" for m in benefit_matches)

    def test_recognize_benefit_expansion_quality(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test that benefit expansions are relevant."""
        test_cases = [
            ("혜택", ["지급", "지원", "급여"]),
            ("지급", ["발급", "지원"]),
            ("장학금", ["장학", "성적장학금", "재정지원"]),
        ]

        for keyword, expected_terms in test_cases:
            result = recognizer.recognize(keyword)
            matches = [m for m in result.matches if m.text == keyword]

            if matches:
                expanded = matches[0].expanded_terms
                for term in expected_terms:
                    assert term in expanded, (
                        f"Expected '{term}' in expansion of '{keyword}': {expanded}"
                    )

    # ============ DEADLINE Recognition Tests ============

    def test_recognize_deadline_기한(self, recognizer: RegulationEntityRecognizer):
        """Test recognition of '기한' keyword (REQ-ER-005)."""
        result = recognizer.recognize("신청 기한 마감 날짜")

        deadline_matches = [
            m for m in result.matches if m.entity_type == EntityType.DEADLINE
        ]

        assert len(deadline_matches) >= 3
        assert any(m.text == "기한" for m in deadline_matches)
        assert any(m.text == "마감" for m in deadline_matches)

    def test_recognize_deadline_various_terms(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test recognition of various deadline keywords."""
        deadline_terms = ["기한", "마감", "날짜", "기간", "까지", "이내"]

        for term in deadline_terms:
            result = recognizer.recognize(f"신청 {term}")
            deadline_matches = [
                m for m in result.matches if m.entity_type == EntityType.DEADLINE
            ]

            assert len(deadline_matches) >= 1, f"No deadline match for '{term}'"

    # ============ HYPERNYM Recognition Tests ============

    def test_recognize_hypernym_등록금(self, recognizer: RegulationEntityRecognizer):
        """Test hypernym expansion for '등록금' (REQ-ER-006)."""
        result = recognizer.recognize("등록금 납부")

        hypernym_matches = [
            m for m in result.matches if m.entity_type == EntityType.HYPERNYM
        ]

        assert len(hypernym_matches) >= 1

        # Check expansion includes hierarchical terms
        등록금_match = [m for m in hypernym_matches if m.text == "등록금"][0]
        assert "학사" in 등록금_match.expanded_terms
        assert "행정" in 등록금_match.expanded_terms

    def test_recognize_hypernym_장학금(self, recognizer: RegulationEntityRecognizer):
        """Test hypernym expansion for '장학금'."""
        result = recognizer.recognize("장학금 신청")

        hypernym_matches = [
            m for m in result.matches if m.entity_type == EntityType.HYPERNYM
        ]

        # 장학금 should also trigger HYPERNYM (it's in HYPERNYM_MAPPINGS)
        assert len(hypernym_matches) >= 1

        장학금_match = [m for m in hypernym_matches if m.text == "장학금"]
        if 장학금_match:
            assert "재정" in 장학금_match[0].expanded_terms
            assert "지원" in 장학금_match[0].expanded_terms

    def test_recognize_hypernym_professor_terms(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test hypernym expansion for professor-related terms."""
        result = recognizer.recognize("교수 승진")

        hypernym_matches = [
            m for m in result.matches if m.entity_type == EntityType.HYPERNYM
        ]

        # 교수 should have 교원, 교직원 in expansion
        교수_match = [m for m in hypernym_matches if m.text == "교수"]
        if 교수_match:
            assert "교원" in 교수_match[0].expanded_terms
            assert "교직원" in 교수_match[0].expanded_terms

    # ============ Comprehensive Recognition Tests ============

    def test_recognize_complex_query_multiple_entity_types(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test recognition of query with multiple entity types."""
        result = recognizer.recognize("장학금 신청 자격 요건 기한 내")

        # Should recognize multiple entity types
        entity_types_found = set(m.entity_type for m in result.matches)

        # Expected types: BENEFIT, PROCEDURE, REQUIREMENT, DEADLINE
        assert EntityType.BENEFIT in entity_types_found
        assert EntityType.PROCEDURE in entity_types_found
        assert EntityType.REQUIREMENT in entity_types_found
        assert EntityType.DEADLINE in entity_types_found

    def test_recognize_empty_query(self, recognizer: RegulationEntityRecognizer):
        """Test recognition of empty query."""
        result = recognizer.recognize("")

        assert not result.has_entities
        assert len(result.matches) == 0
        assert result.primary_entity is None

    def test_recognize_no_entities(self, recognizer: RegulationEntityRecognizer):
        """Test query with no recognizable entities."""
        result = recognizer.recognize("일반적인 문장입니다")

        # May still have some keyword matches, but should be limited
        # This documents current behavior
        print(f"\n[DEBUG] No-entity query matches: {len(result.matches)}")
        for m in result.matches:
            print(f"  - {m.entity_type.value}: {m.text} (confidence={m.confidence})")

    # ============ Confidence Filtering Tests ============

    def test_confidence_threshold_filters_low_confidence(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test that low-confidence matches are filtered out."""
        # Default threshold is 0.5
        result = recognizer.recognize("장학금 신청")

        # All matches should have confidence >= 0.5
        for match in result.matches:
            assert match.confidence >= 0.5, (
                f"Match '{match.text}' has low confidence: {match.confidence}"
            )

    def test_custom_confidence_threshold(self):
        """Test recognizer with custom confidence threshold."""
        strict_recognizer = RegulationEntityRecognizer(confidence_threshold=0.9)

        result = strict_recognizer.recognize("장학금 신청 기한")

        # All matches should have confidence >= 0.9
        for match in result.matches:
            assert match.confidence >= 0.9

        # Should have fewer matches than default threshold
        default_recognizer = RegulationEntityRecognizer()
        default_result = default_recognizer.recognize("장학금 신청 기한")

        assert len(result.matches) <= len(default_result.matches)

    # ============ Expansion Query Tests ============

    def test_get_expanded_query_simple(self, recognizer: RegulationEntityRecognizer):
        """Test query expansion for simple case."""
        expanded = recognizer.get_expanded_query("장학금 신청")

        # Should include original terms
        assert "장학금" in expanded
        assert "신청" in expanded

        # Should include expanded terms
        assert "장학" in expanded  # From BENEFIT expansion
        assert "신청서" in expanded or "제출" in expanded  # From PROCEDURE expansion

    def test_get_expanded_query_max_terms_limit(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test that max_terms parameter limits expansion."""
        # Query that would generate many expansions
        expanded_5 = recognizer.get_expanded_query("장학금 신청 자격 요건", max_terms=5)
        expanded_10 = recognizer.get_expanded_query(
            "장학금 신청 자격 요건", max_terms=10
        )

        # Count number of spaces + 1 (rough estimate of term count)
        terms_5 = len(set(expanded_5.split()))
        terms_10 = len(set(expanded_10.split()))

        # max_terms=5 should have fewer or equal terms than max_terms=10
        assert terms_5 <= terms_10

    def test_get_expanded_query_without_entities(self):
        """Test query expansion without entity recognition."""
        recognizer = RegulationEntityRecognizer()
        expanded = recognizer.get_expanded_query("장학금 신청", use_entities=False)

        # Should return original query unchanged
        assert expanded == "장학금 신청"

    # ============ EntityRecognitionResult Tests ============

    def test_entity_result_primary_entity_selection(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test that primary_entity is highest-confidence match."""
        result = recognizer.recognize("장학금 신청 방법")

        if result.has_entities:
            # Primary entity should have highest confidence
            assert result.primary_entity.confidence == max(
                m.confidence for m in result.matches
            )

    def test_entity_result_total_expanded_terms_unique(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test that total_expanded_terms contains unique terms."""
        result = recognizer.recognize("장학금 혜택 지급")

        if result.total_expanded_terms:
            # Check no duplicates
            assert len(result.total_expanded_terms) == len(
                set(result.total_expanded_terms)
            )

    def test_entity_result_get_expanded_query(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test EntityRecognitionResult.get_expanded_query method."""
        result = recognizer.recognize("장학금 신청")

        expanded = result.get_expanded_query(max_terms=10)

        # Should contain original query
        assert "장학금" in expanded or "신청" in expanded

        # Should be longer than original (if entities found)
        if result.total_expanded_terms:
            assert len(expanded) >= len(result.original_query)

    # ============ SPEC Scenario Tests ============

    def test_spec_scenario_1_scholarship_application(
        self, recognizer: RegulationEntityRecognizer
    ):
        """
        SPEC Scenario 1: "장학금 신청 방법"

        Expected entities:
        - BENEFIT: 장학금
        - PROCEDURE: 신청, 방법
        """
        result = recognizer.recognize("장학금 신청 방법")

        entity_types = {m.entity_type for m in result.matches}

        assert EntityType.BENEFIT in entity_types
        assert EntityType.PROCEDURE in entity_types

        # Check for expected keywords
        benefit_texts = [
            m.text for m in result.matches if m.entity_type == EntityType.BENEFIT
        ]
        procedure_texts = [
            m.text for m in result.matches if m.entity_type == EntityType.PROCEDURE
        ]

        assert "장학금" in benefit_texts
        assert "신청" in procedure_texts
        assert "방법" in procedure_texts

    def test_spec_scenario_2_research_year_eligibility(
        self, recognizer: RegulationEntityRecognizer
    ):
        """
        SPEC Scenario 2: "연구년 자격 요건"

        Expected entities:
        - HYPERNYM: 연구년 (expands to 안식년, 교원, 휴직)
        - REQUIREMENT: 자격, 요건
        """
        result = recognizer.recognize("연구년 자격 요건")

        entity_types = {m.entity_type for m in result.matches}

        assert EntityType.HYPERNYM in entity_types
        assert EntityType.REQUIREMENT in entity_types

        # Check hypernym expansion
        연구년_matches = [
            m
            for m in result.matches
            if m.text == "연구년" and m.entity_type == EntityType.HYPERNYM
        ]
        if 연구년_matches:
            assert "안식년" in 연구년_matches[0].expanded_terms

    def test_spec_scenario_3_ta_working_hours(
        self, recognizer: RegulationEntityRecognizer
    ):
        """
        SPEC Scenario 3: "조교 근무 시간"

        Expected entities:
        - HYPERNYM: 조교 (expands to 교육조교, 연구조교, 지원)
        """
        result = recognizer.recognize("조교 근무 시간")

        hypernym_matches = [
            m for m in result.matches if m.entity_type == EntityType.HYPERNYM
        ]

        # 조교 should trigger hypernym expansion
        조교_matches = [m for m in hypernym_matches if m.text == "조교"]
        if 조교_matches:
            # Check expansion includes related terms
            expanded = 조교_matches[0].expanded_terms
            assert (
                "교육조교" in expanded or "연구조교" in expanded or "지원" in expanded
            )

    # ============ Edge Case Tests ============

    def test_edge_case_overlapping_keywords(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test handling of keywords that belong to multiple entity types."""
        # "지원" appears in both BENEFIT and HYPERNYM expansions
        result = recognizer.recognize("장학금 지원")

        entity_types = {m.entity_type for m in result.matches}

        # Should recognize both types
        assert EntityType.BENEFIT in entity_types

    def test_edge_case_single_character_keywords(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test handling of very short keywords."""
        # Single character keywords may not be in our lists
        result = recognizer.recognize("A 학점")

        # Should not crash, may return empty or partial matches
        assert result is not None

    def test_edge_case_long_query(self, recognizer: RegulationEntityRecognizer):
        """Test handling of very long queries."""
        long_query = " ".join(["장학금"] * 100)
        result = recognizer.recognize(long_query)

        # Should handle gracefully
        assert result is not None
        assert result.original_query == long_query

    def test_edge_case_unicode_normalization(
        self, recognizer: RegulationEntityRecognizer
    ):
        """Test handling of Unicode normalization."""
        # Different representations of same text
        query1 = "휴학 방법"
        query2 = "휴학 方法"  # Mixed script (edge case)

        result1 = recognizer.recognize(query1)
        result2 = recognizer.recognize(query2)

        # Both should process without errors
        assert result1 is not None
        assert result2 is not None


class TestEntityMatchValidation:
    """Tests for EntityMatch dataclass validation."""

    def test_entity_match_confidence_validation(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            EntityMatch(
                entity_type=EntityType.BENEFIT,
                text="test",
                start=0,
                end=4,
                confidence=1.5,  # Invalid: > 1.0
                expanded_terms=[],
            )

        with pytest.raises(ValueError):
            EntityMatch(
                entity_type=EntityType.BENEFIT,
                text="test",
                start=0,
                end=4,
                confidence=-0.1,  # Invalid: < 0.0
                expanded_terms=[],
            )

    def test_entity_match_confidence_properties(self):
        """Test confidence property methods."""
        high_match = EntityMatch(
            entity_type=EntityType.BENEFIT,
            text="test",
            start=0,
            end=4,
            confidence=0.9,
            expanded_terms=[],
        )

        medium_match = EntityMatch(
            entity_type=EntityType.BENEFIT,
            text="test",
            start=0,
            end=4,
            confidence=0.7,
            expanded_terms=[],
        )

        assert high_match.is_high_confidence
        assert not medium_match.is_high_confidence
        assert medium_match.is_medium_confidence


class TestEntityRecognitionResultConstruction:
    """Tests for EntityRecognitionResult construction methods."""

    def test_from_matches_empty(self):
        """Test constructing result from empty match list."""
        result = EntityRecognitionResult.from_matches("test query", [])

        assert not result.has_entities
        assert len(result.matches) == 0
        assert result.primary_entity is None
        assert result.total_expanded_terms == []

    def test_from_matches_sorts_by_confidence(self):
        """Test that matches are sorted by confidence."""
        matches = [
            EntityMatch(
                entity_type=EntityType.BENEFIT,
                text="장학금",
                start=0,
                end=3,
                confidence=0.7,
                expanded_terms=["장학"],
            ),
            EntityMatch(
                entity_type=EntityType.SECTION,
                text="제15조",
                start=4,
                end=7,
                confidence=0.95,
                expanded_terms=[],
            ),
            EntityMatch(
                entity_type=EntityType.PROCEDURE,
                text="신청",
                start=8,
                end=10,
                confidence=0.85,
                expanded_terms=["신청서"],
            ),
        ]

        result = EntityRecognitionResult.from_matches("test", matches)

        # Should be sorted: SECTION (0.95) > PROCEDURE (0.85) > BENEFIT (0.7)
        assert result.matches[0].entity_type == EntityType.SECTION
        assert result.matches[1].entity_type == EntityType.PROCEDURE
        assert result.matches[2].entity_type == EntityType.BENEFIT

        # Primary entity should be highest confidence
        assert result.primary_entity.entity_type == EntityType.SECTION

    def test_from_matches_deduplicates_expanded_terms(self):
        """Test that expanded terms are deduplicated while preserving order."""
        matches = [
            EntityMatch(
                entity_type=EntityType.PROCEDURE,
                text="신청",
                start=0,
                end=2,
                confidence=0.9,
                expanded_terms=["신청", "신청서", "제출", "등록"],
            ),
            EntityMatch(
                entity_type=EntityType.PROCEDURE,
                text="서류",
                start=3,
                end=5,
                confidence=0.85,
                expanded_terms=["서류", "신청서", "제출서"],
            ),
        ]

        result = EntityRecognitionResult.from_matches("test", matches)

        # Check deduplication: "신청서" and "제출" appear in both
        expected_terms = ["신청", "신청서", "제출", "등록", "서류", "제출서"]
        assert result.total_expanded_terms == expected_terms
