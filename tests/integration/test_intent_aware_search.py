"""
SPEC-RAG-QUALITY-007: IntentClassifier Integration Tests

Purpose: Validate IntentClassifier integration and accuracy with search_usecase.

This test module verifies:
1. Intent classification accuracy (target: 90%+)
2. Intent-aware search parameter application
3. Edge cases (ambiguous queries, mixed intents)

Test Categories:
- test_intent_classification_accuracy: Measures accuracy on 100+ labeled queries
- test_procedure_intent_search_params: PROCEDURE -> top_k=15
- test_deadline_intent_search_params: DEADLINE -> boost_date
- test_eligibility_intent_search_params: ELIGIBILITY -> top_k=12
- test_edge_cases: Ambiguous queries, mixed intents
"""

import pytest

from src.rag.application.intent_classifier import (
    IntentCategory,
    IntentClassifier,
    IntentClassificationResult,
)
from src.rag.application.search_usecase import INTENT_SEARCH_CONFIGS


# =============================================================================
# Test Data: Labeled Queries Dataset (100+ queries)
# =============================================================================

# PROCEDURE queries: "how to", "method", "process" (어떻게, 방법, 신청, 절차)
PROCEDURE_QUERIES = [
    {"query": "휴학 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "복학 어떻게 해", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "등록금 납부 절차는", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "성적증명서 발급 어떡해", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "졸업 신청 방법 알려줘", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "전과 신청 어떻게 하나", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "재입학 절차가 어떻게 되나", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "학점 교환 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "자퇴 절차 알려주", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "휴학 어떻게해", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "복학 신청 어디서 해", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "등록금 분할 납부 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "성적 이의신청 절차", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "수강신청 변경 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "졸업요건 확인 어떡하", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "학적부 정정 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "부전공 신청 절차는", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "복수전공 변경 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "교환학생 지원 절차", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "장학금 신청 어디서", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "기숙사 입사 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "도서관 이용증 발급 절차", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "학생증 재발급 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "계절수업 신청 절차", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "성적 정정 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "군휴학 신청 어떻게", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "일반휴학 절차는", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "질병휴학 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "복학시 수강신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "졸업논문 제출 절차", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "학위수여식 참석 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "성적표 우송 신청", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "학적증명서 발급 절차", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "교직이수 신청 방법", "expected_intent": IntentCategory.PROCEDURE},
    {"query": "교원자격시험 응시 절차", "expected_intent": IntentCategory.PROCEDURE},
]

# ELIGIBILITY queries: "can I", "qualification", "condition" (받을 수 있, 가능해, 자격, 조건)
ELIGIBILITY_QUERIES = [
    {"query": "장학금 받을 수 있나", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "복학 가능해", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "전과 자격이 있나", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "재입학 조건이 뭐야", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "등록금 감면 받을 수 있", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "졸업 가능한가요", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "장학금 자격이 되나", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "부전공 할 수 있나", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "교환학생 가능한가", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "기숙사 입사 자격", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "학점 교환 할 수 있나요", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "졸업유예 가능해", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "성적 향상 장학금 받을수있", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "근로장학생 될까", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "복수전공 자격이", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "계절수업 수강 조건이", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "휴학 연장 가능한가요", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "성적 이의신청 할 수 있나", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "조기졸업 자격 요건", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "교직이수 받나요", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "누가 장학금 받을 수 있", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "등록금 분할 납부 조건", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "학사경고 면제 가능해", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "성적 우수 장학금 자격", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "군복학 지원 받을 수 있나", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "저소득층 장학금 가능한", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "성적 인정 조건이 뭐야", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "외국인 학생 장학금 되나요", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "편입학 자격이 되려면어", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "학점 초과 수강 할 수 있", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "다전공 신청 자격 요건", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "봉사장학금 받을 수 있나요", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "성적 경고 학생 수강 가능", "expected_intent": IntentCategory.ELIGIBILITY},
    {"query": "졸업요건 충족 조건", "expected_intent": IntentCategory.ELIGIBILITY},
]

# DEADLINE queries: "when", "period", "deadline" (언제까지, 기간, 마감, 언제)
# Note: Queries with strong PROCEDURE keywords (신청, 정정) may be classified as PROCEDURE
# These queries focus primarily on time/date-related questions
DEADLINE_QUERIES = [
    {"query": "휴학 신청 언제까지", "expected_intent": IntentCategory.DEADLINE},
    {"query": "등록금 납부 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "수강신청 마감일", "expected_intent": IntentCategory.DEADLINE},
    {"query": "복학 신청 언제까지야", "expected_intent": IntentCategory.DEADLINE},
    {"query": "졸업 신청 기한", "expected_intent": IntentCategory.DEADLINE},
    {"query": "장학금 신청 마감", "expected_intent": IntentCategory.DEADLINE},
    {"query": "학점 교환 마감일", "expected_intent": IntentCategory.DEADLINE},
    {"query": "기숙사 입사 신청 언제부터", "expected_intent": IntentCategory.DEADLINE},
    {"query": "등록금 분할 납부 기한", "expected_intent": IntentCategory.DEADLINE},
    {"query": "졸업논문 제출 마감", "expected_intent": IntentCategory.DEADLINE},
    {"query": "교환학생 지원 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "재입학 신청 언제까지", "expected_intent": IntentCategory.DEADLINE},
    {"query": "교직이수 신청 기한", "expected_intent": IntentCategory.DEADLINE},
    {"query": "계절수업 수강신청 마감일", "expected_intent": IntentCategory.DEADLINE},
    {"query": "복수전공 변경 마감", "expected_intent": IntentCategory.DEADLINE},
    {"query": "졸업요건 확인 언제까지", "expected_intent": IntentCategory.DEADLINE},
    {"query": "성적표 발급 마감일", "expected_intent": IntentCategory.DEADLINE},
    {"query": "학기 시작 언제", "expected_intent": IntentCategory.DEADLINE},
    {"query": "시험 기간 며칠간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "방학 기간이 언제부터", "expected_intent": IntentCategory.DEADLINE},
    {"query": "개강일 언제", "expected_intent": IntentCategory.DEADLINE},
    {"query": "종강일 마감", "expected_intent": IntentCategory.DEADLINE},
    {"query": "성적 공시 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "학사 일정 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "등록 기간 마감일", "expected_intent": IntentCategory.DEADLINE},
    {"query": "휴가 기간 언제까지", "expected_intent": IntentCategory.DEADLINE},
    {"query": "등록금 납부 마감일", "expected_intent": IntentCategory.DEADLINE},
    {"query": "수강 철회 기한", "expected_intent": IntentCategory.DEADLINE},
    {"query": "성적 처리 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "학기 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "등록 기간", "expected_intent": IntentCategory.DEADLINE},
    {"query": "시험 일정 언제", "expected_intent": IntentCategory.DEADLINE},
    {"query": "방학 언제까지", "expected_intent": IntentCategory.DEADLINE},
    {"query": "개학 기간", "expected_intent": IntentCategory.DEADLINE},
]

# GENERAL queries: ambiguous or general queries (default)
GENERAL_QUERIES = [
    {"query": "규정이 뭐야", "expected_intent": IntentCategory.GENERAL},
    {"query": "학칙", "expected_intent": IntentCategory.GENERAL},
    {"query": "교원인사규정 내용", "expected_intent": IntentCategory.GENERAL},
    {"query": "학사관련 규정", "expected_intent": IntentCategory.GENERAL},
    {"query": "장학규정 요약", "expected_intent": IntentCategory.GENERAL},
    {"query": "기숙사 규정", "expected_intent": IntentCategory.GENERAL},
    {"query": "등록에 관한 규정", "expected_intent": IntentCategory.GENERAL},
    {"query": "수업 규정 설명", "expected_intent": IntentCategory.GENERAL},
    {"query": "학점 제도", "expected_intent": IntentCategory.GENERAL},
    {"query": "성적 평가 방식", "expected_intent": IntentCategory.GENERAL},
]

# Edge cases: ambiguous queries, mixed intents
EDGE_CASE_QUERIES = [
    # Mixed intent: both PROCEDURE and DEADLINE keywords
    {"query": "휴학 신청 언제까지 어떻게 해", "expected_intent": IntentCategory.PROCEDURE, "note": "Mixed PROCEDURE+DEADLINE"},
    # Mixed intent: both ELIGIBILITY and DEADLINE keywords
    {"query": "장학금 자격이고 신청 기간 언제", "expected_intent": IntentCategory.DEADLINE, "note": "Mixed ELIGIBILITY+DEADLINE"},
    # Ambiguous: could be PROCEDURE or ELIGIBILITY
    {"query": "졸업 관련해서", "expected_intent": IntentCategory.GENERAL, "note": "Ambiguous - needs context"},
    # Colloquial ending
    {"query": "휴학 하고 싶은데 해야 되나요", "expected_intent": IntentCategory.PROCEDURE, "note": "Colloquial PROCEDURE"},
    # Multiple procedure keywords
    {"query": "어떻게 신청하는 방법 절차 알려줘", "expected_intent": IntentCategory.PROCEDURE, "note": "Multiple PROCEDURE keywords"},
    # Short query with strong signal
    {"query": "언제까지", "expected_intent": IntentCategory.DEADLINE, "note": "Short but strong DEADLINE signal"},
    # Query with no clear keywords
    {"query": "안녕하세요", "expected_intent": IntentCategory.GENERAL, "note": "No keywords"},
    # Empty-like query
    {"query": "   ", "expected_intent": IntentCategory.GENERAL, "note": "Whitespace only"},
    # Query with numbers and dates
    {"query": "2024년 1학기 등록금 납부 기간", "expected_intent": IntentCategory.DEADLINE, "note": "Date + DEADLINE keyword"},
    # Complex query
    {"query": "성적이 낮은데 장학금 받을 수 있을까 방법이 있나", "expected_intent": IntentCategory.ELIGIBILITY, "note": "Complex ELIGIBILITY"},
    # Negative query
    {"query": "휴학 안 하고 싶은데", "expected_intent": IntentCategory.PROCEDURE, "note": "Negative PROCEDURE"},
    # Query about condition with procedure
    {"query": "장학금 자격 조건과 신청 방법", "expected_intent": IntentCategory.ELIGIBILITY, "note": "ELIGIBILITY + PROCEDURE"},
    # Deadline with procedure elements
    {"query": "언제까지 신청해야 해", "expected_intent": IntentCategory.DEADLINE, "note": "DEADLINE + PROCEDURE"},
]

# Combine all queries into the full dataset
LABELED_QUERIES = (
    PROCEDURE_QUERIES
    + ELIGIBILITY_QUERIES
    + DEADLINE_QUERIES
    + GENERAL_QUERIES
    + EDGE_CASE_QUERIES
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def intent_classifier() -> IntentClassifier:
    """Create IntentClassifier instance for testing."""
    return IntentClassifier(confidence_threshold=0.5)


@pytest.fixture
def labeled_queries():
    """Return the labeled queries dataset."""
    return LABELED_QUERIES


# =============================================================================
# Test: Intent Classification Accuracy
# =============================================================================


@pytest.mark.integration
class TestIntentClassificationAccuracy:
    """Test IntentClassifier accuracy on labeled queries."""

    def test_intent_classification_accuracy_overall(
        self, intent_classifier: IntentClassifier, labeled_queries
    ):
        """
        Test overall intent classification accuracy.

        REQUIREMENT: 90%+ accuracy on labeled queries (SPEC-RAG-QUALITY-007 REQ-002)
        """
        correct = 0
        total = len(labeled_queries)
        misclassifications = []

        for item in labeled_queries:
            result = intent_classifier.classify(item["query"])
            if result.category == item["expected_intent"]:
                correct += 1
            else:
                misclassifications.append({
                    "query": item["query"],
                    "expected": item["expected_intent"].value,
                    "actual": result.category.value,
                    "confidence": result.confidence,
                    "note": item.get("note", ""),
                })

        accuracy = correct / total

        # Print detailed results for debugging
        if misclassifications:
            print(f"\n=== Misclassifications ({len(misclassifications)} queries) ===")
            for m in misclassifications[:10]:  # Show first 10
                print(f"  Query: '{m['query']}'")
                print(f"    Expected: {m['expected']}, Got: {m['actual']} (conf: {m['confidence']:.2f})")
                if m["note"]:
                    print(f"    Note: {m['note']}")

        assert accuracy >= 0.90, (
            f"Intent classification accuracy {accuracy:.2%} is below 90% threshold. "
            f"Correct: {correct}/{total}, Misclassified: {len(misclassifications)}"
        )

    def test_procedure_classification_accuracy(
        self, intent_classifier: IntentClassifier
    ):
        """Test PROCEDURE intent classification accuracy (target: 90%+)."""
        correct = 0
        total = len(PROCEDURE_QUERIES)

        for item in PROCEDURE_QUERIES:
            result = intent_classifier.classify(item["query"])
            if result.category == item["expected_intent"]:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"PROCEDURE accuracy {accuracy:.2%} below 90%"

    def test_eligibility_classification_accuracy(
        self, intent_classifier: IntentClassifier
    ):
        """Test ELIGIBILITY intent classification accuracy (target: 90%+)."""
        correct = 0
        total = len(ELIGIBILITY_QUERIES)

        for item in ELIGIBILITY_QUERIES:
            result = intent_classifier.classify(item["query"])
            if result.category == item["expected_intent"]:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"ELIGIBILITY accuracy {accuracy:.2%} below 90%"

    def test_deadline_classification_accuracy(
        self, intent_classifier: IntentClassifier
    ):
        """Test DEADLINE intent classification accuracy (target: 90%+)."""
        correct = 0
        total = len(DEADLINE_QUERIES)

        for item in DEADLINE_QUERIES:
            result = intent_classifier.classify(item["query"])
            if result.category == item["expected_intent"]:
                correct += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"DEADLINE accuracy {accuracy:.2%} below 90%"


# =============================================================================
# Test: Intent-Aware Search Parameters
# =============================================================================


@pytest.mark.integration
class TestIntentAwareSearchParams:
    """Test intent-specific search parameter application."""

    def test_procedure_intent_search_params(self):
        """
        Test that PROCEDURE queries get top_k=15.

        REQUIREMENT: PROCEDURE -> top_k=15, boost_procedure=1.5
        """
        config = INTENT_SEARCH_CONFIGS.get(IntentCategory.PROCEDURE, {})

        assert config.get("top_k") == 15, (
            f"PROCEDURE top_k should be 15, got {config.get('top_k')}"
        )
        assert config.get("boost_procedure") == 1.5, (
            f"PROCEDURE boost_procedure should be 1.5, got {config.get('boost_procedure')}"
        )

    def test_deadline_intent_search_params(self):
        """
        Test that DEADLINE queries get boost_date parameter.

        REQUIREMENT: DEADLINE -> top_k=10, boost_date=1.3
        """
        config = INTENT_SEARCH_CONFIGS.get(IntentCategory.DEADLINE, {})

        assert config.get("top_k") == 10, (
            f"DEADLINE top_k should be 10, got {config.get('top_k')}"
        )
        assert config.get("boost_date") == 1.3, (
            f"DEADLINE boost_date should be 1.3, got {config.get('boost_date')}"
        )

    def test_eligibility_intent_search_params(self):
        """
        Test that ELIGIBILITY queries get top_k=12.

        REQUIREMENT: ELIGIBILITY -> top_k=12, boost_condition=1.4
        """
        config = INTENT_SEARCH_CONFIGS.get(IntentCategory.ELIGIBILITY, {})

        assert config.get("top_k") == 12, (
            f"ELIGIBILITY top_k should be 12, got {config.get('top_k')}"
        )
        assert config.get("boost_condition") == 1.4, (
            f"ELIGIBILITY boost_condition should be 1.4, got {config.get('boost_condition')}"
        )

    def test_general_intent_search_params(self):
        """
        Test that GENERAL queries get default top_k=10.

        REQUIREMENT: GENERAL -> top_k=10 (default)
        """
        config = INTENT_SEARCH_CONFIGS.get(IntentCategory.GENERAL, {})

        assert config.get("top_k") == 10, (
            f"GENERAL top_k should be 10, got {config.get('top_k')}"
        )

    def test_all_intent_categories_have_configs(self):
        """Verify all IntentCategory values have corresponding search configs."""
        for category in IntentCategory:
            assert category in INTENT_SEARCH_CONFIGS, (
                f"Missing search config for IntentCategory.{category.value}"
            )


# =============================================================================
# Test: Edge Cases
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases: ambiguous queries, mixed intents, boundary conditions."""

    def test_empty_query_handling(self, intent_classifier: IntentClassifier):
        """Test that empty queries return GENERAL with low confidence."""
        result = intent_classifier.classify("")
        assert result.category == IntentCategory.GENERAL
        assert result.confidence == 0.0

    def test_whitespace_only_query(self, intent_classifier: IntentClassifier):
        """Test that whitespace-only queries return GENERAL with low confidence."""
        result = intent_classifier.classify("   ")
        assert result.category == IntentCategory.GENERAL
        assert result.confidence == 0.0

    def test_none_query_handling(self, intent_classifier: IntentClassifier):
        """Test that None queries return GENERAL with zero confidence."""
        result = intent_classifier.classify(None)
        assert result.category == IntentCategory.GENERAL
        assert result.confidence == 0.0

    def test_very_long_query(self, intent_classifier: IntentClassifier):
        """Test handling of very long queries."""
        long_query = "휴학 신청 방법 " * 100
        result = intent_classifier.classify(long_query)
        # Should still classify correctly despite length
        assert result.category in [IntentCategory.PROCEDURE, IntentCategory.GENERAL]

    def test_special_characters_query(self, intent_classifier: IntentClassifier):
        """Test handling of queries with special characters."""
        result = intent_classifier.classify("휴학@신청#방법!!!")
        # Should handle special characters gracefully
        assert isinstance(result, IntentClassificationResult)

    def test_mixed_language_query(self, intent_classifier: IntentClassifier):
        """Test handling of mixed Korean-English queries."""
        result = intent_classifier.classify("휴학 신청 방법 how to apply")
        assert isinstance(result, IntentClassificationResult)

    def test_confidence_threshold_boundary(
        self, intent_classifier: IntentClassifier
    ):
        """Test that confidence threshold correctly distinguishes GENERAL."""
        # Queries with very weak signals should be classified as GENERAL
        weak_query = "학교"
        result = intent_classifier.classify(weak_query)
        # Should either be GENERAL or have low confidence
        if result.category != IntentCategory.GENERAL:
            assert result.confidence >= intent_classifier.confidence_threshold

    def test_batch_classification(self, intent_classifier: IntentClassifier):
        """Test batch classification performance."""
        queries = [item["query"] for item in LABELED_QUERIES[:20]]
        results = intent_classifier.classify_batch(queries)

        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, IntentClassificationResult)

    def test_matched_keywords_populated(self, intent_classifier: IntentClassifier):
        """Test that matched_keywords are populated for non-GENERAL queries."""
        result = intent_classifier.classify("휴학 신청 방법")
        assert result.category == IntentCategory.PROCEDURE
        assert len(result.matched_keywords) > 0
        assert any("신청" in kw or "방법" in kw for kw in result.matched_keywords)


# =============================================================================
# Test: Intent-Aware Search Integration
# =============================================================================


@pytest.mark.integration
class TestIntentAwareSearchIntegration:
    """Test IntentClassifier integration with search_usecase."""

    def test_intent_classifier_used_in_search_config(self):
        """Verify IntentClassifier is properly integrated with search configs."""
        # This test verifies the integration exists
        # The actual search_usecase tests would be in a separate integration test

        classifier = IntentClassifier()

        # Classify sample queries
        procedure_result = classifier.classify("휴학 신청 방법")
        deadline_result = classifier.classify("등록금 납부 기간")
        eligibility_result = classifier.classify("장학금 자격이 되나")

        # Verify search configs exist for each category
        assert procedure_result.category in INTENT_SEARCH_CONFIGS
        assert deadline_result.category in INTENT_SEARCH_CONFIGS
        assert eligibility_result.category in INTENT_SEARCH_CONFIGS

        # Verify configs have required keys
        for category, config in INTENT_SEARCH_CONFIGS.items():
            assert "top_k" in config, f"Missing top_k in {category.value} config"

    def test_search_config_top_k_values_reasonable(self):
        """Verify top_k values are within reasonable bounds."""
        for category, config in INTENT_SEARCH_CONFIGS.items():
            top_k = config.get("top_k", 0)
            assert 5 <= top_k <= 30, (
                f"top_k={top_k} for {category.value} seems unreasonable. "
                "Expected value between 5 and 30."
            )

    def test_search_config_boost_values_reasonable(self):
        """Verify boost values are within reasonable bounds."""
        for category, config in INTENT_SEARCH_CONFIGS.items():
            for key, value in config.items():
                if key.startswith("boost_"):
                    assert 1.0 <= value <= 2.0, (
                        f"boost value {value} for {category.value}.{key} "
                        "seems unreasonable. Expected value between 1.0 and 2.0."
                    )


# =============================================================================
# Parametrized Tests for Coverage
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,expected_category",
    [
        # PROCEDURE cases
        ("휴학 신청 방법", IntentCategory.PROCEDURE),
        ("복학 어떻게 해", IntentCategory.PROCEDURE),
        ("등록금 납부 절차", IntentCategory.PROCEDURE),
        ("성적증명서 발급 방법", IntentCategory.PROCEDURE),
        # ELIGIBILITY cases
        ("장학금 받을 수 있나", IntentCategory.ELIGIBILITY),
        ("복학 가능해", IntentCategory.ELIGIBILITY),
        ("전과 자격이", IntentCategory.ELIGIBILITY),
        # DEADLINE cases
        ("휴학 신청 언제까지", IntentCategory.DEADLINE),
        ("등록금 납부 기간", IntentCategory.DEADLINE),
        ("수강신청 마감", IntentCategory.DEADLINE),
    ],
)
def test_intent_classification_parametrized(
    intent_classifier: IntentClassifier, query: str, expected_category: IntentCategory
):
    """Parametrized test for various query types."""
    result = intent_classifier.classify(query)
    assert result.category == expected_category, (
        f"Query '{query}' classified as {result.category.value}, "
        f"expected {expected_category.value}"
    )
