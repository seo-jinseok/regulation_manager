"""
Tests for IntentClassifier.

Follows TDD approach: Tests are written FIRST to define expected behavior.
"""

import pytest

from src.rag.application.intent_classifier import (
    IntentCategory,
    IntentClassificationResult,
    IntentClassifier,
)


class TestIntentCategory:
    """Tests for IntentCategory enum."""

    def test_intent_categories_exist(self):
        """Test that all required intent categories exist."""
        assert hasattr(IntentCategory, "PROCEDURE")
        assert hasattr(IntentCategory, "ELIGIBILITY")
        assert hasattr(IntentCategory, "DEADLINE")
        assert hasattr(IntentCategory, "GENERAL")

    def test_intent_category_values(self):
        """Test that intent categories have correct Korean names."""
        assert IntentCategory.PROCEDURE.value == "PROCEDURE"
        assert IntentCategory.ELIGIBILITY.value == "ELIGIBILITY"
        assert IntentCategory.DEADLINE.value == "DEADLINE"
        assert IntentCategory.GENERAL.value == "GENERAL"


class TestIntentClassificationResult:
    """Tests for IntentClassificationResult dataclass."""

    def test_result_creation(self):
        """Test creating IntentClassificationResult."""
        result = IntentClassificationResult(
            category=IntentCategory.PROCEDURE,
            confidence=0.85,
            matched_keywords=["어떻게", "방법"],
        )
        assert result.category == IntentCategory.PROCEDURE
        assert result.confidence == 0.85
        assert result.matched_keywords == ["어떻게", "방법"]

    def test_result_default_values(self):
        """Test IntentClassificationResult default values."""
        result = IntentClassificationResult(
            category=IntentCategory.GENERAL,
            confidence=0.5,
        )
        assert result.matched_keywords == []


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create IntentClassifier instance."""
        return IntentClassifier()

    # ==================== PROCEDURE Intent Tests ====================

    def test_procedure_intent_how_to(self, classifier):
        """Test detecting procedure intent with '어떻게' keyword."""
        result = classifier.classify("휴학을 어떻게 신청하나요?")
        assert result.category == IntentCategory.PROCEDURE
        assert "어떻게" in result.matched_keywords
        assert result.confidence > 0.5

    def test_procedure_intent_method(self, classifier):
        """Test detecting procedure intent with '방법' keyword."""
        result = classifier.classify("등록금 납부 방법을 알려주세요.")
        assert result.category == IntentCategory.PROCEDURE
        assert "방법" in result.matched_keywords
        assert result.confidence > 0.5

    def test_procedure_intent_apply(self, classifier):
        """Test detecting procedure intent with '신청' keyword."""
        result = classifier.classify("장학금 신청은 어디서 하나요?")
        assert result.category == IntentCategory.PROCEDURE
        assert "신청" in result.matched_keywords
        assert result.confidence > 0.5

    def test_procedure_intent_process(self, classifier):
        """Test detecting procedure intent with '절차' keyword."""
        result = classifier.classify("졸업 절차가 어떻게 되나요?")
        assert result.category == IntentCategory.PROCEDURE
        assert "절차" in result.matched_keywords
        assert result.confidence > 0.5

    def test_procedure_intent_colloquial_geonaeyo(self, classifier):
        """Test detecting procedure intent with colloquial '게요' ending."""
        result = classifier.classify("휴학할건데 어떻게 해요?")
        assert result.category == IntentCategory.PROCEDURE

    # ==================== ELIGIBILITY Intent Tests ====================

    def test_eligibility_intent_can_receive(self, classifier):
        """Test detecting eligibility intent with '받을 수 있어' pattern."""
        result = classifier.classify("장학금을 받을 수 있어?")
        assert result.category == IntentCategory.ELIGIBILITY
        assert result.confidence > 0.5

    def test_eligibility_intent_possible(self, classifier):
        """Test detecting eligibility intent with '가능해' keyword."""
        result = classifier.classify("휴학이 가능해?")
        assert result.category == IntentCategory.ELIGIBILITY
        # Matches "가능해" which is a variant containing "가능"
        assert any("가능" in kw for kw in result.matched_keywords)
        assert result.confidence > 0.5

    def test_eligibility_intent_qualification(self, classifier):
        """Test detecting eligibility intent with '자격' keyword."""
        result = classifier.classify("장학금 자격이 뭐야?")
        assert result.category == IntentCategory.ELIGIBILITY
        assert "자격" in result.matched_keywords
        assert result.confidence > 0.5

    def test_eligibility_intent_condition(self, classifier):
        """Test detecting eligibility intent with '조건' keyword."""
        result = classifier.classify("등록금 분납 조건이 어떻게 되나요?")
        assert result.category == IntentCategory.ELIGIBILITY
        assert "조건" in result.matched_keywords
        assert result.confidence > 0.5

    def test_eligibility_intent_can_i(self, classifier):
        """Test detecting eligibility intent with '할 수 있나요' pattern."""
        result = classifier.classify("제가 이 수업을 들을 수 있나요?")
        # "을 수 있" matches ELIGIBILITY pattern
        assert result.category == IntentCategory.ELIGIBILITY
        assert result.confidence > 0.5

    # ==================== DEADLINE Intent Tests ====================

    def test_deadline_intent_until_when(self, classifier):
        """Test detecting deadline intent with '언제까지' keyword."""
        result = classifier.classify("휴학 신청 언제까지야?")
        assert result.category == IntentCategory.DEADLINE
        assert "언제까지" in result.matched_keywords
        assert result.confidence > 0.5

    def test_deadline_intent_period(self, classifier):
        """Test detecting deadline intent with '기간' keyword."""
        result = classifier.classify("수강신청 기간이 언제인가요?")
        assert result.category == IntentCategory.DEADLINE
        assert "기간" in result.matched_keywords
        assert result.confidence > 0.5

    def test_deadline_intent_deadline(self, classifier):
        """Test detecting deadline intent with '마감' keyword."""
        result = classifier.classify("마감일이 언제야?")
        assert result.category == IntentCategory.DEADLINE
        assert "마감" in result.matched_keywords
        assert result.confidence > 0.5

    def test_deadline_intent_when(self, classifier):
        """Test detecting deadline intent with '언제' keyword."""
        result = classifier.classify("등록금 납부는 언제 하나요?")
        assert result.category == IntentCategory.DEADLINE
        assert "언제" in result.matched_keywords
        assert result.confidence > 0.5

    def test_deadline_intent_deadline_synonym(self, classifier):
        """Test detecting deadline intent with '기한' keyword."""
        result = classifier.classify("신청 기한이 지났나요?")
        assert result.category == IntentCategory.DEADLINE
        assert "기한" in result.matched_keywords
        assert result.confidence > 0.5

    # ==================== GENERAL Intent Tests ====================

    def test_general_intent_simple_question(self, classifier):
        """Test detecting general intent for simple question."""
        # Note: "알려주세요" triggers PROCEDURE intent because it's action-oriented
        # Use a truly general query without procedure keywords
        result = classifier.classify("학칙이란 무엇인가요?")
        assert result.category == IntentCategory.GENERAL

    def test_general_intent_definition(self, classifier):
        """Test detecting general intent for definition query."""
        result = classifier.classify("학사경고가 뭐야?")
        assert result.category == IntentCategory.GENERAL

    def test_general_intent_comparison(self, classifier):
        """Test detecting general intent for comparison query."""
        result = classifier.classify("일반휴학과 병결휴학의 차이점은?")
        assert result.category == IntentCategory.GENERAL

    # ==================== Edge Cases ====================

    def test_empty_query(self, classifier):
        """Test handling empty query."""
        result = classifier.classify("")
        assert result.category == IntentCategory.GENERAL
        assert result.confidence == 0.0

    def test_whitespace_only_query(self, classifier):
        """Test handling whitespace-only query."""
        result = classifier.classify("   ")
        assert result.category == IntentCategory.GENERAL
        assert result.confidence == 0.0

    def test_multiple_keywords_same_category(self, classifier):
        """Test query with multiple keywords from same category."""
        result = classifier.classify("휴학 신청 방법과 절차를 알려주세요.")
        assert result.category == IntentCategory.PROCEDURE
        assert len(result.matched_keywords) >= 2
        assert result.confidence > 0.7  # Higher confidence with more keywords

    def test_multiple_keywords_different_categories(self, classifier):
        """Test query with keywords from different categories."""
        # "언제까지" (DEADLINE) + "방법" (PROCEDURE)
        result = classifier.classify("휴학 신청 방법과 언제까지 해야 되나요?")
        # Should pick the one with stronger signal (first/earlier or more specific)
        assert result.category in [IntentCategory.PROCEDURE, IntentCategory.DEADLINE]
        assert result.confidence > 0.5

    def test_colloquial_expression_haeya(self, classifier):
        """Test colloquial expression '해야' (have to)."""
        result = classifier.classify("휴학하려면 뭘 해야 해?")
        assert result.category == IntentCategory.PROCEDURE

    def test_colloquial_expression_jwoyo(self, classifier):
        """Test colloquial expression '주세요' (please give me)."""
        result = classifier.classify("등록금 납부 안내해 주세요.")
        assert result.category == IntentCategory.PROCEDURE

    # ==================== Confidence Scoring ====================

    def test_confidence_increases_with_more_keywords(self, classifier):
        """Test that confidence increases with more matching keywords."""
        result_one = classifier.classify("휴학 방법")
        result_two = classifier.classify("휴학 신청 방법과 절차를 알려주세요")
        assert result_two.confidence > result_one.confidence

    def test_low_confidence_for_ambiguous_query(self, classifier):
        """Test low confidence for ambiguous queries."""
        result = classifier.classify("학교")
        assert result.confidence < 0.5

    # ==================== Integration-like Tests ====================

    def test_real_world_procedure_query(self, classifier):
        """Test real-world procedure query."""
        # Focus on procedure-specific query without conflicting keywords
        result = classifier.classify(
            "휴학 신청 방법과 절차를 알려주세요."
        )
        assert result.category == IntentCategory.PROCEDURE

    def test_real_world_eligibility_query(self, classifier):
        """Test real-world eligibility query."""
        result = classifier.classify(
            "제가 이번 학기 성적이 좋지 않은데 장학금을 받을 수 있을까요?"
        )
        assert result.category == IntentCategory.ELIGIBILITY

    def test_real_world_deadline_query(self, classifier):
        """Test real-world deadline query."""
        # Focus on deadline-specific query
        result = classifier.classify(
            "수강신청 기간이 언제까지인가요?"
        )
        assert result.category == IntentCategory.DEADLINE
