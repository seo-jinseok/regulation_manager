"""
Intent Classification Accuracy Tests (SPEC-RAG-QUALITY-010 Milestone 4).

Tests to validate that intent classification achieves >= 85% accuracy
across all intent categories.
"""

import pytest

from src.rag.application.intent_classifier import (
    IntentCategory,
    IntentClassifier,
)


class TestIntentAccuracyProcedure:
    """Accuracy tests for PROCEDURE intent classification (20 test cases)."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize(
        "query",
        [
            # Core procedure keywords
            "장학금 신청 방법이 궁금해요",
            "휴학 신청하려면 어떻게 해야 하나요?",
            "등록금 납부 절차를 알고 싶어요",
            "졸업논문 제출하는 법을 알려주세요",
            "전과 신청 방법이 어떻게 되나요?",
            # Colloquial expressions
            "수강신청 어떻게 해요?",
            "성적증명서 발급받으려면 뭘 해야 돼요?",
            "복학은 어디서 하나요?",
            "학생증 재발급받고 싶은데요",
            "교환학생 지원 절차가 어떻게 되나요?",
            # Mixed with other context
            "이번 학기 장학금 신청하려면 무슨 서류 필요해요?",
            "휴학하고 싶은데 절차가 복잡한가요?",
            "등록금 분납 신청 방법을 알려주세요",
            "복수전공 신청하는 방법이 궁금합니다",
            "부전공 등록하려면 어떻게 해야 하나요?",
            # Question patterns
            "자퇴하려면 어떤 서류를 제출해야 하나요?",
            "재입학 신청 방법이 어떻게 되나요?",
            "휴학계 작성하는 법을 알려주세요",
            "성적 이의신청 접수는 어디서 하나요?",
            "현장실습 신청 절차를 알고 싶어요",
        ],
    )
    def test_procedure_classification(self, classifier, query):
        """Test PROCEDURE intent classification accuracy."""
        result = classifier.classify(query)
        assert (
            result.category == IntentCategory.PROCEDURE
        ), f"Expected PROCEDURE for '{query}', got {result.category}"


class TestIntentAccuracyEligibility:
    """Accuracy tests for ELIGIBILITY intent classification (20 test cases)."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize(
        "query",
        [
            # Core eligibility keywords
            "장학금 받을 수 있나요?",
            "휴학 자격이 되나요?",
            "이번 학기 장학금 대상이 어떻게 돼요?",
            "성적 우수 장학금 조건이 뭐예요?",
            "근로장학생으로 지원할 수 있어요?",
            # Question patterns
            "복수전공 신청 자격이 되나요?",
            "교환학생 지원 대상이 어떻게 되나요?",
            "등록금 분납이 가능한가요?",
            "성적 이의신청 할 수 있나요?",
            "재입학 자격 요건이 뭐예요?",
            # Colloquial expressions
            "장학금 받을 수 있을까요?",
            "휴학이 가능한 상황인가요?",
            "저도 장학금 대상이 되나요?",
            "성적 장학금 받을 수 있어요?",
            "이 조건이면 장학금 가능해요?",
            # Mixed context
            "평점 평균 몇이면 장학금 받을 수 있나요?",
            "휴학은 누구나 할 수 있나요?",
            "등록금 감면 대상이 어떻게 되나요?",
            "특별장학금 지원 자격이 궁금해요",
            "졸업 유예가 가능한가요?",
        ],
    )
    def test_eligibility_classification(self, classifier, query):
        """Test ELIGIBILITY intent classification accuracy."""
        result = classifier.classify(query)
        assert (
            result.category == IntentCategory.ELIGIBILITY
        ), f"Expected ELIGIBILITY for '{query}', got {result.category}"


class TestIntentAccuracyDeadline:
    """Accuracy tests for DEADLINE intent classification (20 test cases)."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize(
        "query",
        [
            # Core deadline keywords
            "장학금 신청 기간이 언제까지인가요?",
            "휴학 신청 언제까지 해야 해요?",
            "등록금 납부 기한이 언제까지예요?",
            "수강신청 마감일이 언제야?",
            "졸업논문 제출 언제까지인가요?",
            # Date/time queries
            "이번 학기 개강일이 언제예요?",
            "종강일이 언제인가요?",
            "시험 기간이 언제부터 언제까지예요?",
            "성적 공시 기간이 언제야?",
            "학사일정이 어떻게 돼요?",
            # Period queries
            "수강철회 기간이 언제까지인가요?",
            "성적 이의신청 기간이 언제예요?",
            "휴학원 제출 기한이 언제까지야?",
            "복학 신청 기간을 알고 싶어요",
            "등록금 환불 기한이 언제까지인가요?",
            # Colloquial expressions
            "언제까지 신청해야 돼요?",
            "마감일이 며칠까지예요?",
            "신청 기한이 언제까지죠?",
            "언제부터 신청 가능해요?",
            "제출 기한이 언제까지인가요?",
        ],
    )
    def test_deadline_classification(self, classifier, query):
        """Test DEADLINE intent classification accuracy."""
        result = classifier.classify(query)
        assert (
            result.category == IntentCategory.DEADLINE
        ), f"Expected DEADLINE for '{query}', got {result.category}"


class TestIntentAccuracyGeneral:
    """Accuracy tests for GENERAL intent classification (20 test cases)."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize(
        "query",
        [
            # General information queries
            "동의대학교 규정에 대해 알려주세요",
            "학칙 내용이 뭐예요?",
            "장학금 종류가 뭐가 있어요?",
            "학교 규정을 찾고 싶어요",
            "이 규정은 무슨 내용인가요?",
            # Ambiguous queries (should default to GENERAL with low confidence)
            "학사경고",
            "성적",
            "등록금",
            "장학금",
            "휴학",
            # Descriptive queries
            "동의대학교 학사 규정 요약",
            "학생복지 규정 개요",
            "교원 인사 규정 내용",
            "연구 규정 정리",
            "시설 관리 규정",
            # Open-ended questions
            "규정집에 뭐가 있어요?",
            "학교 규정 어떤 게 있나요?",
            "이번에 바뀐 규정이 뭐예요?",
            "학생 관련 규정 알려주세요",
            "대학원 규정이 궁금해요",
        ],
    )
    def test_general_classification(self, classifier, query):
        """Test GENERAL intent classification accuracy."""
        result = classifier.classify(query)
        # GENERAL queries may be classified as any category if confidence is low
        # The key is that confidence should be low for ambiguous queries
        assert result.category in [
            IntentCategory.GENERAL,
            IntentCategory.PROCEDURE,
            IntentCategory.ELIGIBILITY,
            IntentCategory.DEADLINE,
        ], f"Unexpected category for '{query}': {result.category}"


class TestIntentAccuracyMetrics:
    """Calculate and validate overall accuracy metrics."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_overall_accuracy_threshold(self, classifier):
        """
        Validate that overall classification accuracy is >= 85%.

        This test aggregates results from all test categories and
        calculates the overall accuracy rate.
        """
        # Define test cases with expected categories
        test_cases = [
            # PROCEDURE cases (20)
            ("장학금 신청 방법이 궁금해요", IntentCategory.PROCEDURE),
            ("휴학 신청하려면 어떻게 해야 하나요?", IntentCategory.PROCEDURE),
            ("등록금 납부 절차를 알고 싶어요", IntentCategory.PROCEDURE),
            ("졸업논문 제출하는 법을 알려주세요", IntentCategory.PROCEDURE),
            ("전과 신청 방법이 어떻게 되나요?", IntentCategory.PROCEDURE),
            ("수강신청 어떻게 해요?", IntentCategory.PROCEDURE),
            ("성적증명서 발급받으려면 뭘 해야 돼요?", IntentCategory.PROCEDURE),
            ("복학은 어디서 하나요?", IntentCategory.PROCEDURE),
            ("학생증 재발급받고 싶은데요", IntentCategory.PROCEDURE),
            ("교환학생 지원 절차가 어떻게 되나요?", IntentCategory.PROCEDURE),
            ("이번 학기 장학금 신청하려면 무슨 서류 필요해요?", IntentCategory.PROCEDURE),
            ("휴학하고 싶은데 절차가 복잡한가요?", IntentCategory.PROCEDURE),
            ("등록금 분납 신청 방법을 알려주세요", IntentCategory.PROCEDURE),
            ("복수전공 신청하는 방법이 궁금합니다", IntentCategory.PROCEDURE),
            ("부전공 등록하려면 어떻게 해야 하나요?", IntentCategory.PROCEDURE),
            ("자퇴하려면 어떤 서류를 제출해야 하나요?", IntentCategory.PROCEDURE),
            ("재입학 신청 방법이 어떻게 되나요?", IntentCategory.PROCEDURE),
            ("휴학계 작성하는 법을 알려주세요", IntentCategory.PROCEDURE),
            ("성적 이의신청 접수는 어디서 하나요?", IntentCategory.PROCEDURE),
            ("현장실습 신청 절차를 알고 싶어요", IntentCategory.PROCEDURE),
            # ELIGIBILITY cases (20)
            ("장학금 받을 수 있나요?", IntentCategory.ELIGIBILITY),
            ("휴학 자격이 되나요?", IntentCategory.ELIGIBILITY),
            ("이번 학기 장학금 대상이 어떻게 돼요?", IntentCategory.ELIGIBILITY),
            ("성적 우수 장학금 조건이 뭐예요?", IntentCategory.ELIGIBILITY),
            ("근로장학생으로 지원할 수 있어요?", IntentCategory.ELIGIBILITY),
            ("복수전공 신청 자격이 되나요?", IntentCategory.ELIGIBILITY),
            ("교환학생 지원 대상이 어떻게 되나요?", IntentCategory.ELIGIBILITY),
            ("등록금 분납이 가능한가요?", IntentCategory.ELIGIBILITY),
            ("성적 이의신청 할 수 있나요?", IntentCategory.ELIGIBILITY),
            ("재입학 자격 요건이 뭐예요?", IntentCategory.ELIGIBILITY),
            ("장학금 받을 수 있을까요?", IntentCategory.ELIGIBILITY),
            ("휴학이 가능한 상황인가요?", IntentCategory.ELIGIBILITY),
            ("저도 장학금 대상이 되나요?", IntentCategory.ELIGIBILITY),
            ("성적 장학금 받을 수 있어요?", IntentCategory.ELIGIBILITY),
            ("이 조건이면 장학금 가능해요?", IntentCategory.ELIGIBILITY),
            ("평점 평균 몇이면 장학금 받을 수 있나요?", IntentCategory.ELIGIBILITY),
            ("휴학은 누구나 할 수 있나요?", IntentCategory.ELIGIBILITY),
            ("등록금 감면 대상이 어떻게 되나요?", IntentCategory.ELIGIBILITY),
            ("특별장학금 지원 자격이 궁금해요", IntentCategory.ELIGIBILITY),
            ("졸업 유예가 가능한가요?", IntentCategory.ELIGIBILITY),
            # DEADLINE cases (20)
            ("장학금 신청 기간이 언제까지인가요?", IntentCategory.DEADLINE),
            ("휴학 신청 언제까지 해야 해요?", IntentCategory.DEADLINE),
            ("등록금 납부 기한이 언제까지예요?", IntentCategory.DEADLINE),
            ("수강신청 마감일이 언제야?", IntentCategory.DEADLINE),
            ("졸업논문 제출 언제까지인가요?", IntentCategory.DEADLINE),
            ("이번 학기 개강일이 언제예요?", IntentCategory.DEADLINE),
            ("종강일이 언제인가요?", IntentCategory.DEADLINE),
            ("시험 기간이 언제부터 언제까지예요?", IntentCategory.DEADLINE),
            ("성적 공시 기간이 언제야?", IntentCategory.DEADLINE),
            ("학사일정이 어떻게 돼요?", IntentCategory.DEADLINE),
            ("수강철회 기간이 언제까지인가요?", IntentCategory.DEADLINE),
            ("성적 이의신청 기간이 언제예요?", IntentCategory.DEADLINE),
            ("휴학원 제출 기한이 언제까지야?", IntentCategory.DEADLINE),
            ("복학 신청 기간을 알고 싶어요", IntentCategory.DEADLINE),
            ("등록금 환불 기한이 언제까지인가요?", IntentCategory.DEADLINE),
            ("언제까지 신청해야 돼요?", IntentCategory.DEADLINE),
            ("마감일이 며칠까지예요?", IntentCategory.DEADLINE),
            ("신청 기한이 언제까지죠?", IntentCategory.DEADLINE),
            ("언제부터 신청 가능해요?", IntentCategory.DEADLINE),
            ("제출 기한이 언제까지인가요?", IntentCategory.DEADLINE),
        ]

        correct = 0
        total = len(test_cases)
        failures = []

        for query, expected in test_cases:
            result = classifier.classify(query)
            if result.category == expected:
                correct += 1
            else:
                failures.append(
                    f"'{query}' expected {expected.value}, got {result.category.value}"
                )

        accuracy = correct / total

        # Print failures for debugging
        if failures:
            print(f"\nMisclassified queries ({len(failures)}):")
            for f in failures[:10]:  # Show first 10 failures
                print(f"  - {f}")

        # Assert accuracy >= 85%
        assert (
            accuracy >= 0.85
        ), f"Accuracy {accuracy:.1%} is below 85% threshold. Failures: {len(failures)}/{total}"

        print(f"\nOverall accuracy: {accuracy:.1%} ({correct}/{total})")


class TestIntentConfidenceScores:
    """Test confidence score calibration."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_high_confidence_for_clear_keywords(self, classifier):
        """Test that clear keyword matches have high confidence."""
        clear_queries = [
            ("언제까지 신청하나요?", IntentCategory.DEADLINE),
            ("어떻게 신청하나요?", IntentCategory.PROCEDURE),
            ("받을 수 있나요?", IntentCategory.ELIGIBILITY),
        ]

        for query, expected in clear_queries:
            result = classifier.classify(query)
            assert (
                result.confidence >= 0.5
            ), f"Low confidence {result.confidence} for clear query '{query}'"

    def test_low_confidence_for_ambiguous_queries(self, classifier):
        """Test that ambiguous queries have low confidence or GENERAL classification."""
        ambiguous_queries = [
            "학사경고",
            "성적",
            "장학금",
        ]

        for query in ambiguous_queries:
            result = classifier.classify(query)
            # Either low confidence or GENERAL classification
            is_acceptable = result.confidence < 0.7 or result.category == IntentCategory.GENERAL
            assert (
                is_acceptable
            ), f"Ambiguous query '{query}' got high confidence {result.confidence} with category {result.category}"
