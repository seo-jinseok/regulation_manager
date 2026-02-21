"""
Unit tests for PersonaEvaluator and related types.

Tests for SPEC-RAG-QUALITY-010 Milestone 6: Persona Evaluation System.
TDD RED Phase - Write failing tests first.
"""

import pytest
from datetime import datetime

from src.rag.domain.evaluation.persona_definition import (
    PersonaDefinition,
    DEFAULT_PERSONAS,
)
from src.rag.domain.evaluation.persona_evaluator import (
    PersonaEvaluationScore,
    PersonaEvaluationResult,
    PersonaEvaluator,
    PersonaDashboardData,
)


@pytest.fixture
def sample_persona():
    """Create a sample persona for testing."""
    return DEFAULT_PERSONAS["freshman"]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "휴학하려면 뭐 해야 해요?"


@pytest.fixture
def good_response():
    """A well-formed response matching freshman persona requirements."""
    return """휴학 신청 방법을 알려드릴게요!

1. 학생정보시스템에서 휴학원을 작성하세요.
2. 지도교수님 승인을 받으세요.
3. 학사지원팀에 제출하시면 됩니다.

궁금한 게 더 있으면 언제든 물어보세요!"""


@pytest.fixture
def poor_response():
    """A poorly-formed response that doesn't match freshman persona requirements."""
    return """학칙 제40조에 따르면 휴학은 매학기 개시 1개월 전까지 신청하여야 하며,
「학칙」 제41조에 따라 교내시스템을 통해 지도교수 승인을 득한 후
학사지원팀에 제출하여야 한다. 휴학 기간은 통산하여 2년(4학기)을
초과할 수 없으며(「학칙」 제40조제3항), 군입대 및 질병의 경우는 예외로 한다."""


class TestPersonaEvaluationScore:
    """Tests for PersonaEvaluationScore dataclass."""

    def test_create_evaluation_score(self):
        """Test creating a PersonaEvaluationScore."""
        score = PersonaEvaluationScore(
            relevancy=0.85,
            clarity=0.90,
            completeness=0.80,
            citation_quality=0.75,
            overall=0.825,
        )

        assert score.relevancy == 0.85
        assert score.clarity == 0.90
        assert score.completeness == 0.80
        assert score.citation_quality == 0.75
        assert score.overall == 0.825

    def test_evaluation_score_to_dict(self):
        """Test serialization."""
        score = PersonaEvaluationScore(
            relevancy=0.85,
            clarity=0.90,
            completeness=0.80,
            citation_quality=0.75,
            overall=0.825,
        )

        data = score.to_dict()

        assert data["relevancy"] == 0.85
        assert data["clarity"] == 0.90
        assert data["completeness"] == 0.80
        assert data["citation_quality"] == 0.75
        assert data["overall"] == 0.825

    def test_evaluation_score_bounds(self):
        """Test that scores should be between 0.0 and 1.0."""
        # Valid scores
        score = PersonaEvaluationScore(
            relevancy=0.0,
            clarity=1.0,
            completeness=0.5,
            citation_quality=0.5,
            overall=0.5,
        )
        assert score.relevancy == 0.0
        assert score.clarity == 1.0


class TestPersonaEvaluationResult:
    """Tests for PersonaEvaluationResult dataclass."""

    def test_create_evaluation_result(self, sample_persona, sample_query, good_response):
        """Test creating a PersonaEvaluationResult."""
        scores = PersonaEvaluationScore(
            relevancy=0.85,
            clarity=0.90,
            completeness=0.80,
            citation_quality=0.75,
            overall=0.825,
        )

        result = PersonaEvaluationResult(
            persona_id=sample_persona.persona_id,
            query=sample_query,
            response=good_response,
            scores=scores,
            issues=["일부 정보 누락"],
            recommendations=["기한 정보 추가 권장"],
        )

        assert result.persona_id == "freshman"
        assert result.query == sample_query
        assert result.response == good_response
        assert result.scores.overall == 0.825
        assert len(result.issues) == 1
        assert len(result.recommendations) == 1

    def test_evaluation_result_to_dict(self, sample_persona, sample_query, good_response):
        """Test serialization."""
        scores = PersonaEvaluationScore(
            relevancy=0.85,
            clarity=0.90,
            completeness=0.80,
            citation_quality=0.75,
            overall=0.825,
        )

        result = PersonaEvaluationResult(
            persona_id=sample_persona.persona_id,
            query=sample_query,
            response=good_response,
            scores=scores,
            issues=[],
            recommendations=[],
        )

        data = result.to_dict()

        assert data["persona_id"] == "freshman"
        assert "scores" in data
        assert data["scores"]["overall"] == 0.825


class TestPersonaEvaluator:
    """Tests for PersonaEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a PersonaEvaluator instance."""
        return PersonaEvaluator()

    def test_init(self):
        """Test initialization."""
        evaluator = PersonaEvaluator()
        assert evaluator is not None

    def test_evaluate_persona(self, evaluator, sample_persona, sample_query, good_response):
        """Test evaluating a single persona response."""
        result = evaluator.evaluate_persona(
            query=sample_query,
            response=good_response,
            persona=sample_persona,
        )

        assert isinstance(result, PersonaEvaluationResult)
        assert result.persona_id == sample_persona.persona_id
        assert result.query == sample_query
        assert result.response == good_response
        assert isinstance(result.scores, PersonaEvaluationScore)
        assert result.scores.overall >= 0.0
        assert result.scores.overall <= 1.0

    def test_evaluate_poor_response(self, evaluator, sample_persona, sample_query, poor_response):
        """Test that poor response gets lower scores."""
        result = evaluator.evaluate_persona(
            query=sample_query,
            response=poor_response,
            persona=sample_persona,
        )

        # Poor response should have issues
        assert len(result.issues) > 0
        # Freshman expects simple language, poor response uses formal language
        assert result.scores.clarity < 0.8

    def test_evaluate_all_personas(self, evaluator, good_response):
        """Test evaluating responses for all personas."""
        queries = ["휴학 절차 알려주세요"] * 6
        responses = [good_response] * 6
        personas = list(DEFAULT_PERSONAS.values())

        results = evaluator.evaluate_all_personas(
            queries=queries,
            responses=responses,
            personas=personas,
        )

        assert isinstance(results, dict)
        assert len(results) == 6

        for persona_id, result_list in results.items():
            assert persona_id in ["freshman", "student", "professor", "staff", "parent", "international"]
            assert all(isinstance(r, PersonaEvaluationResult) for r in result_list)

    def test_get_weak_personas(self, evaluator):
        """Test identifying weak personas below threshold."""
        # Create mock results with low scores
        mock_results = [
            PersonaEvaluationResult(
                persona_id="international",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(
                    relevancy=0.6,
                    clarity=0.5,
                    completeness=0.6,
                    citation_quality=0.6,
                    overall=0.575,  # Below 0.65 threshold
                ),
                issues=["언어 장벽"],
                recommendations=["간단한 한국어 사용"],
            ),
            PersonaEvaluationResult(
                persona_id="freshman",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(
                    relevancy=0.8,
                    clarity=0.85,
                    completeness=0.75,
                    citation_quality=0.8,
                    overall=0.80,  # Above threshold
                ),
                issues=[],
                recommendations=[],
            ),
        ]

        weak_personas = evaluator.get_weak_personas(mock_results, threshold=0.65)

        assert "international" in weak_personas
        assert "freshman" not in weak_personas

    def test_get_weak_personas_empty(self, evaluator):
        """Test weak personas with empty results."""
        weak_personas = evaluator.get_weak_personas([], threshold=0.65)
        assert weak_personas == []

    def test_get_weak_personas_all_passing(self, evaluator):
        """Test weak personas when all pass."""
        mock_results = [
            PersonaEvaluationResult(
                persona_id="freshman",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(
                    relevancy=0.8,
                    clarity=0.8,
                    completeness=0.8,
                    citation_quality=0.8,
                    overall=0.8,
                ),
                issues=[],
                recommendations=[],
            ),
        ]

        weak_personas = evaluator.get_weak_personas(mock_results, threshold=0.65)
        assert weak_personas == []


class TestPersonaDashboardData:
    """Tests for PersonaDashboardData structure."""

    def test_create_dashboard_data(self):
        """Test creating dashboard data."""
        dashboard = PersonaDashboardData(
            evaluation_date="2026-02-21",
            personas={
                "freshman": {
                    "avg_overall": 0.72,
                    "avg_relevancy": 0.75,
                    "avg_clarity": 0.80,
                    "avg_completeness": 0.68,
                    "avg_citation": 0.65,
                    "issue_count": 3,
                },
            },
            weak_personas=["international"],
            recommendations={
                "international": "간단한 한국어 사용 및 복잡한 용어 설명 강화 필요",
            },
        )

        assert dashboard.evaluation_date == "2026-02-21"
        assert "freshman" in dashboard.personas
        assert dashboard.weak_personas == ["international"]

    def test_dashboard_to_dict(self):
        """Test dashboard serialization."""
        dashboard = PersonaDashboardData(
            evaluation_date="2026-02-21",
            personas={
                "freshman": {
                    "avg_overall": 0.72,
                    "avg_relevancy": 0.75,
                    "avg_clarity": 0.80,
                    "avg_completeness": 0.68,
                    "avg_citation": 0.65,
                    "issue_count": 3,
                },
            },
            weak_personas=[],
            recommendations={},
        )

        data = dashboard.to_dict()

        assert data["evaluation_date"] == "2026-02-21"
        assert "personas" in data
        assert "weak_personas" in data
        assert "recommendations" in data

    def test_generate_from_results(self):
        """Test generating dashboard from evaluation results."""
        results = [
            PersonaEvaluationResult(
                persona_id="freshman",
                query="q1",
                response="r1",
                scores=PersonaEvaluationScore(
                    relevancy=0.8,
                    clarity=0.85,
                    completeness=0.7,
                    citation_quality=0.75,
                    overall=0.775,
                ),
                issues=["issue1"],
                recommendations=[],
            ),
            PersonaEvaluationResult(
                persona_id="freshman",
                query="q2",
                response="r2",
                scores=PersonaEvaluationScore(
                    relevancy=0.75,
                    clarity=0.8,
                    completeness=0.65,
                    citation_quality=0.7,
                    overall=0.725,
                ),
                issues=[],
                recommendations=[],
            ),
        ]

        dashboard = PersonaDashboardData.from_results(
            results=results,
            evaluation_date="2026-02-21",
        )

        assert dashboard.evaluation_date == "2026-02-21"
        assert "freshman" in dashboard.personas
        # Average of 0.775 and 0.725 = 0.75
        assert dashboard.personas["freshman"]["avg_overall"] == pytest.approx(0.75, 0.01)
        assert dashboard.personas["freshman"]["issue_count"] == 1


class TestPersonaEvaluatorScoring:
    """Tests for persona-specific scoring logic."""

    @pytest.fixture
    def evaluator(self):
        """Create a PersonaEvaluator instance."""
        return PersonaEvaluator()

    def test_freshman_simple_language_scoring(self, evaluator):
        """Test that simple language scores high for freshman."""
        simple_response = "휴학하려면 학생정보시스템에서 신청하세요. 지도교수님 승인 받으면 돼요!"
        result = evaluator.evaluate_persona(
            query="휴학하고 싶어요",
            response=simple_response,
            persona=DEFAULT_PERSONAS["freshman"],
        )

        # Simple language should score high on clarity for freshman
        assert result.scores.clarity >= 0.7

    def test_professor_technical_language_scoring(self, evaluator):
        """Test that technical language with citations scores high for professor."""
        technical_response = """「학칙」 제40조에 따라 휴학 절차를 안내합니다.

1. 신청 기간: 매학기 개시 1개월 전까지(「학칙」 제40조제1항)
2. 승인 절차: 지도교수 승인 후 학사지원팀 제출(「학칙」 제41조)
3. 휴학 기간: 통산 2년(4학기) 이내(「학칙」 제40조제3항)"""

        result = evaluator.evaluate_persona(
            query="휴학 관련 규정 해석 부탁드립니다",
            response=technical_response,
            persona=DEFAULT_PERSONAS["professor"],
        )

        # Technical language with citations should score high for professor
        assert result.scores.citation_quality >= 0.7

    def test_parent_friendly_language_scoring(self, evaluator):
        """Test that friendly language with contact info scores high for parent."""
        friendly_response = """안녕하세요, 학부모님! 휴학 신청에 대해 안내드릴게요.

자녀분이 학생정보시스템에서 신청하시면 됩니다.
궁금하신 점이 있으시면 학사지원팀(051-XXX-XXXX)으로 연락 주세요."""

        result = evaluator.evaluate_persona(
            query="우리 아이 휴학 신청 어떻게 하나요?",
            response=friendly_response,
            persona=DEFAULT_PERSONAS["parent"],
        )

        # Friendly language should score high for parent
        assert result.scores.clarity >= 0.7

    def test_international_simple_korean_scoring(self, evaluator):
        """Test that simple Korean scores high for international students."""
        simple_korean_response = """휴학 신청 방법:

1. 컴퓨터에서 신청하세요.
2. 교수님 승인 받으세요.
3. 제출 완료!

질문 있으면 물어보세요."""

        result = evaluator.evaluate_persona(
            query="휴학 어떻게 해요?",
            response=simple_korean_response,
            persona=DEFAULT_PERSONAS["international"],
        )

        # Simple Korean should score well for international
        assert result.scores.clarity >= 0.6


class TestWeakPersonaIdentification:
    """Tests for weak persona identification and recommendations."""

    @pytest.fixture
    def evaluator(self):
        """Create a PersonaEvaluator instance."""
        return PersonaEvaluator()

    def test_generate_improvement_recommendations(self, evaluator):
        """Test generating improvement recommendations for weak personas."""
        weak_results = [
            PersonaEvaluationResult(
                persona_id="international",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(
                    relevancy=0.6,
                    clarity=0.5,
                    completeness=0.6,
                    citation_quality=0.6,
                    overall=0.575,
                ),
                issues=["복잡한 한국어 사용", "전문 용어 미설명"],
                recommendations=[],
            ),
        ]

        recommendations = evaluator.generate_recommendations(weak_results)

        assert "international" in recommendations
        # recommendations["international"] is a string, check directly
        rec = recommendations["international"]
        assert "한국어" in rec or "용어" in rec

    def test_identify_common_issues(self, evaluator):
        """Test identifying common issues across evaluations."""
        results = [
            PersonaEvaluationResult(
                persona_id="freshman",
                query="q1",
                response="r1",
                scores=PersonaEvaluationScore(0.7, 0.7, 0.7, 0.7, 0.7),
                issues=["복잡한 표현"],
                recommendations=[],
            ),
            PersonaEvaluationResult(
                persona_id="freshman",
                query="q2",
                response="r2",
                scores=PersonaEvaluationScore(0.7, 0.7, 0.7, 0.7, 0.7),
                issues=["복잡한 표현"],
                recommendations=[],
            ),
        ]

        common_issues = evaluator.identify_common_issues(results, "freshman")

        assert "복잡한 표현" in common_issues

    def test_recommendation_for_professor_citation_issue(self, evaluator):
        """Test recommendation for professor with citation issues."""
        results = [
            PersonaEvaluationResult(
                persona_id="professor",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(0.6, 0.6, 0.6, 0.5, 0.575),
                issues=["조항 인용 부족"],
                recommendations=[],
            ),
        ]

        recommendations = evaluator.generate_recommendations(results)

        assert "professor" in recommendations
        assert "인용" in recommendations["professor"] or "조항" in recommendations["professor"]

    def test_recommendation_for_parent_contact_issue(self, evaluator):
        """Test recommendation for parent with contact info issues."""
        results = [
            PersonaEvaluationResult(
                persona_id="parent",
                query="연락처 알려주세요",
                response="담당 부서에 문의하세요.",
                scores=PersonaEvaluationScore(0.6, 0.6, 0.6, 0.6, 0.6),
                issues=["연락처 정보 누락"],
                recommendations=[],
            ),
        ]

        recommendations = evaluator.generate_recommendations(results)

        assert "parent" in recommendations
        assert "연락처" in recommendations["parent"]

    def test_recommendation_for_staff_deadline_issue(self, evaluator):
        """Test recommendation for staff with deadline issues."""
        results = [
            PersonaEvaluationResult(
                persona_id="staff",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(0.6, 0.6, 0.6, 0.6, 0.6),
                issues=["처리 기한 누락", "담당 부서 정보 부족"],
                recommendations=[],
            ),
        ]

        recommendations = evaluator.generate_recommendations(results)

        assert "staff" in recommendations
        assert "기한" in recommendations["staff"] or "부서" in recommendations["staff"]

    def test_dashboard_with_weak_persona(self, evaluator):
        """Test dashboard generation with weak persona."""
        results = [
            PersonaEvaluationResult(
                persona_id="international",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(0.5, 0.5, 0.5, 0.5, 0.5),
                issues=["복잡한 한국어 사용"],
                recommendations=[],
            ),
            PersonaEvaluationResult(
                persona_id="freshman",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(0.8, 0.8, 0.8, 0.8, 0.8),
                issues=[],
                recommendations=[],
            ),
        ]

        dashboard = PersonaDashboardData.from_results(
            results=results,
            evaluation_date="2026-02-21",
            threshold=0.65,
        )

        assert "international" in dashboard.weak_personas
        assert "freshman" not in dashboard.weak_personas
        assert "international" in dashboard.recommendations

    def test_recommendation_without_korean_issue(self, evaluator):
        """Test recommendation for international without Korean-specific issue."""
        results = [
            PersonaEvaluationResult(
                persona_id="international",
                query="test",
                response="test",
                scores=PersonaEvaluationScore(0.5, 0.5, 0.5, 0.5, 0.5),
                issues=["정보 누락"],  # No Korean-specific issue
                recommendations=[],
            ),
        ]

        recommendations = evaluator.generate_recommendations(results)

        assert "international" in recommendations
        # Should get the fallback recommendation for international
        assert "언어" in recommendations["international"] or "시각" in recommendations["international"]
