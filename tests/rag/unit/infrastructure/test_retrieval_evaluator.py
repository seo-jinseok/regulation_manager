"""
Unit tests for RetrievalEvaluator (Corrective RAG).

Tests cover:
- Relevance score calculation
- Correction trigger decision
- Component score evaluation (top score, keyword overlap, diversity)
"""

from unittest.mock import MagicMock

import pytest

from src.rag.infrastructure.retrieval_evaluator import (
    CorrectionStrategy,
    RetrievalEvaluator,
)


class FakeChunk:
    """Fake Chunk for testing."""

    def __init__(
        self,
        id: str,
        text: str,
        title: str = "",
        rule_code: str = "",
    ):
        self.id = id
        self.text = text
        self.title = title
        self.rule_code = rule_code


class FakeSearchResult:
    """Fake SearchResult for testing."""

    def __init__(self, chunk: FakeChunk, score: float, rank: int = 1):
        self.chunk = chunk
        self.score = score
        self.rank = rank


def make_result(
    doc_id: str,
    text: str,
    score: float,
    title: str = "",
    rule_code: str = "",
    rank: int = 1,
) -> FakeSearchResult:
    """Helper to create fake search results."""
    chunk = FakeChunk(id=doc_id, text=text, title=title, rule_code=rule_code)
    return FakeSearchResult(chunk=chunk, score=score, rank=rank)


class TestRetrievalEvaluatorBasic:
    """Test basic evaluation functionality."""

    @pytest.fixture
    def evaluator(self) -> RetrievalEvaluator:
        return RetrievalEvaluator()

    def test_evaluate_empty_results(self, evaluator: RetrievalEvaluator):
        """빈 결과는 0점"""
        score = evaluator.evaluate("query", [])
        assert score == 0.0

    def test_evaluate_returns_normalized_score(self, evaluator: RetrievalEvaluator):
        """점수가 0-1 범위인지 확인"""
        results = [
            make_result("doc1", "장학금 신청 방법", 0.8, title="장학규정"),
        ]
        score = evaluator.evaluate("장학금 신청", results)
        assert 0.0 <= score <= 1.0

    def test_evaluate_high_score_for_relevant_results(
        self, evaluator: RetrievalEvaluator
    ):
        """관련성 높은 결과는 높은 점수"""
        results = [
            make_result(
                "doc1",
                "장학금 신청 절차와 방법에 대한 규정",
                0.95,
                title="장학금 규정",
                rule_code="3-1-24",
            ),
            make_result(
                "doc2",
                "장학금 지급 기준",
                0.85,
                title="장학규정",
                rule_code="3-1-25",
            ),
        ]
        score = evaluator.evaluate("장학금 신청", results)
        assert score >= 0.5

    def test_evaluate_low_score_for_irrelevant_results(
        self, evaluator: RetrievalEvaluator
    ):
        """관련성 낮은 결과는 낮은 점수"""
        results = [
            make_result("doc1", "무관한 내용입니다", 0.3, title="다른규정"),
            make_result("doc2", "전혀 다른 주제", 0.2, title="기타"),
        ]
        score = evaluator.evaluate("장학금 신청", results)
        assert score < 0.5


class TestNeedsCorrection:
    """Test correction trigger decision."""

    @pytest.fixture
    def evaluator(self) -> RetrievalEvaluator:
        return RetrievalEvaluator(relevance_threshold=0.4)

    def test_needs_correction_empty_results(self, evaluator: RetrievalEvaluator):
        """빈 결과는 수정 필요"""
        assert evaluator.needs_correction("query", []) is True

    def test_needs_correction_few_results(self, evaluator: RetrievalEvaluator):
        """결과가 너무 적으면 수정 필요"""
        results = [make_result("doc1", "content", 0.5)]
        assert evaluator.needs_correction("query", results) is True

    def test_needs_correction_low_relevance(self, evaluator: RetrievalEvaluator):
        """관련성 낮으면 수정 필요"""
        results = [
            make_result("doc1", "무관한 내용", 0.2),
            make_result("doc2", "역시 무관한", 0.1),
        ]
        assert evaluator.needs_correction("장학금", results) is True

    def test_no_correction_high_relevance(self, evaluator: RetrievalEvaluator):
        """관련성 높으면 수정 불필요"""
        results = [
            make_result(
                "doc1",
                "장학금 신청 절차",
                0.9,
                title="장학금 규정",
                rule_code="3-1-24",
            ),
            make_result(
                "doc2",
                "장학금 지급 기준",
                0.85,
                title="장학규정",
                rule_code="3-1-25",
            ),
            make_result(
                "doc3",
                "장학생 선발",
                0.8,
                title="장학세칙",
                rule_code="3-2-1",
            ),
        ]
        assert evaluator.needs_correction("장학금 신청", results) is False

    def test_custom_threshold(self):
        """사용자 정의 임계값 적용"""
        strict_evaluator = RetrievalEvaluator(relevance_threshold=0.8)

        results = [
            make_result("doc1", "장학금 내용", 0.6, title="장학규정"),
            make_result("doc2", "장학금 신청", 0.55, title="장학세칙"),
        ]

        # 0.8 임계값으로는 수정 필요
        assert strict_evaluator.needs_correction("장학금", results) is True

        # 0.4 임계값으로는 수정 불필요
        lenient_evaluator = RetrievalEvaluator(relevance_threshold=0.4)
        assert lenient_evaluator.needs_correction("장학금", results) is False


class TestComponentScores:
    """Test individual component score calculations."""

    @pytest.fixture
    def evaluator(self) -> RetrievalEvaluator:
        return RetrievalEvaluator()

    def test_top_score_component(self, evaluator: RetrievalEvaluator):
        """Top result score가 반영되는지 확인"""
        high_score_results = [make_result("doc1", "content", 0.95)]
        low_score_results = [make_result("doc1", "content", 0.3)]

        high_eval = evaluator._evaluate_top_score(high_score_results)
        low_eval = evaluator._evaluate_top_score(low_score_results)

        assert high_eval > low_eval

    def test_keyword_overlap_component(self, evaluator: RetrievalEvaluator):
        """키워드 오버랩이 반영되는지 확인"""
        # 키워드 오버랩 높은 경우
        high_overlap = [
            make_result("doc1", "장학금 신청 방법", 0.8, title="장학금"),
        ]
        # 키워드 오버랩 낮은 경우
        low_overlap = [
            make_result("doc1", "무관한 내용", 0.8, title="다른규정"),
        ]

        high_score = evaluator._evaluate_keyword_overlap("장학금 신청", high_overlap)
        low_score = evaluator._evaluate_keyword_overlap("장학금 신청", low_overlap)

        assert high_score > low_score

    def test_diversity_component(self, evaluator: RetrievalEvaluator):
        """결과 다양성이 반영되는지 확인"""
        # 다양한 규정에서 온 결과
        diverse_results = [
            make_result("doc1", "content", 0.8, rule_code="1-1-1"),
            make_result("doc2", "content", 0.7, rule_code="2-1-1"),
            make_result("doc3", "content", 0.6, rule_code="3-1-1"),
        ]
        # 같은 규정에서 온 결과
        same_source = [
            make_result("doc1", "content", 0.8, rule_code="1-1-1"),
            make_result("doc2", "content", 0.7, rule_code="1-1-2"),
            make_result("doc3", "content", 0.6, rule_code="1-1-3"),
        ]

        diverse_score = evaluator._evaluate_diversity(diverse_results)
        same_score = evaluator._evaluate_diversity(same_source)

        assert diverse_score > same_score


class TestTokenization:
    """Test tokenization for keyword matching."""

    @pytest.fixture
    def evaluator(self) -> RetrievalEvaluator:
        return RetrievalEvaluator()

    def test_tokenize_korean(self, evaluator: RetrievalEvaluator):
        """한글 토큰화"""
        tokens = evaluator._tokenize("장학금 신청 절차")
        assert "장학금" in tokens
        assert "신청" in tokens
        assert "절차" in tokens

    def test_tokenize_removes_stopwords(self, evaluator: RetrievalEvaluator):
        """불용어 제거"""
        tokens = evaluator._tokenize("장학금을 신청하는 방법은")
        # "을", "는" 등 조사가 제거되어야 함
        assert "을" not in tokens
        assert "는" not in tokens

    def test_tokenize_removes_short_tokens(self, evaluator: RetrievalEvaluator):
        """짧은 토큰 제거"""
        tokens = evaluator._tokenize("가 나 다 장학금")
        # 1글자 토큰은 제거
        assert "가" not in tokens
        assert "장학금" in tokens


class TestCorrectionStrategy:
    """Test correction strategy for query rewriting."""

    def test_get_corrected_query_no_analyzer(self):
        """QueryAnalyzer 없으면 None 반환"""
        strategy = CorrectionStrategy(query_analyzer=None)
        result = strategy.get_corrected_query("original query")
        assert result is None

    def test_get_corrected_query_with_expansion(self):
        """쿼리 확장으로 수정된 쿼리 반환"""
        mock_analyzer = MagicMock()
        mock_analyzer.expand_query.return_value = "original query 동의어1 동의어2"

        strategy = CorrectionStrategy(query_analyzer=mock_analyzer)
        result = strategy.get_corrected_query("original query")

        assert result == "original query 동의어1 동의어2"
        mock_analyzer.expand_query.assert_called_once_with("original query")

    def test_get_corrected_query_no_change(self):
        """확장되지 않으면 LLM 리라이팅 시도"""
        mock_analyzer = MagicMock()
        mock_analyzer.expand_query.return_value = "original query"  # 변화 없음

        mock_rewrite_result = MagicMock()
        mock_rewrite_result.rewritten = "rewritten query"
        mock_analyzer.rewrite_query_with_info.return_value = mock_rewrite_result

        strategy = CorrectionStrategy(query_analyzer=mock_analyzer)
        result = strategy.get_corrected_query("original query")

        assert result == "rewritten query"

    def test_get_corrected_query_expansion_preferred(self):
        """확장이 있으면 LLM 리라이팅보다 우선"""
        mock_analyzer = MagicMock()
        mock_analyzer.expand_query.return_value = "expanded query"

        strategy = CorrectionStrategy(query_analyzer=mock_analyzer)
        result = strategy.get_corrected_query("original query")

        # LLM 리라이팅이 호출되지 않아야 함
        mock_analyzer.rewrite_query_with_info.assert_not_called()
        assert result == "expanded query"


class TestDynamicThresholds:
    """Test dynamic threshold support for query complexity."""

    def test_threshold_dict_accepted(self):
        """쿼리 유형별 임계값 딕셔너리를 받을 수 있어야 함."""
        thresholds = {"simple": 0.3, "medium": 0.4, "complex": 0.5}
        evaluator = RetrievalEvaluator(relevance_threshold=thresholds)

        assert evaluator._thresholds == thresholds

    def test_float_threshold_converted_to_dict(self):
        """단일 float 임계값은 내부적으로 딕셔너리로 변환."""
        evaluator = RetrievalEvaluator(relevance_threshold=0.35)

        # 모든 쿼리 유형에 동일한 임계값 적용
        assert evaluator._thresholds["simple"] == 0.35
        assert evaluator._thresholds["medium"] == 0.35
        assert evaluator._thresholds["complex"] == 0.35

    def test_needs_correction_uses_query_complexity(self):
        """needs_correction이 쿼리 복잡도를 고려해야 함."""
        thresholds = {"simple": 0.3, "medium": 0.4, "complex": 0.5}
        evaluator = RetrievalEvaluator(relevance_threshold=thresholds)

        # 점수 0.35인 결과
        results = [
            make_result("doc1", "내용", 0.35, rule_code="1-1-1"),
            make_result("doc2", "내용2", 0.30, rule_code="2-1-1"),
        ]

        # simple (임계값 0.3): 0.35 > 0.3 → 수정 불필요
        assert evaluator.needs_correction("쿼리", results, complexity="simple") is False

        # complex (임계값 0.5): 0.35 < 0.5 → 수정 필요
        assert evaluator.needs_correction("쿼리", results, complexity="complex") is True

    def test_needs_correction_default_complexity_medium(self):
        """complexity 미지정 시 medium 사용."""
        thresholds = {"simple": 0.3, "medium": 0.4, "complex": 0.5}
        evaluator = RetrievalEvaluator(relevance_threshold=thresholds)

        results = [
            make_result("doc1", "내용", 0.35, rule_code="1-1-1"),
            make_result("doc2", "내용2", 0.30, rule_code="2-1-1"),
        ]

        # medium (임계값 0.4): 0.35 < 0.4 → 수정 필요
        assert evaluator.needs_correction("쿼리", results) is True

    def test_default_thresholds_from_config(self):
        """설정에서 기본 임계값을 가져와야 함."""
        from src.rag.config import get_config, reset_config
        reset_config()

        evaluator = RetrievalEvaluator()
        config = get_config()

        assert evaluator._thresholds == config.corrective_rag_thresholds
        reset_config()


class TestEvaluatorWeights:
    """Test the weighted combination of component scores."""

    def test_weight_distribution(self):
        """가중치 분포 확인: top 50%, keyword 30%, diversity 20%"""
        evaluator = RetrievalEvaluator()

        # 모든 컴포넌트가 1.0일 때 최종 점수도 1.0
        results = [
            make_result(
                "doc1",
                "장학금 신청 방법",
                1.0,  # top_score = 1.0
                title="장학금 규정",
                rule_code="1-1-1",
            ),
            make_result(
                "doc2",
                "장학금 지급 기준",
                0.9,
                title="장학규정",
                rule_code="2-1-1",
            ),
            make_result(
                "doc3",
                "장학금 선발 기준",
                0.8,
                title="장학세칙",
                rule_code="3-1-1",
            ),
        ]

        score = evaluator.evaluate("장학금 신청", results)

        # 모든 컴포넌트가 높으면 최종 점수도 높아야 함
        assert score >= 0.7
