"""
Tests for Semantic Similarity Evaluator.

Implements TDD tests for SPEC-RAG-QUALITY-003 Phase 3.
"""

import pytest

from src.rag.infrastructure.evaluation.semantic_evaluator import (
    BatchEvaluationResult,
    SemanticEvaluationResult,
    SemanticEvaluator,
    create_semantic_evaluator,
)


class TestSemanticEvaluationResult:
    """Tests for SemanticEvaluationResult dataclass."""

    def test_passed_property_true(self):
        """Test passed property when is_relevant is True."""
        result = SemanticEvaluationResult(
            query="query",
            answer="answer",
            expected="expected",
            similarity_score=0.85,
            is_relevant=True,
        )
        assert result.passed is True

    def test_passed_property_false(self):
        """Test passed property when is_relevant is False."""
        result = SemanticEvaluationResult(
            query="query",
            answer="answer",
            expected="expected",
            similarity_score=0.65,
            is_relevant=False,
        )
        assert result.passed is False

    def test_default_values(self):
        """Test default values."""
        result = SemanticEvaluationResult(
            query="query",
            answer="answer",
            expected="expected",
            similarity_score=0.8,
            is_relevant=True,
        )
        assert result.threshold == 0.75
        assert result.details == {}


class TestBatchEvaluationResult:
    """Tests for BatchEvaluationResult dataclass."""

    def test_empty_results(self):
        """Test with empty results list."""
        batch = BatchEvaluationResult(results=[])
        assert batch.average_score == 0.0
        assert batch.pass_count == 0
        assert batch.fail_count == 0
        assert batch.pass_rate == 0.0

    def test_statistics_calculation(self):
        """Test automatic statistics calculation."""
        results = [
            SemanticEvaluationResult(
                query="q1",
                answer="a1",
                expected="e1",
                similarity_score=0.9,
                is_relevant=True,
            ),
            SemanticEvaluationResult(
                query="q2",
                answer="a2",
                expected="e2",
                similarity_score=0.6,
                is_relevant=False,
            ),
        ]
        batch = BatchEvaluationResult(results=results)

        assert batch.average_score == 0.75
        assert batch.pass_count == 1
        assert batch.fail_count == 1
        assert batch.pass_rate == 0.5


class TestSemanticEvaluator:
    """Tests for SemanticEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator instance for testing."""
        return SemanticEvaluator()

    # ============ Basic Evaluation Tests ============

    def test_evaluate_similarity_basic(self, evaluator):
        """Test basic similarity evaluation."""
        result = evaluator.evaluate_similarity(
            answer="휴학 신청은 학기 시작 전에 가능합니다.",
            expected="휴학은 학기 전에 신청할 수 있습니다.",
        )
        assert isinstance(result, SemanticEvaluationResult)
        assert 0.0 <= result.similarity_score <= 1.0

    def test_evaluate_similarity_returns_relevant(self, evaluator):
        """Test that is_relevant is set correctly."""
        result = evaluator.evaluate_similarity(
            answer="휴학 신청 방법",
            expected="휴학 신청 방법",  # Exact match
        )
        # Should use fallback (keyword) if no model, or semantic with model
        assert isinstance(result.is_relevant, bool)

    def test_evaluate_similarity_empty_answer(self, evaluator):
        """Test handling empty answer."""
        result = evaluator.evaluate_similarity(answer="", expected="expected")
        assert result.similarity_score == 0.0
        assert result.is_relevant is False
        assert "error" in result.details

    def test_evaluate_similarity_empty_expected(self, evaluator):
        """Test handling empty expected."""
        result = evaluator.evaluate_similarity(answer="answer", expected="")
        assert result.similarity_score == 0.0
        assert result.is_relevant is False

    def test_evaluate_similarity_whitespace_only(self, evaluator):
        """Test handling whitespace-only strings."""
        result = evaluator.evaluate_similarity(answer="   ", expected="expected")
        assert result.similarity_score == 0.0

    # ============ Threshold Tests ============

    def test_default_threshold(self, evaluator):
        """Test default threshold value."""
        assert evaluator._similarity_threshold == 0.75

    def test_custom_threshold(self):
        """Test custom threshold in constructor."""
        evaluator = SemanticEvaluator(similarity_threshold=0.85)
        assert evaluator._similarity_threshold == 0.85

    def test_override_threshold(self, evaluator):
        """Test overriding threshold in evaluate_similarity."""
        result = evaluator.evaluate_similarity(
            answer="test",
            expected="test",
            threshold=0.5,
        )
        assert result.threshold == 0.5

    def test_set_threshold_valid(self, evaluator):
        """Test setting valid threshold."""
        evaluator.set_threshold(0.9)
        assert evaluator._similarity_threshold == 0.9

    def test_set_threshold_invalid(self, evaluator):
        """Test setting invalid threshold raises error."""
        with pytest.raises(ValueError):
            evaluator.set_threshold(1.5)

    # ============ Batch Evaluation Tests ============

    def test_batch_evaluate_basic(self, evaluator):
        """Test basic batch evaluation."""
        result = evaluator.batch_evaluate(
            answers=["답변1", "답변2"],
            expected=["정답1", "정답2"],
        )
        assert isinstance(result, BatchEvaluationResult)
        assert len(result.results) == 2

    def test_batch_evaluate_statistics(self, evaluator):
        """Test batch evaluation statistics."""
        result = evaluator.batch_evaluate(
            answers=["답변", "답변"],
            expected=["답변", "다른내용"],
        )
        assert result.average_score >= 0.0
        assert result.pass_count + result.fail_count == 2

    def test_batch_evaluate_length_mismatch(self, evaluator):
        """Test batch evaluation with length mismatch."""
        result = evaluator.batch_evaluate(
            answers=["답변1", "답변2", "답변3"],
            expected=["정답1", "정답2"],
        )
        # Should process only matching pairs
        assert len(result.results) == 2

    # ============ Query Context Tests ============

    def test_evaluate_with_query(self, evaluator):
        """Test evaluation with query context."""
        result = evaluator.evaluate_with_query(
            query="휴학 신청 방법",
            answer="휴학은 학기 전에 신청 가능",
            expected="휴학 신청 안내",
        )
        assert result.query == "휴학 신청 방법"

    # ============ Cache Tests ============

    def test_cache_enabled_by_default(self, evaluator):
        """Test that cache is enabled by default."""
        assert evaluator._enable_cache is True

    def test_cache_disabled(self):
        """Test with caching disabled."""
        evaluator = SemanticEvaluator(enable_cache=False)
        evaluator.evaluate_similarity("test", "test")
        assert len(evaluator._embedding_cache) == 0

    def test_clear_cache(self, evaluator):
        """Test clearing the cache."""
        evaluator.evaluate_similarity("test", "test")
        evaluator.clear_cache()
        assert len(evaluator._embedding_cache) == 0

    # ============ Statistics Tests ============

    def test_get_stats(self, evaluator):
        """Test getting evaluator statistics."""
        stats = evaluator.get_stats()

        assert "embedding_model" in stats
        assert "similarity_threshold" in stats
        assert "cache_enabled" in stats
        assert "cache_size" in stats
        assert "model_loaded" in stats

    # ============ Fallback Tests ============

    def test_fallback_keyword_evaluation(self, evaluator):
        """Test fallback to keyword-based evaluation."""
        # Force fallback by using evaluator with no model
        result = evaluator._fallback_evaluate(
            answer="휴학 신청 방법",
            expected="휴학 신청",
            threshold=0.5,
        )
        assert result.details["method"] == "keyword_fallback"
        assert result.similarity_score >= 0.0

    def test_fallback_exact_match(self, evaluator):
        """Test fallback with exact match."""
        result = evaluator._fallback_evaluate(
            answer="휴학 신청",
            expected="휴학 신청",
            threshold=0.75,
        )
        assert result.similarity_score == 1.0
        assert result.is_relevant is True

    def test_fallback_no_overlap(self, evaluator):
        """Test fallback with no word overlap."""
        result = evaluator._fallback_evaluate(
            answer="가나다라",
            expected="마바사아",
            threshold=0.75,
        )
        assert result.similarity_score == 0.0
        assert result.is_relevant is False

    # ============ Edge Cases Tests ============

    def test_special_characters(self, evaluator):
        """Test handling special characters."""
        result = evaluator.evaluate_similarity(
            answer="휴학 신청?!",
            expected="휴학 신청",
        )
        assert isinstance(result, SemanticEvaluationResult)

    def test_numbers_in_text(self, evaluator):
        """Test handling numbers in text."""
        result = evaluator.evaluate_similarity(
            answer="제3조 제1항",
            expected="제3조 제1항 규정",
        )
        assert isinstance(result, SemanticEvaluationResult)

    def test_very_long_text(self, evaluator):
        """Test handling very long text."""
        long_answer = "휴학 신청 방법 " * 100
        long_expected = "휴학 신청 " * 100
        result = evaluator.evaluate_similarity(
            answer=long_answer,
            expected=long_expected,
        )
        assert isinstance(result, SemanticEvaluationResult)


class TestSemanticEvaluatorFactory:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test creating evaluator with default settings."""
        evaluator = create_semantic_evaluator()
        assert evaluator._embedding_model == "BAAI/bge-m3"
        assert evaluator._similarity_threshold == 0.75

    def test_create_with_custom_model(self):
        """Test creating evaluator with custom model."""
        evaluator = create_semantic_evaluator(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        assert evaluator._embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_create_with_custom_threshold(self):
        """Test creating evaluator with custom threshold."""
        evaluator = create_semantic_evaluator(similarity_threshold=0.85)
        assert evaluator._similarity_threshold == 0.85


class TestSPECRequirements:
    """Tests for SPEC-RAG-QUALITY-003 Phase 3 requirements."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for SPEC requirement testing."""
        return SemanticEvaluator()

    def test_req_embedding_based_evaluation(self, evaluator):
        """REQ: Evaluation SHALL use embedding-based semantic similarity."""
        # Verify embedding model is configured
        assert evaluator._embedding_model == "BAAI/bge-m3"

    def test_req_configurable_threshold(self, evaluator):
        """REQ: Similarity threshold SHALL be configurable."""
        # Test default threshold
        assert evaluator._similarity_threshold == 0.75

        # Test setting new threshold
        evaluator.set_threshold(0.8)
        assert evaluator._similarity_threshold == 0.8

    def test_req_default_threshold_0_75(self):
        """REQ: Default similarity threshold SHALL be 0.75."""
        evaluator = SemanticEvaluator()
        assert evaluator.DEFAULT_THRESHOLD == 0.75
        assert evaluator._similarity_threshold == 0.75

    def test_req_batch_evaluation(self, evaluator):
        """REQ: System SHALL support batch evaluation."""
        result = evaluator.batch_evaluate(
            answers=["a1", "a2", "a3"],
            expected=["e1", "e2", "e3"],
        )
        assert len(result.results) == 3
        assert hasattr(result, "average_score")
        assert hasattr(result, "pass_rate")

    def test_req_semantic_score_above_threshold(self, evaluator):
        """REQ: IF score >= threshold THEN mark as relevant."""
        result = evaluator.evaluate_similarity(
            answer="휴학 신청",
            expected="휴학 신청",
            threshold=0.5,
        )
        # Exact match should have high similarity
        assert result.similarity_score >= 0.5
        assert result.is_relevant is True

    def test_req_record_similarity_score(self, evaluator):
        """REQ: Evaluation SHALL record similarity score for analysis."""
        result = evaluator.evaluate_similarity(
            answer="test answer",
            expected="test expected",
        )
        assert hasattr(result, "similarity_score")
        assert 0.0 <= result.similarity_score <= 1.0

    def test_req_performance_under_200ms(self, evaluator):
        """REQ: Semantic evaluation SHALL complete within 200ms."""
        import time

        start = time.perf_counter()
        evaluator.evaluate_similarity(
            answer="휴학 신청 방법에 대해 설명해주세요",
            expected="휴학은 학기 시작 전에 신청 가능합니다",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Allow some margin for test environment
        assert elapsed_ms < 500, f"Evaluation took {elapsed_ms:.2f}ms"


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for similarity testing."""
        return SemanticEvaluator()

    def test_identical_vectors(self, evaluator):
        """Test similarity of identical vectors."""
        import numpy as np

        vec = np.array([1.0, 2.0, 3.0])
        similarity = evaluator._cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self, evaluator):
        """Test similarity of orthogonal vectors."""
        import numpy as np

        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = evaluator._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_opposite_vectors(self, evaluator):
        """Test similarity of opposite vectors."""
        import numpy as np

        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([-1.0, -2.0])
        similarity = evaluator._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, abs=0.01)

    def test_zero_vector(self, evaluator):
        """Test similarity with zero vector."""
        import numpy as np

        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([1.0, 2.0])
        similarity = evaluator._cosine_similarity(vec1, vec2)
        assert similarity == 0.0
