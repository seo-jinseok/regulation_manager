"""
Characterization tests for CustomLLMJudge module.

These tests document the current behavior of the custom LLM-as-Judge
evaluation system without prescribing how it should behave.

Module under test: src/rag/domain/evaluation/custom_judge.py
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import asdict
import asyncio
import json

import anyio

from src.rag.domain.evaluation.custom_judge import (
    CustomLLMJudge,
    CustomJudgeConfig,
    CustomEvaluationResult,
    EVALUATION_PROMPT,
    OPENAI_AVAILABLE,
)


class TestCustomJudgeConfig:
    """Characterization tests for CustomJudgeConfig."""

    def test_config_default_values(self):
        """Document default configuration values."""
        # Arrange & Act
        config = CustomJudgeConfig()

        # Assert
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.timeout == 60

    def test_config_default_weights(self):
        """Document default evaluation weights."""
        # Arrange & Act
        config = CustomJudgeConfig()

        # Assert - Weights should sum to 1.0
        assert config.faithfulness_weight == 0.35
        assert config.answer_relevancy_weight == 0.25
        assert config.contextual_precision_weight == 0.20
        assert config.contextual_recall_weight == 0.20
        total = (
            config.faithfulness_weight
            + config.answer_relevancy_weight
            + config.contextual_precision_weight
            + config.contextual_recall_weight
        )
        assert total == 1.0

    def test_config_api_key_from_env(self):
        """Document API key loading from environment."""
        # Arrange
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            # Act
            config = CustomJudgeConfig()

        # Assert
        assert config.api_key == "test-key"

    def test_config_custom_api_key(self):
        """Document custom API key setting."""
        # Arrange & Act
        config = CustomJudgeConfig(api_key="custom-key")

        # Assert
        assert config.api_key == "custom-key"

    def test_config_weight_normalization(self):
        """Document weight normalization when not summing to 1.0."""
        # Arrange & Act - Weights sum to 0.9
        config = CustomJudgeConfig(
            faithfulness_weight=0.4,
            answer_relevancy_weight=0.3,
            contextual_precision_weight=0.1,
            contextual_recall_weight=0.1,
        )

        # Assert - Weights should be normalized
        total = (
            config.faithfulness_weight
            + config.answer_relevancy_weight
            + config.contextual_precision_weight
            + config.contextual_recall_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_config_retry_settings(self):
        """Document retry configuration."""
        # Arrange & Act
        config = CustomJudgeConfig()

        # Assert
        assert config.max_retries == 2
        assert config.retry_delay == 1.0


class TestCustomEvaluationResult:
    """Characterization tests for CustomEvaluationResult."""

    def test_result_creation(self):
        """Document CustomEvaluationResult creation."""
        # Arrange & Act
        result = CustomEvaluationResult(
            query="test query",
            response="test response",
            contexts=["context"],
            faithfulness_score=0.9,
            answer_relevancy_score=0.85,
            contextual_precision_score=0.8,
            contextual_recall_score=0.75,
            overall_score=0.825,
            passed=True,
        )

        # Assert
        assert result.query == "test query"
        assert result.response == "test response"
        assert result.passed is True

    def test_result_auto_timestamp(self):
        """Document auto-generated timestamp."""
        # Arrange & Act
        result = CustomEvaluationResult(
            query="test",
            response="response",
            contexts=[],
        )

        # Assert
        assert result.evaluation_timestamp != ""

    def test_result_default_values(self):
        """Document default values."""
        # Arrange & Act
        result = CustomEvaluationResult(
            query="test",
            response="response",
            contexts=[],
        )

        # Assert
        assert result.passed is False
        assert result.failure_reasons == []
        assert result.judge_model == "gpt-4o"
        assert result.stage == 1

    def test_result_to_dict(self):
        """Document result serialization."""
        # Arrange
        result = CustomEvaluationResult(
            query="test query",
            response="test response",
            contexts=["context"],
            faithfulness_score=0.9,
            overall_score=0.9,
            passed=True,
        )

        # Act
        data = result.to_dict()

        # Assert
        assert data["query"] == "test query"
        assert data["overall_score"] == 0.9
        assert "faithfulness" in data
        assert isinstance(data["faithfulness"], dict)

    def test_result_to_evaluation_result(self):
        """Document conversion to EvaluationResult."""
        # Arrange
        from src.rag.domain.evaluation.models import EvaluationThresholds

        result = CustomEvaluationResult(
            query="test query",
            response="test response",
            contexts=["context"],
            faithfulness_score=0.9,
            answer_relevancy_score=0.85,
            contextual_precision_score=0.8,
            contextual_recall_score=0.75,
            overall_score=0.825,
            passed=True,
            faithfulness_reasoning="Test reasoning",
        )
        thresholds = EvaluationThresholds(stage=1)

        # Act
        eval_result = result.to_evaluation_result(thresholds)

        # Assert
        assert eval_result.query == "test query"
        assert eval_result.answer == "test response"
        assert eval_result.faithfulness == 0.9
        assert eval_result.metadata["framework"] == "custom_judge"


class TestCustomLLMJudgeInit:
    """Characterization tests for CustomLLMJudge initialization."""

    def test_init_requires_openai(self):
        """Document that OpenAI library is required."""
        # Arrange
        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", False):
            # Act & Assert
            with pytest.raises(ImportError) as exc_info:
                CustomLLMJudge()

            assert "OpenAI library is required" in str(exc_info.value)

    def test_init_with_default_config(self):
        """Document initialization with default config."""
        # Arrange
        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", True):
            with patch("src.rag.domain.evaluation.custom_judge.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()

                # Act
                judge = CustomLLMJudge()

        # Assert
        assert judge.config.model == "gpt-4o"

    def test_init_with_custom_config(self):
        """Document initialization with custom config."""
        # Arrange
        config = CustomJudgeConfig(
            model="gpt-4o-mini",
            temperature=0.1,
        )

        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", True):
            with patch("src.rag.domain.evaluation.custom_judge.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()

                # Act
                judge = CustomLLMJudge(config=config)

        # Assert
        assert judge.config.model == "gpt-4o-mini"
        assert judge.config.temperature == 0.1


class TestCustomLLMJudgeEvaluate:
    """Characterization tests for CustomLLMJudge.evaluate method."""

    @pytest.fixture
    def mock_judge(self):
        """Create CustomLLMJudge with mocked OpenAI client."""
        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", True):
            with patch("src.rag.domain.evaluation.custom_judge.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client

                config = CustomJudgeConfig(api_key="test-key")
                judge = CustomLLMJudge(config=config)
                judge.client = mock_client
                return judge

    @pytest.mark.anyio
    async def test_evaluate_returns_custom_result(self, mock_judge):
        """Document that evaluate returns CustomEvaluationResult."""
        # Arrange
        mock_judge.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({
                                "faithfulness": {"score": 0.9, "reasoning": "Good"},
                                "answer_relevancy": {"score": 0.85, "reasoning": "Good"},
                                "contextual_precision": {"score": 0.8, "reasoning": "Good"},
                                "contextual_recall": {"score": 0.75, "reasoning": "Good"},
                            })
                        )
                    )
                ]
            )
        )

        # Act
        result = await mock_judge.evaluate(
            query="test query",
            response="test response",
            contexts=["context"],
        )

        # Assert
        assert isinstance(result, CustomEvaluationResult)

    @pytest.mark.anyio
    async def test_evaluate_calculates_weighted_overall(self, mock_judge):
        """Document weighted overall score calculation."""
        # Arrange
        mock_judge.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({
                                "faithfulness": {"score": 1.0, "reasoning": ""},
                                "answer_relevancy": {"score": 1.0, "reasoning": ""},
                                "contextual_precision": {"score": 1.0, "reasoning": ""},
                                "contextual_recall": {"score": 1.0, "reasoning": ""},
                            })
                        )
                    )
                ]
            )
        )

        # Act
        result = await mock_judge.evaluate(
            query="test",
            response="response",
            contexts=["context"],
        )

        # Assert - With all 1.0 scores, overall should be 1.0
        assert result.overall_score == 1.0

    @pytest.mark.anyio
    async def test_evaluate_with_expected_answer(self, mock_judge):
        """Document evaluation with expected answer."""
        # Arrange
        mock_judge.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({
                                "faithfulness": {"score": 0.9, "reasoning": ""},
                                "answer_relevancy": {"score": 0.85, "reasoning": ""},
                                "contextual_precision": {"score": 0.8, "reasoning": ""},
                                "contextual_recall": {"score": 0.75, "reasoning": ""},
                            })
                        )
                    )
                ]
            )
        )

        # Act
        result = await mock_judge.evaluate(
            query="test",
            response="response",
            contexts=["context"],
            expected_answer="expected",
        )

        # Assert
        assert result.ground_truth == "expected"

    @pytest.mark.anyio
    async def test_evaluate_pass_fail_determination(self, mock_judge):
        """Document pass/fail determination based on thresholds."""
        # Arrange - Low scores that should fail
        mock_judge.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({
                                "faithfulness": {"score": 0.5, "reasoning": ""},
                                "answer_relevancy": {"score": 0.5, "reasoning": ""},
                                "contextual_precision": {"score": 0.5, "reasoning": ""},
                                "contextual_recall": {"score": 0.5, "reasoning": ""},
                            })
                        )
                    )
                ]
            )
        )

        # Act
        result = await mock_judge.evaluate(
            query="test",
            response="response",
            contexts=["context"],
        )

        # Assert
        assert result.passed is False
        assert len(result.failure_reasons) > 0

    @pytest.mark.anyio
    async def test_evaluate_critical_threshold_violation(self, mock_judge):
        """Document critical threshold violation detection."""
        # Arrange - Very low faithfulness (critical)
        mock_judge.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({
                                "faithfulness": {"score": 0.3, "reasoning": ""},
                                "answer_relevancy": {"score": 0.9, "reasoning": ""},
                                "contextual_precision": {"score": 0.9, "reasoning": ""},
                                "contextual_recall": {"score": 0.9, "reasoning": ""},
                            })
                        )
                    )
                ]
            )
        )

        # Act
        result = await mock_judge.evaluate(
            query="test",
            response="response",
            contexts=["context"],
        )

        # Assert - Should have critical failure reason
        assert any("CRITICAL" in reason for reason in result.failure_reasons)


class TestCustomLLMJudgeBatch:
    """Characterization tests for batch evaluation."""

    @pytest.fixture
    def mock_judge(self):
        """Create CustomLLMJudge with mocked OpenAI client."""
        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", True):
            with patch("src.rag.domain.evaluation.custom_judge.AsyncOpenAI") as mock_openai:
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client

                config = CustomJudgeConfig(api_key="test-key")
                judge = CustomLLMJudge(config=config)
                judge.client = mock_client
                return judge

    @pytest.mark.anyio
    async def test_evaluate_batch_returns_list(self, mock_judge):
        """Document batch evaluation returns list."""
        # Arrange
        mock_judge.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=json.dumps({
                                "faithfulness": {"score": 0.9, "reasoning": ""},
                                "answer_relevancy": {"score": 0.9, "reasoning": ""},
                                "contextual_precision": {"score": 0.9, "reasoning": ""},
                                "contextual_recall": {"score": 0.9, "reasoning": ""},
                            })
                        )
                    )
                ]
            )
        )

        test_cases = [
            {"query": "q1", "response": "r1", "contexts": ["c1"]},
            {"query": "q2", "response": "r2", "contexts": ["c2"]},
        ]

        # Act
        results = await mock_judge.evaluate_batch(test_cases)

        # Assert
        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.anyio
    async def test_evaluate_batch_handles_errors(self, mock_judge):
        """Document batch evaluation error handling."""
        # Arrange
        mock_judge.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        test_cases = [
            {"query": "q1", "response": "r1", "contexts": ["c1"]},
        ]

        # Act
        results = await mock_judge.evaluate_batch(test_cases)

        # Assert - Should return failed result instead of raising
        assert len(results) == 1
        assert results[0].passed is False
        assert len(results[0].failure_reasons) > 0


class TestCustomLLMJudgeFormatContexts:
    """Characterization tests for context formatting."""

    @pytest.fixture
    def mock_judge(self):
        """Create CustomLLMJudge with mocked client."""
        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", True):
            with patch("src.rag.domain.evaluation.custom_judge.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                config = CustomJudgeConfig(api_key="test-key")
                return CustomLLMJudge(config=config)

    def test_format_contexts_empty(self, mock_judge):
        """Document formatting empty contexts."""
        # Arrange
        contexts = []

        # Act
        formatted = mock_judge._format_contexts(contexts)

        # Assert
        assert formatted == "[No contexts provided]"

    def test_format_contexts_single(self, mock_judge):
        """Document formatting single context."""
        # Arrange
        contexts = ["Test context"]

        # Act
        formatted = mock_judge._format_contexts(contexts)

        # Assert
        assert "[Context 1]" in formatted
        assert "Test context" in formatted

    def test_format_contexts_multiple(self, mock_judge):
        """Document formatting multiple contexts."""
        # Arrange
        contexts = ["Context 1", "Context 2", "Context 3"]

        # Act
        formatted = mock_judge._format_contexts(contexts)

        # Assert
        assert "[Context 1]" in formatted
        assert "[Context 2]" in formatted
        assert "[Context 3]" in formatted

    def test_format_contexts_truncation(self, mock_judge):
        """Document context truncation for long contexts."""
        # Arrange
        long_context = "x" * 2000
        contexts = [long_context]

        # Act
        formatted = mock_judge._format_contexts(contexts)

        # Assert - Should be truncated
        assert len(formatted) < len(long_context)

    def test_format_contexts_limit(self, mock_judge):
        """Document context limit (max 10)."""
        # Arrange
        contexts = [f"Context {i}" for i in range(20)]

        # Act
        formatted = mock_judge._format_contexts(contexts)

        # Assert - Should only include first 10
        assert "[Context 10]" in formatted
        assert "[Context 11]" not in formatted


class TestCustomLLMJudgeDefaultEvaluation:
    """Characterization tests for default evaluation fallback."""

    @pytest.fixture
    def mock_judge(self):
        """Create CustomLLMJudge with mocked client."""
        with patch("src.rag.domain.evaluation.custom_judge.OPENAI_AVAILABLE", True):
            with patch("src.rag.domain.evaluation.custom_judge.AsyncOpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                config = CustomJudgeConfig(api_key="test-key")
                return CustomLLMJudge(config=config)

    def test_get_default_evaluation(self, mock_judge):
        """Document default evaluation scores."""
        # Act
        default = mock_judge._get_default_evaluation()

        # Assert
        assert default["faithfulness"]["score"] == 0.5
        assert default["answer_relevancy"]["score"] == 0.5
        assert default["contextual_precision"]["score"] == 0.5
        assert default["contextual_recall"]["score"] == 0.5
        assert "API call failed" in default["faithfulness"]["reasoning"]


class TestEvaluationPrompt:
    """Characterization tests for evaluation prompt template."""

    def test_prompt_contains_dimensions(self):
        """Document prompt contains evaluation dimensions."""
        # Assert
        assert "Faithfulness" in EVALUATION_PROMPT
        assert "Answer Relevancy" in EVALUATION_PROMPT
        assert "Contextual Precision" in EVALUATION_PROMPT
        assert "Contextual Recall" in EVALUATION_PROMPT

    def test_prompt_contains_weight_info(self):
        """Document prompt contains weight information."""
        # Assert
        assert "35%" in EVALUATION_PROMPT  # Faithfulness weight
        assert "25%" in EVALUATION_PROMPT  # Answer Relevancy weight
        assert "20%" in EVALUATION_PROMPT  # Contextual Precision weight
        assert "20%" in EVALUATION_PROMPT  # Contextual Recall weight

    def test_prompt_format_placeholders(self):
        """Document prompt format placeholders."""
        # Arrange
        formatted = EVALUATION_PROMPT.format(
            query="test query",
            contexts="test contexts",
            response="test response",
        )

        # Assert - Should not raise and placeholders should be filled
        assert "test query" in formatted
        assert "test contexts" in formatted
        assert "test response" in formatted
