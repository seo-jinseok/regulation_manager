"""
Custom LLM-as-Judge Evaluation for RAG Quality Assessment.

This module implements a direct OpenAI API-based evaluation system for RAG quality.
Unlike RAGAS library, this system directly calls GPT-4o for evaluation, providing
meaningful scores with proper LLM reasoning.

Metrics Calculated:
- Faithfulness (35%): Is the response faithful to retrieved contexts? (No hallucinations)
- Answer Relevancy (25%): Is the response relevant to the query?
- Contextual Precision (20%): Are retrieved contexts relevant to query?
- Contextual Recall (20%): Does response use relevant information from contexts?

Clean Architecture: Domain layer with infrastructure dependency only for OpenAI client.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# OpenAI client import
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from .models import (
    EvaluationResult,
    EvaluationThresholds,
)

logger = logging.getLogger(__name__)

# Evaluation prompt template for LLM-as-Judge
EVALUATION_PROMPT = """You are an expert judge for RAG (Retrieval-Augmented Generation) system evaluation.

Evaluate the following RAG response on 4 dimensions:

**Query**: {query}

**Retrieved Contexts**:
{contexts}

**RAG Response**: {response}

For each dimension, provide:
1. Score (0.0-1.0): Where 1.0 is excellent, 0.5 is moderate, 0.0 is poor
2. Reasoning (2-3 sentences): Brief explanation for the score

**Evaluation Dimensions**:

1. **Faithfulness (35% weight)**: Is the response faithful to the retrieved contexts?
   - Does the response contain information NOT supported by the contexts? (hallucinations)
   - Are all claims in the response verifiable from the contexts?
   - Score 1.0: All claims supported by contexts
   - Score 0.5: Some claims unsupported or minor hallucinations
   - Score 0.0: Major hallucinations or response contradicts contexts

2. **Answer Relevancy (25% weight)**: Is the response relevant to the original query?
   - Does the response directly address the user's question?
   - Is the response complete and comprehensive?
   - Score 1.0: Direct, complete answer to the query
   - Score 0.5: Partially addresses the query
   - Score 0.0: Irrelevant or misses the query intent

3. **Contextual Precision (20% weight)**: Are the retrieved contexts relevant to the query?
   - Do the contexts contain information useful for answering the query?
   - Are the contexts ranked appropriately (most relevant first)?
   - Score 1.0: All contexts highly relevant
   - Score 0.5: Mixed relevance or poor ranking
   - Score 0.0: Contexts mostly irrelevant to query

4. **Contextual Recall (20% weight)**: Does the response use relevant information from the contexts?
   - Did the response extract key information from the contexts?
   - Is important information from contexts missing in the response?
   - Score 1.0: All relevant context information used
   - Score 0.5: Some relevant information missed
   - Score 0.0: Failed to extract relevant information

**Output Format**:
Respond ONLY with valid JSON in this exact format:
{{
    "faithfulness": {{"score": 0.0-1.0, "reasoning": "..."}},
    "answer_relevancy": {{"score": 0.0-1.0, "reasoning": "..."}},
    "contextual_precision": {{"score": 0.0-1.0, "reasoning": "..."}},
    "contextual_recall": {{"score": 0.0-1.0, "reasoning": "..."}}
}}
"""


@dataclass
class CustomJudgeConfig:
    """Configuration for custom LLM-as-Judge evaluation."""

    # OpenAI configuration
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0  # Deterministic for evaluation
    max_tokens: int = 1000
    timeout: int = 60

    # Evaluation weights
    faithfulness_weight: float = 0.35
    answer_relevancy_weight: float = 0.25
    contextual_precision_weight: float = 0.20
    contextual_recall_weight: float = 0.20

    # Retry configuration
    max_retries: int = 2
    retry_delay: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")

        # Validate weights sum to 1.0
        total_weight = (
            self.faithfulness_weight
            + self.answer_relevancy_weight
            + self.contextual_precision_weight
            + self.contextual_recall_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Evaluation weights sum to {total_weight}, should be 1.0. Normalizing."
            )
            # Normalize weights
            self.faithfulness_weight /= total_weight
            self.answer_relevancy_weight /= total_weight
            self.contextual_precision_weight /= total_weight
            self.contextual_recall_weight /= total_weight


@dataclass
class CustomEvaluationResult:
    """Result from custom LLM-as-Judge evaluation."""

    # Input data
    query: str
    response: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    # Metric scores with reasoning
    faithfulness_score: float = 0.0
    faithfulness_reasoning: str = ""
    answer_relevancy_score: float = 0.0
    answer_relevancy_reasoning: str = ""
    contextual_precision_score: float = 0.0
    contextual_precision_reasoning: str = ""
    contextual_recall_score: float = 0.0
    contextual_recall_reasoning: str = ""

    # Overall score (weighted average)
    overall_score: float = 0.0

    # Pass/fail status
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    # Metadata
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    judge_model: str = "gpt-4o"
    stage: int = 1

    # Raw response from LLM
    raw_evaluation: Optional[Dict[str, Any]] = None

    # Additional metadata for comparison (not required, added dynamically)
    scenario_id: str = ""
    persona: str = ""
    category: str = ""
    mock_scores: Dict[str, float] = field(default_factory=dict)

    def to_evaluation_result(
        self, thresholds: EvaluationThresholds
    ) -> EvaluationResult:
        """Convert to EvaluationResult for compatibility."""
        return EvaluationResult(
            query=self.query,
            answer=self.response,
            contexts=self.contexts,
            faithfulness=self.faithfulness_score,
            answer_relevancy=self.answer_relevancy_score,
            contextual_precision=self.contextual_precision_score,
            contextual_recall=self.contextual_recall_score,
            overall_score=self.overall_score,
            passed=self.passed,
            failure_reasons=self.failure_reasons,
            metadata={
                "framework": "custom_judge",
                "judge_model": self.judge_model,
                "evaluation_method": "llm_as_judge",
                "faithfulness_reasoning": self.faithfulness_reasoning,
                "answer_relevancy_reasoning": self.answer_relevancy_reasoning,
                "contextual_precision_reasoning": self.contextual_precision_reasoning,
                "contextual_recall_reasoning": self.contextual_recall_reasoning,
                "raw_evaluation": self.raw_evaluation,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "response": self.response,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "faithfulness": {
                "score": self.faithfulness_score,
                "reasoning": self.faithfulness_reasoning,
            },
            "answer_relevancy": {
                "score": self.answer_relevancy_score,
                "reasoning": self.answer_relevancy_reasoning,
            },
            "contextual_precision": {
                "score": self.contextual_precision_score,
                "reasoning": self.contextual_precision_reasoning,
            },
            "contextual_recall": {
                "score": self.contextual_recall_score,
                "reasoning": self.contextual_recall_reasoning,
            },
            "overall_score": self.overall_score,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "evaluation_timestamp": self.evaluation_timestamp,
            "judge_model": self.judge_model,
            "stage": self.stage,
            "raw_evaluation": self.raw_evaluation,
        }


class CustomLLMJudge:
    """
    Custom LLM-as-Judge for RAG quality evaluation.

    Uses OpenAI GPT-4o directly for evaluation, providing meaningful scores
    with proper LLM reasoning for each metric.
    """

    def __init__(
        self,
        config: Optional[CustomJudgeConfig] = None,
        thresholds: Optional[EvaluationThresholds] = None,
    ):
        """
        Initialize the custom LLM judge.

        Args:
            config: Configuration for the judge
            thresholds: Evaluation thresholds for pass/fail determination
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library is required for CustomLLMJudge. "
                "Install with: pip install openai"
            )

        self.config = config or CustomJudgeConfig()
        self.thresholds = thresholds or EvaluationThresholds.for_stage(stage=1)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

        logger.info(
            f"Custom LLM Judge initialized with model: {self.config.model}, "
            f"stage: {self.thresholds.stage}"
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        expected_answer: Optional[str] = None,
    ) -> CustomEvaluationResult:
        """
        Evaluate RAG response using LLM-as-Judge.

        Args:
            query: User query
            response: RAG system response
            contexts: Retrieved context documents
            expected_answer: Optional expected/ground truth answer

        Returns:
            CustomEvaluationResult with scores and reasoning
        """
        logger.info(f"Evaluating query: {query[:50]}...")

        # Prepare contexts text
        contexts_text = self._format_contexts(contexts)

        # Create evaluation prompt
        prompt = EVALUATION_PROMPT.format(
            query=query,
            contexts=contexts_text,
            response=response,
        )

        # Call OpenAI API with retry logic
        evaluation_data = await self._call_openai_with_retry(prompt)

        # Extract scores and reasoning
        faithfulness_data = evaluation_data.get("faithfulness", {})
        answer_relevancy_data = evaluation_data.get("answer_relevancy", {})
        contextual_precision_data = evaluation_data.get("contextual_precision", {})
        contextual_recall_data = evaluation_data.get("contextual_recall", {})

        faithfulness_score = faithfulness_data.get("score", 0.5)
        faithfulness_reasoning = faithfulness_data.get("reasoning", "")
        answer_relevancy_score = answer_relevancy_data.get("score", 0.5)
        answer_relevancy_reasoning = answer_relevancy_data.get("reasoning", "")
        contextual_precision_score = contextual_precision_data.get("score", 0.5)
        contextual_precision_reasoning = contextual_precision_data.get("reasoning", "")
        contextual_recall_score = contextual_recall_data.get("score", 0.5)
        contextual_recall_reasoning = contextual_recall_data.get("reasoning", "")

        # Calculate weighted overall score
        overall_score = (
            faithfulness_score * self.config.faithfulness_weight
            + answer_relevancy_score * self.config.answer_relevancy_weight
            + contextual_precision_score * self.config.contextual_precision_weight
            + contextual_recall_score * self.config.contextual_recall_weight
        )

        # Determine pass/fail
        failure_reasons = []
        passed = True

        if faithfulness_score < self.thresholds.faithfulness:
            passed = False
            failure_reasons.append(
                f"Faithfulness below threshold: {faithfulness_score:.3f} < {self.thresholds.faithfulness}"
            )

        if answer_relevancy_score < self.thresholds.answer_relevancy:
            passed = False
            failure_reasons.append(
                f"Answer Relevancy below threshold: {answer_relevancy_score:.3f} < {self.thresholds.answer_relevancy}"
            )

        if contextual_precision_score < self.thresholds.contextual_precision:
            passed = False
            failure_reasons.append(
                f"Contextual Precision below threshold: {contextual_precision_score:.3f} < {self.thresholds.contextual_precision}"
            )

        if contextual_recall_score < self.thresholds.contextual_recall:
            passed = False
            failure_reasons.append(
                f"Contextual Recall below threshold: {contextual_recall_score:.3f} < {self.thresholds.contextual_recall}"
            )

        # Check for critical threshold violations
        if self.thresholds.is_below_critical("faithfulness", faithfulness_score):
            failure_reasons.append(
                "CRITICAL: Faithfulness below critical threshold - high hallucination risk"
            )

        return CustomEvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            ground_truth=expected_answer,
            faithfulness_score=round(faithfulness_score, 3),
            faithfulness_reasoning=faithfulness_reasoning,
            answer_relevancy_score=round(answer_relevancy_score, 3),
            answer_relevancy_reasoning=answer_relevancy_reasoning,
            contextual_precision_score=round(contextual_precision_score, 3),
            contextual_precision_reasoning=contextual_precision_reasoning,
            contextual_recall_score=round(contextual_recall_score, 3),
            contextual_recall_reasoning=contextual_recall_reasoning,
            overall_score=round(overall_score, 3),
            passed=passed,
            failure_reasons=failure_reasons,
            judge_model=self.config.model,
            stage=self.thresholds.stage,
            raw_evaluation=evaluation_data,
        )

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> List[CustomEvaluationResult]:
        """
        Evaluate multiple test cases in batch.

        Args:
            test_cases: List of dicts with 'query', 'response', 'contexts', 'expected_answer'

        Returns:
            List of CustomEvaluationResult objects
        """
        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")

        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")
            try:
                result = await self.evaluate(
                    query=test_case["query"],
                    response=test_case["response"],
                    contexts=test_case.get("contexts", []),
                    expected_answer=test_case.get("expected_answer"),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating test case {i + 1}: {e}")
                # Create a failed result
                results.append(
                    CustomEvaluationResult(
                        query=test_case.get("query", ""),
                        response=test_case.get("response", ""),
                        contexts=test_case.get("contexts", []),
                        overall_score=0.0,
                        passed=False,
                        failure_reasons=[f"Evaluation error: {str(e)}"],
                    )
                )

        # Log aggregate statistics
        pass_count = sum(1 for r in results if r.passed)
        logger.info(
            f"Batch evaluation complete: {pass_count}/{len(results)} passed "
            f"({pass_count / len(results) * 100:.1f}%)"
        )

        return results

    def _format_contexts(self, contexts: List[str]) -> str:
        """Format contexts for the evaluation prompt."""
        if not contexts:
            return "[No contexts provided]"

        formatted = []
        for i, ctx in enumerate(contexts[:10], 1):  # Limit to top 10 contexts
            # Truncate very long contexts
            ctx_text = ctx[:1000] + "..." if len(ctx) > 1000 else ctx
            formatted.append(f"[Context {i}]: {ctx_text}")

        return "\n\n".join(formatted)

    async def _call_openai_with_retry(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic.

        Args:
            prompt: Evaluation prompt

        Returns:
            Parsed evaluation data as dictionary
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(f"Calling OpenAI API (attempt {attempt + 1})")

                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert RAG system evaluator. "
                            "Provide objective, thorough evaluations.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )

                # Parse response
                content = response.choices[0].message.content
                evaluation_data = json.loads(content)

                logger.debug("OpenAI API call successful")
                return evaluation_data

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                if attempt == self.config.max_retries:
                    # Return default scores on final failure
                    logger.warning("All retries failed, returning default scores")
                    return self._get_default_evaluation()
                await asyncio.sleep(self.config.retry_delay)

            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                if attempt == self.config.max_retries:
                    logger.warning("All retries failed, returning default scores")
                    return self._get_default_evaluation()
                await asyncio.sleep(self.config.retry_delay)

        return self._get_default_evaluation()

    def _get_default_evaluation(self) -> Dict[str, Any]:
        """Get default evaluation scores when API calls fail."""
        return {
            "faithfulness": {
                "score": 0.5,
                "reasoning": "API call failed, default score",
            },
            "answer_relevancy": {
                "score": 0.5,
                "reasoning": "API call failed, default score",
            },
            "contextual_precision": {
                "score": 0.5,
                "reasoning": "API call failed, default score",
            },
            "contextual_recall": {
                "score": 0.5,
                "reasoning": "API call failed, default score",
            },
        }


# Export main classes
__all__ = [
    "CustomLLMJudge",
    "CustomJudgeConfig",
    "CustomEvaluationResult",
]
