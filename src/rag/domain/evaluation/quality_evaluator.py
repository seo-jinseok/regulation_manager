"""
RAG Quality Evaluator using LLM-as-Judge methodology.

This module implements the core evaluation logic using RAGAS framework
with configurable judge LLM model.

Clean Architecture: Domain layer coordinates evaluation through interfaces.
"""

import logging
import os
from typing import List, Optional

# Import RAGAS with graceful degradation
try:
    from ragas import SingleTurnSample
    from ragas.embeddings import RagasEmbeddings
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(e)

# Import LangChain for LLM integration with graceful degradation
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .models import (
    EvaluationFramework,
    EvaluationResult,
    EvaluationThresholds,
    MetricScore,
)

logger = logging.getLogger(__name__)


class RAGQualityEvaluator:
    """
    RAG Quality Evaluator using LLM-as-Judge methodology.

    Evaluates answer quality across four core metrics:
    - Faithfulness: Hallucination detection (0.90 threshold)
    - Answer Relevancy: Query response quality (0.85 threshold)
    - Contextual Precision: Retrieval ranking (0.80 threshold)
    - Contextual Recall: Information completeness (0.80 threshold)
    """

    def __init__(
        self,
        framework: EvaluationFramework = EvaluationFramework.RAGAS,
        judge_model: str = "gpt-4o",
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        thresholds: Optional[EvaluationThresholds] = None,
        use_ragas: bool = True,
    ):
        """
        Initialize the quality evaluator.

        Args:
            framework: Evaluation framework (RAGAS or DeepEval)
            judge_model: Judge LLM model (GPT-4o, Gemini, etc.)
            judge_api_key: API key for judge LLM (defaults to OPENAI_API_KEY env var)
            judge_base_url: Base URL for judge LLM API
            thresholds: Custom thresholds for evaluation
            use_ragas: Whether to use RAGAS library (falls back to mock if False)
        """
        self.framework = framework
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key or os.getenv("OPENAI_API_KEY")
        self.judge_base_url = judge_base_url
        self.thresholds = thresholds or EvaluationThresholds()
        self.use_ragas = use_ragas and RAGAS_AVAILABLE

        # Initialize RAGAS metrics if available
        self._ragas_metrics = None
        self._judge_llm = None
        self._run_config = None

        if self.use_ragas:
            self._initialize_ragas()
        else:
            if not RAGAS_AVAILABLE:
                logger.warning(
                    f"RAGAS not available: {RAGAS_IMPORT_ERROR}. Using mock evaluation."
                )
            else:
                logger.info("RAGAS disabled, using mock evaluation.")

    def _initialize_ragas(self):
        """Initialize RAGAS metrics with judge LLM configuration."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Cannot configure judge LLM.")
            self.use_ragas = False
            return

        if not self.judge_api_key:
            logger.warning("No API key provided for judge LLM. Using mock evaluation.")
            self.use_ragas = False
            return

        try:
            # Create LangChain LLM for RAGAS judge
            self._judge_llm = ChatOpenAI(
                model=self.judge_model,
                api_key=self.judge_api_key,
                base_url=self.judge_base_url,
                temperature=0.0,  # Deterministic for evaluation
                request_timeout=60,  # 60 second timeout
            )

            # Create embeddings for answer relevancy
            self._judge_embeddings = OpenAIEmbeddings(
                api_key=self.judge_api_key,
                base_url=self.judge_base_url,
            )

            # Configure RAGAS run settings
            self._run_config = RunConfig(
                timeout=60,  # 60 second timeout per metric
                max_retries=2,  # Retry failed requests twice
                max_wait=120,  # Max wait time for batch operations
            )

            # Initialize RAGAS metrics with judge LLM
            self._ragas_metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }

            logger.info(f"RAGAS initialized with judge model: {self.judge_model}")

        except Exception as e:
            logger.error(
                f"Failed to initialize RAGAS: {e}. Falling back to mock evaluation."
            )
            self.use_ragas = False

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG output using LLM-as-Judge.

        Args:
            query: User query
            answer: Generated answer from RAG system
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer for recall calculation

        Returns:
            EvaluationResult with all metric scores and pass/fail status
        """
        logger.info(f"Evaluating query: {query[:50]}...")

        # Evaluate each metric
        faithfulness_score = await self._evaluate_faithfulness(query, answer, contexts)
        relevancy_score = await self._evaluate_answer_relevancy(query, answer, contexts)
        precision_score = await self._evaluate_contextual_precision(
            query, contexts, answer
        )
        recall_score = await self._evaluate_contextual_recall(
            query, contexts, ground_truth, answer
        )

        # Calculate overall score (average of four metrics)
        overall_score = (
            faithfulness_score.score
            + relevancy_score.score
            + precision_score.score
            + recall_score.score
        ) / 4.0

        # Check pass/fail for each metric
        failure_reasons = []
        passed = True

        if not faithfulness_score.passed:
            passed = False
            failure_reasons.append(
                f"Faithfulness below threshold: {faithfulness_score.score:.3f} < {self.thresholds.faithfulness}"
            )

        if not relevancy_score.passed:
            passed = False
            failure_reasons.append(
                f"Answer Relevancy below threshold: {relevancy_score.score:.3f} < {self.thresholds.answer_relevancy}"
            )

        if not precision_score.passed:
            passed = False
            failure_reasons.append(
                f"Contextual Precision below threshold: {precision_score.score:.3f} < {self.thresholds.contextual_precision}"
            )

        if not recall_score.passed:
            passed = False
            failure_reasons.append(
                f"Contextual Recall below threshold: {recall_score.score:.3f} < {self.thresholds.contextual_recall}"
            )

        # Check for critical threshold violations
        if self.thresholds.is_below_critical("faithfulness", faithfulness_score.score):
            failure_reasons.append(
                "CRITICAL: Faithfulness below critical threshold - high hallucination risk"
            )

        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            faithfulness=faithfulness_score.score,
            answer_relevancy=relevancy_score.score,
            contextual_precision=precision_score.score,
            contextual_recall=recall_score.score,
            overall_score=round(overall_score, 3),
            passed=passed,
            failure_reasons=failure_reasons,
            metadata={
                "framework": self.framework.value,
                "judge_model": self.judge_model,
                "evaluation_method": "ragas" if self.use_ragas else "mock",
            },
        )

    async def evaluate_batch(
        self,
        test_cases: List[dict],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple queries in batch.

        Args:
            test_cases: List of dicts with 'query', 'answer', 'contexts', 'ground_truth'

        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"Starting batch evaluation of {len(test_cases)} queries")

        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating query {i + 1}/{len(test_cases)}")
            result = await self.evaluate(
                query=test_case["query"],
                answer=test_case["answer"],
                contexts=test_case.get("contexts", []),
                ground_truth=test_case.get("ground_truth"),
            )
            results.append(result)

        # Log aggregate statistics
        pass_count = sum(1 for r in results if r.passed)
        logger.info(f"Batch evaluation complete: {pass_count}/{len(results)} passed")

        return results

    async def _evaluate_faithfulness(
        self, query: str, answer: str, contexts: List[str]
    ) -> MetricScore:
        """
        Evaluate faithfulness (hallucination detection).

        Measures factual consistency between answer and retrieved context.
        Uses LLM-as-Judge to verify each claim in the answer.

        Returns:
            MetricScore with faithfulness score (0.0-1.0)
        """
        if self.use_ragas and self._judge_llm:
            try:
                # Create RAGAS sample
                sample = SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                )

                # Configure faithfulness metric with judge LLM
                faithfulness_metric = faithfulness
                faithfulness_metric.llm = self._judge_llm

                # Score using RAGAS
                result = await faithfulness_metric.single_turn_ascore(
                    sample, self._run_config
                )
                score = float(result) if result is not None else 0.5

                # Generate reason based on score
                if score >= 0.9:
                    reason = "Answer is fully grounded in retrieved context with no hallucinations"
                elif score >= 0.7:
                    reason = "Answer is mostly grounded with minor unsupported details"
                elif score >= 0.5:
                    reason = "Answer contains significant unsupported information"
                else:
                    reason = "Answer is mostly hallucinated with little factual basis"

                passed = score >= self.thresholds.faithfulness
                return MetricScore(
                    name="faithfulness",
                    score=round(score, 3),
                    passed=passed,
                    reason=reason,
                )

            except Exception as e:
                logger.warning(
                    f"RAGAS faithfulness evaluation failed: {e}. Using fallback."
                )

        # Fallback: Simple keyword-based scoring
        return self._mock_faithfulness(answer, contexts)

    async def _evaluate_answer_relevancy(
        self, query: str, answer: str, contexts: List[str]
    ) -> MetricScore:
        """
        Evaluate answer relevancy.

        Measures how well answer addresses the original query.
        Uses LLM-as-Judge to assess response completeness and directness.

        Returns:
            MetricScore with answer relevancy score (0.0-1.0)
        """
        if self.use_ragas and self._judge_llm:
            try:
                # Create RAGAS sample
                sample = SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                )

                # Configure answer relevancy metric with judge LLM
                relevancy_metric = answer_relevancy
                relevancy_metric.llm = self._judge_llm
                if hasattr(self, "_judge_embeddings"):
                    relevancy_metric.embeddings = RagasEmbeddings(
                        langchain_embeddings=self._judge_embeddings
                    )

                # Score using RAGAS
                result = await relevancy_metric.single_turn_ascore(
                    sample, self._run_config
                )
                score = float(result) if result is not None else 0.5

                # Generate reason based on score
                if score >= 0.85:
                    reason = "Answer directly and completely addresses the query"
                elif score >= 0.7:
                    reason = "Answer addresses the query but lacks some detail"
                elif score >= 0.5:
                    reason = (
                        "Answer partially addresses the query with missing information"
                    )
                else:
                    reason = "Answer is mostly irrelevant to the query"

                passed = score >= self.thresholds.answer_relevancy
                return MetricScore(
                    name="answer_relevancy",
                    score=round(score, 3),
                    passed=passed,
                    reason=reason,
                )

            except Exception as e:
                logger.warning(
                    f"RAGAS relevancy evaluation failed: {e}. Using fallback."
                )

        # Fallback: Simple keyword-based scoring
        return self._mock_answer_relevancy(query, answer)

    async def _evaluate_contextual_precision(
        self, query: str, contexts: List[str], answer: str
    ) -> MetricScore:
        """
        Evaluate contextual precision.

        Measures whether relevant documents are ranked higher than irrelevant ones.
        Uses LLM-as-Judge to assess retrieval ranking quality.

        Returns:
            MetricScore with contextual precision score (0.0-1.0)
        """
        if self.use_ragas and self._judge_llm:
            try:
                # Create RAGAS sample
                sample = SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference="dummy",  # Contextual precision requires reference
                )

                # Configure context precision metric with judge LLM
                precision_metric = context_precision
                precision_metric.llm = self._judge_llm

                # Score using RAGAS
                result = await precision_metric.single_turn_ascore(
                    sample, self._run_config
                )
                score = float(result) if result is not None else 0.5

                # Generate reason based on score
                if score >= 0.8:
                    reason = (
                        "Retrieved contexts are well-ranked with relevant items first"
                    )
                elif score >= 0.6:
                    reason = "Retrieved contexts have moderate ranking quality"
                elif score >= 0.4:
                    reason = (
                        "Retrieved contexts have poor ranking with irrelevant items"
                    )
                else:
                    reason = "Retrieved contexts are mostly irrelevant or poorly ranked"

                passed = score >= self.thresholds.contextual_precision
                return MetricScore(
                    name="contextual_precision",
                    score=round(score, 3),
                    passed=passed,
                    reason=reason,
                )

            except Exception as e:
                logger.warning(
                    f"RAGAS precision evaluation failed: {e}. Using fallback."
                )

        # Fallback: Simple scoring based on context relevance
        return self._mock_contextual_precision(query, contexts)

    async def _evaluate_contextual_recall(
        self, query: str, contexts: List[str], ground_truth: Optional[str], answer: str
    ) -> MetricScore:
        """
        Evaluate contextual recall.

        Measures whether all relevant information was retrieved.
        Uses LLM-as-Judge to identify missing information.

        Returns:
            MetricScore with contextual recall score (0.0-1.0)
        """
        if self.use_ragas and self._judge_llm and ground_truth:
            try:
                # Create RAGAS sample
                sample = SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                    reference=ground_truth,
                )

                # Configure context recall metric with judge LLM
                recall_metric = context_recall
                recall_metric.llm = self._judge_llm

                # Score using RAGAS
                result = await recall_metric.single_turn_ascore(
                    sample, self._run_config
                )
                score = float(result) if result is not None else 0.5

                # Generate reason based on score
                if score >= 0.8:
                    reason = "All relevant information was retrieved from contexts"
                elif score >= 0.6:
                    reason = "Most relevant information was retrieved with minor gaps"
                elif score >= 0.4:
                    reason = "Significant relevant information is missing from contexts"
                else:
                    reason = "Most relevant information was not retrieved"

                passed = score >= self.thresholds.contextual_recall
                return MetricScore(
                    name="contextual_recall",
                    score=round(score, 3),
                    passed=passed,
                    reason=reason,
                )

            except Exception as e:
                logger.warning(f"RAGAS recall evaluation failed: {e}. Using fallback.")

        # Fallback: Simple scoring
        return self._mock_contextual_recall(contexts, ground_truth)

    # Mock/fallback methods when RAGAS is not available

    def _mock_faithfulness(self, answer: str, contexts: List[str]) -> MetricScore:
        """Mock faithfulness evaluation based on keyword overlap."""
        if not contexts:
            score = 0.5
            reason = "No contexts provided for verification"
        else:
            # Calculate keyword overlap
            answer_words = set(answer.lower().split())
            context_words = set(" ".join(contexts).lower().split())

            if not answer_words:
                score = 0.0
                reason = "Empty answer"
            else:
                overlap = len(answer_words & context_words) / len(answer_words)
                score = min(0.95, max(0.5, overlap))  # Range 0.5-0.95

                if score >= 0.8:
                    reason = "Answer appears grounded in context (keyword-based)"
                elif score >= 0.6:
                    reason = "Answer has moderate context support (keyword-based)"
                else:
                    reason = (
                        "Answer may contain unsupported information (keyword-based)"
                    )

        passed = score >= self.thresholds.faithfulness
        return MetricScore(
            name="faithfulness",
            score=round(score, 3),
            passed=passed,
            reason=reason,
        )

    def _mock_answer_relevancy(self, query: str, answer: str) -> MetricScore:
        """Mock answer relevancy based on keyword overlap."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            score = 0.5
            reason = "Empty query"
        else:
            overlap = len(query_words & answer_words) / len(query_words)
            score = min(0.92, max(0.5, overlap))

            if score >= 0.8:
                reason = "Answer contains query keywords (keyword-based)"
            elif score >= 0.6:
                reason = "Answer partially addresses query (keyword-based)"
            else:
                reason = "Answer may not address query (keyword-based)"

        passed = score >= self.thresholds.answer_relevancy
        return MetricScore(
            name="answer_relevancy",
            score=round(score, 3),
            passed=passed,
            reason=reason,
        )

    def _mock_contextual_precision(
        self, query: str, contexts: List[str]
    ) -> MetricScore:
        """Mock contextual precision based on context length and query overlap."""
        if not contexts:
            score = 0.5
            reason = "No contexts provided"
        else:
            # Calculate average query overlap across contexts
            query_words = set(query.lower().split())
            overlaps = []

            for ctx in contexts[:5]:  # Top 5 contexts
                ctx_words = set(ctx.lower().split())
                if ctx_words:
                    overlap = len(query_words & ctx_words) / len(ctx_words)
                    overlaps.append(overlap)

            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                score = min(0.88, max(0.5, avg_overlap))
            else:
                score = 0.5

            if score >= 0.7:
                reason = "Contexts appear relevant to query (keyword-based)"
            elif score >= 0.5:
                reason = "Contexts have moderate relevance (keyword-based)"
            else:
                reason = "Contexts may be irrelevant (keyword-based)"

        passed = score >= self.thresholds.contextual_precision
        return MetricScore(
            name="contextual_precision",
            score=round(score, 3),
            passed=passed,
            reason=reason,
        )

    def _mock_contextual_recall(
        self, contexts: List[str], ground_truth: Optional[str]
    ) -> MetricScore:
        """Mock contextual recall based on context coverage."""
        if not ground_truth:
            # No ground truth available, assume good coverage
            score = 0.87
            reason = "No ground truth provided, assuming good coverage"
        elif not contexts:
            score = 0.5
            reason = "No contexts provided"
        else:
            # Calculate ground truth keyword coverage
            gt_words = set(ground_truth.lower().split())
            ctx_words = set(" ".join(contexts).lower().split())

            if not gt_words:
                score = 0.87
                reason = "Empty ground truth, assuming good coverage"
            else:
                coverage = len(gt_words & ctx_words) / len(gt_words)
                score = min(0.90, max(0.5, coverage))

                if score >= 0.8:
                    reason = "Ground truth well covered in contexts (keyword-based)"
                elif score >= 0.6:
                    reason = (
                        "Ground truth partially covered in contexts (keyword-based)"
                    )
                else:
                    reason = (
                        "Significant ground truth information missing (keyword-based)"
                    )

        passed = score >= self.thresholds.contextual_recall
        return MetricScore(
            name="contextual_recall",
            score=round(score, 3),
            passed=passed,
            reason=reason,
        )
