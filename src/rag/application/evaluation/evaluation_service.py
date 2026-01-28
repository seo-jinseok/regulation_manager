"""
Evaluation Service - Orchestrates RAG quality evaluation workflow.

Application layer use case that coordinates evaluation components.

Clean Architecture: Application layer orchestrates domain and infrastructure.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from ...domain.evaluation import (
    EvaluationResult,
    EvaluationSummary,
    PersonaManager,
    PersonaMetrics,
    QualityAnalyzer,
    RAGQualityEvaluator,
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Service for orchestrating RAG quality evaluation.

    Coordinates evaluation across multiple personas, generates reports,
    and manages the evaluation workflow.
    """

    def __init__(
        self,
        evaluator: RAGQualityEvaluator,
        persona_manager: Optional[PersonaManager] = None,
        quality_analyzer: Optional[QualityAnalyzer] = None,
    ):
        """
        Initialize the evaluation service.

        Args:
            evaluator: RAG quality evaluator instance
            persona_manager: Optional persona manager for persona-based evaluation
            quality_analyzer: Optional quality analyzer for feedback loop
        """
        self.evaluator = evaluator
        self.persona_manager = persona_manager or PersonaManager()
        self.quality_analyzer = quality_analyzer or QualityAnalyzer()
        logger.info("Initialized EvaluationService")

    async def evaluate_single_query(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single query-answer pair.

        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth for recall calculation

        Returns:
            EvaluationResult with all metric scores
        """
        logger.info(f"Evaluating single query: {query[:50]}...")

        result = await self.evaluator.evaluate(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        return result

    async def evaluate_persona(
        self,
        persona_name: str,
        rag_pipeline,  # Callable that processes queries
        queries_per_persona: int = 20,
    ) -> PersonaMetrics:
        """
        Evaluate RAG quality for a specific persona.

        Args:
            persona_name: Name of the persona to evaluate
            rag_pipeline: RAG pipeline callable that takes query and returns (answer, contexts)
            queries_per_persona: Number of queries to generate and test

        Returns:
            PersonaMetrics with persona-specific evaluation results
        """
        logger.info(f"Evaluating persona: {persona_name}")

        # Generate queries for persona
        queries = self.persona_manager.generate_queries(
            persona_name=persona_name,
            count=queries_per_persona,
        )

        # Evaluate each query
        results = []
        for query in queries:
            try:
                answer, contexts = await rag_pipeline.search(query)
                result = await self.evaluator.evaluate(query, answer, contexts)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")

        # Calculate aggregate metrics
        if not results:
            logger.warning(f"No results for persona {persona_name}")
            return PersonaMetrics(
                persona_name=persona_name,
                faithfulness=0.0,
                answer_relevancy=0.0,
                contextual_precision=0.0,
                contextual_recall=0.0,
                overall_score=0.0,
                query_count=0,
                pass_rate=0.0,
            )

        faithfulness_avg = sum(r.faithfulness for r in results) / len(results)
        relevancy_avg = sum(r.answer_relevancy for r in results) / len(results)
        precision_avg = sum(r.contextual_precision for r in results) / len(results)
        recall_avg = sum(r.contextual_recall for r in results) / len(results)
        overall_avg = sum(r.overall_score for r in results) / len(results)
        pass_count = sum(1 for r in results if r.passed)
        pass_rate = pass_count / len(results)

        metrics = PersonaMetrics(
            persona_name=persona_name,
            faithfulness=round(faithfulness_avg, 3),
            answer_relevancy=round(relevancy_avg, 3),
            contextual_precision=round(precision_avg, 3),
            contextual_recall=round(recall_avg, 3),
            overall_score=round(overall_avg, 3),
            query_count=len(results),
            pass_rate=round(pass_rate, 3),
        )

        logger.info(
            f"Persona {persona_name}: Overall={metrics.overall_score:.3f}, Pass Rate={metrics.pass_rate:.3f}"
        )

        return metrics

    async def evaluate_all_personas(
        self,
        rag_pipeline,  # Callable that processes queries
        queries_per_persona: int = 20,
    ) -> Dict[str, PersonaMetrics]:
        """
        Evaluate RAG quality across all personas.

        Args:
            rag_pipeline: RAG pipeline callable
            queries_per_persona: Number of queries per persona

        Returns:
            Dictionary mapping persona names to their metrics
        """
        logger.info("Starting evaluation across all personas")

        personas = self.persona_manager.list_personas()
        results = {}

        for persona_name in personas:
            try:
                metrics = await self.evaluate_persona(
                    persona_name=persona_name,
                    rag_pipeline=rag_pipeline,
                    queries_per_persona=queries_per_persona,
                )
                results[persona_name] = metrics
            except Exception as e:
                logger.error(f"Error evaluating persona {persona_name}: {e}")

        # Calculate aggregate statistics
        avg_overall = sum(m.overall_score for m in results.values()) / len(results)
        logger.info(
            f"All personas evaluation complete. Avg overall score: {avg_overall:.3f}"
        )

        return results

    async def evaluate_batch(
        self,
        test_cases: List[dict],
    ) -> EvaluationSummary:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts with 'query', 'answer', 'contexts', 'ground_truth'

        Returns:
            EvaluationSummary with aggregate statistics
        """
        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")

        results = await self.evaluator.evaluate_batch(test_cases)

        # Calculate summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        summary = EvaluationSummary(
            total_queries=total,
            passed_queries=passed,
            pass_rate=round(passed / total, 3) if total > 0 else 0.0,
            avg_faithfulness=round(sum(r.faithfulness for r in results) / total, 3)
            if total > 0
            else 0.0,
            avg_answer_relevancy=round(
                sum(r.answer_relevancy for r in results) / total, 3
            )
            if total > 0
            else 0.0,
            avg_contextual_precision=round(
                sum(r.contextual_precision for r in results) / total, 3
            )
            if total > 0
            else 0.0,
            avg_contextual_recall=round(
                sum(r.contextual_recall for r in results) / total, 3
            )
            if total > 0
            else 0.0,
            avg_overall_score=round(sum(r.overall_score for r in results) / total, 3)
            if total > 0
            else 0.0,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Batch evaluation complete: {summary.passed_queries}/{summary.total_queries} passed"
        )

        return summary

    async def analyze_and_suggest_improvements(
        self,
        results: List[EvaluationResult],
    ) -> List[dict]:
        """
        Analyze evaluation results and generate improvement suggestions.

        Args:
            results: List of evaluation results

        Returns:
            List of improvement suggestions as dictionaries
        """
        logger.info(f"Analyzing {len(results)} results for improvements")

        suggestions = self.quality_analyzer.analyze_failures(results)

        # Convert to dictionaries for easier serialization
        suggestion_dicts = [s.to_dict() for s in suggestions]

        logger.info(f"Generated {len(suggestion_dicts)} improvement suggestions")

        return suggestion_dicts

    def generate_evaluation_report(
        self,
        summary: EvaluationSummary,
        persona_metrics: Optional[Dict[str, PersonaMetrics]] = None,
        suggestions: Optional[List[dict]] = None,
    ) -> dict:
        """
        Generate comprehensive evaluation report.

        Args:
            summary: Evaluation summary
            persona_metrics: Optional persona-specific metrics
            suggestions: Optional improvement suggestions

        Returns:
            Dictionary containing full evaluation report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary.to_dict(),
            "quality_gates_met": self._check_quality_gates(summary),
        }

        if persona_metrics:
            report["persona_breakdown"] = {
                name: metrics.to_dict() for name, metrics in persona_metrics.items()
            }

        if suggestions:
            report["improvement_suggestions"] = suggestions

        return report

    def _check_quality_gates(self, summary: EvaluationSummary) -> bool:
        """
        Check if quality gates are met.

        Args:
            summary: Evaluation summary

        Returns:
            True if all quality gates are passed
        """
        gates_passed = (
            summary.avg_faithfulness >= 0.90
            and summary.avg_answer_relevancy >= 0.85
            and summary.avg_contextual_precision >= 0.80
            and summary.avg_contextual_recall >= 0.80
        )

        logger.info(f"Quality gates {'passed' if gates_passed else 'failed'}")

        return gates_passed
