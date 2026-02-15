#!/usr/bin/env python3
"""
Integrated RAG Quality Evaluation with SPEC-RAG-QUALITY-003 Components.

This script integrates all Phase 1-5 improvements:
- Phase 1: Colloquial Query Transformation
- Phase 2: Korean Morphological Expansion
- Phase 3: Semantic Similarity Evaluation
- Phase 4: LLM-as-Judge Enhancement
- Phase 5: Hybrid Weight Optimization

Run as: uv run python scripts/run_integrated_quality_eval.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.domain.query.colloquial_transformer import (
    ColloquialTransformer,
    create_colloquial_transformer,
)
from src.rag.domain.query.morphological_expander import (
    MorphologicalExpander,
    ExpansionMode,
    create_morphological_expander,
)
from src.rag.infrastructure.evaluation.semantic_evaluator import (
    SemanticEvaluator,
    create_semantic_evaluator,
)
from src.rag.application.hybrid_weight_optimizer import (
    HybridWeightOptimizer,
    create_hybrid_weight_optimizer,
)
from src.rag.domain.evaluation import RAGQualityEvaluator, PersonaManager
from src.rag.domain.evaluation.models import EvaluationThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IntegratedQualityEvaluator:
    """
    Integrated RAG Quality Evaluator using all SPEC-RAG-QUALITY-003 components.
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        similarity_threshold: float = 0.75,
        use_ragas: bool = True,
    ):
        """Initialize all Phase 1-5 components."""
        logger.info("Initializing Integrated Quality Evaluator...")

        # Phase 1: Colloquial Transformer
        self.colloquial_transformer = create_colloquial_transformer()
        logger.info(
            f"Phase 1: ColloquialTransformer loaded ({self.colloquial_transformer.get_stats()['total_mappings']} patterns)"
        )

        # Phase 2: Morphological Expander
        self.morphological_expander = create_morphological_expander(mode="hybrid")
        logger.info(
            f"Phase 2: MorphologicalExpander loaded (mode: {self.morphological_expander.mode.value})"
        )

        # Phase 3: Semantic Evaluator
        self.semantic_evaluator = create_semantic_evaluator(
            similarity_threshold=similarity_threshold
        )
        logger.info(
            f"Phase 3: SemanticEvaluator loaded (threshold: {similarity_threshold})"
        )

        # Phase 5: Hybrid Weight Optimizer
        self.weight_optimizer = create_hybrid_weight_optimizer(
            colloquial_transformer=self.colloquial_transformer
        )
        logger.info("Phase 5: HybridWeightOptimizer loaded")

        # RAG Quality Evaluator (Phase 4 integration via LLM-as-Judge)
        self.quality_evaluator = RAGQualityEvaluator(
            judge_model=judge_model,
            use_ragas=use_ragas,
            stage=1,  # Use initial stage thresholds
        )
        logger.info(f"Phase 4: RAGQualityEvaluator loaded (model: {judge_model})")

        # Persona Manager
        self.persona_manager = PersonaManager()
        logger.info(
            f"PersonaManager loaded ({len(self.persona_manager.list_personas())} personas)"
        )

        # Results storage
        self.results: Dict[str, Any] = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "spec": "SPEC-RAG-QUALITY-003",
                "phases_implemented": [1, 2, 3, 4, 5],
                "judge_model": judge_model,
                "similarity_threshold": similarity_threshold,
            },
            "per_query": [],
            "per_persona": {},
            "aggregates": {},
        }

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Apply Phase 1 (Colloquial Transformation) and Phase 2 (Morphological Expansion).

        Returns preprocessing details for logging.
        """
        preprocessing = {
            "original_query": query,
        }

        # Phase 1: Colloquial Transformation
        transform_result = self.colloquial_transformer.transform(query)
        preprocessing["colloquial_transform"] = {
            "was_transformed": transform_result.was_transformed,
            "transformed_query": transform_result.transformed_query,
            "patterns_matched": transform_result.patterns_matched,
            "confidence": transform_result.confidence,
        }

        # Phase 2: Morphological Expansion
        morph_result = self.morphological_expander.expand(transform_result.transformed_query)
        preprocessing["morphological_expansion"] = {
            "nouns": morph_result.nouns,
            "verbs": morph_result.verbs,
            "expanded_terms": morph_result.expanded_terms,
            "final_expanded": morph_result.final_expanded,
        }

        # Phase 5: Weight Optimization
        weight_decision = self.weight_optimizer.optimize(query)
        preprocessing["weight_optimization"] = {
            "bm25_weight": weight_decision.bm25_weight,
            "vector_weight": weight_decision.vector_weight,
            "is_colloquial": weight_decision.is_colloquial,
            "formality_score": weight_decision.formality_score,
        }

        preprocessing["final_query"] = morph_result.final_expanded

        return preprocessing

    def evaluate_single_query(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        expected_answer: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate a single query using all integrated components.
        """
        # Preprocess query
        preprocessing = self.preprocess_query(query)

        # Phase 3: Semantic Similarity Evaluation
        semantic_result = None
        if expected_answer:
            semantic_result = self.semantic_evaluator.evaluate_with_query(
                query=query,
                answer=answer,
                expected=expected_answer,
            )

        # Phase 4: RAG Quality Evaluation (using RAGAS/LLM-as-Judge)
        quality_result = self.quality_evaluator.evaluate_single_turn(
            query=query,
            contexts=contexts,
            answer=answer,
        )

        # Combine results
        result = {
            "query": query,
            "preprocessing": preprocessing,
            "answer": answer,
            "contexts": contexts,
            "quality_metrics": {
                "faithfulness": quality_result.faithfulness,
                "answer_relevancy": quality_result.answer_relevancy,
                "contextual_precision": quality_result.contextual_precision,
                "contextual_recall": quality_result.contextual_recall,
                "overall_score": quality_result.overall_score,
                "passed": quality_result.passed,
            },
        }

        # Add semantic evaluation if available
        if semantic_result:
            result["semantic_evaluation"] = {
                "similarity_score": semantic_result.similarity_score,
                "is_relevant": semantic_result.is_relevant,
                "threshold": semantic_result.threshold,
            }

        return result

    def evaluate_persona(
        self,
        persona_id: str,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate all queries for a single persona.
        """
        logger.info(f"Evaluating persona: {persona_id} ({len(queries)} queries)")

        persona_results = []
        passed_count = 0

        for i, query_data in enumerate(queries, 1):
            query = query_data.get("query", "")
            answer = query_data.get("answer", "No answer generated")
            contexts = query_data.get("contexts", [])
            expected = query_data.get("expected", "")

            logger.info(f"  Query {i}/{len(queries)}: {query[:40]}...")

            result = self.evaluate_single_query(
                query=query,
                answer=answer,
                contexts=contexts,
                expected_answer=expected,
            )
            result["persona"] = persona_id
            result["category"] = query_data.get("category", "general")
            result["difficulty"] = query_data.get("difficulty", "medium")

            persona_results.append(result)
            self.results["per_query"].append(result)

            if result["quality_metrics"]["passed"]:
                passed_count += 1

        # Calculate persona aggregates
        avg_scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "contextual_precision": 0.0,
            "contextual_recall": 0.0,
            "overall_score": 0.0,
        }

        for r in persona_results:
            for key in avg_scores:
                avg_scores[key] += r["quality_metrics"].get(key, 0.0)

        for key in avg_scores:
            avg_scores[key] /= len(persona_results) if persona_results else 1

        persona_aggregate = {
            "persona": persona_id,
            "total_queries": len(queries),
            "passed": passed_count,
            "failed": len(queries) - passed_count,
            "pass_rate": passed_count / len(queries) if queries else 0.0,
            "avg_scores": avg_scores,
            "colloquial_transform_count": sum(
                1
                for r in persona_results
                if r["preprocessing"]["colloquial_transform"]["was_transformed"]
            ),
            "avg_formality_score": sum(
                r["preprocessing"]["weight_optimization"]["formality_score"]
                for r in persona_results
            )
            / len(persona_results)
            if persona_results
            else 0.0,
        }

        self.results["per_persona"][persona_id] = persona_aggregate
        return persona_aggregate

    def run_full_evaluation(
        self,
        queries_per_persona: int = 5,
        personas: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full evaluation across all personas.
        """
        if personas is None:
            personas = self.persona_manager.list_personas()

        logger.info("=" * 80)
        logger.info("Starting Integrated RAG Quality Evaluation")
        logger.info("=" * 80)
        logger.info(f"Personas: {', '.join(personas)}")
        logger.info(f"Queries per persona: {queries_per_persona}")
        logger.info(f"Total queries: {len(personas) * queries_per_persona}")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Generate queries for each persona
        for persona_id in personas:
            queries = self.persona_manager.generate_queries(
                persona_name=persona_id,
                count=queries_per_persona,
            )

            # For testing, use mock answers (in production, these would come from RAG system)
            mock_queries = []
            for q in queries:
                mock_queries.append({
                    "query": q,
                    "answer": "This is a mock answer for testing.",
                    "contexts": ["Mock context for testing."],
                    "expected": "Expected answer for comparison.",
                    "category": "general",
                    "difficulty": "medium",
                })

            self.evaluate_persona(persona_id, mock_queries)

        # Calculate overall aggregates
        self._calculate_aggregates()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.results["metadata"]["duration_seconds"] = duration
        self.results["metadata"]["end_time"] = end_time.isoformat()

        logger.info("=" * 80)
        logger.info(f"Evaluation completed in {duration:.2f} seconds")
        logger.info("=" * 80)

        return self.results

    def _calculate_aggregates(self) -> None:
        """Calculate overall aggregate statistics."""
        all_results = self.results["per_query"]

        if not all_results:
            return

        total_queries = len(all_results)
        passed_count = sum(1 for r in all_results if r["quality_metrics"]["passed"])

        # Calculate metric averages
        metric_sums = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "contextual_precision": 0.0,
            "contextual_recall": 0.0,
            "overall_score": 0.0,
        }

        for r in all_results:
            for key in metric_sums:
                metric_sums[key] += r["quality_metrics"].get(key, 0.0)

        metric_avgs = {
            key: val / total_queries for key, val in metric_sums.items()
        }

        # Calculate transformation statistics
        colloquial_transforms = sum(
            1
            for r in all_results
            if r["preprocessing"]["colloquial_transform"]["was_transformed"]
        )

        avg_formality = sum(
            r["preprocessing"]["weight_optimization"]["formality_score"]
            for r in all_results
        ) / total_queries

        self.results["aggregates"] = {
            "total_queries": total_queries,
            "passed": passed_count,
            "failed": total_queries - passed_count,
            "pass_rate": passed_count / total_queries,
            "metric_averages": metric_avgs,
            "colloquial_transforms": colloquial_transforms,
            "colloquial_transform_rate": colloquial_transforms / total_queries,
            "avg_formality_score": avg_formality,
        }

    def save_results(self, output_dir: str = "data/evaluations") -> str:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_eval_SPEC-RAG-QUALITY-003_{timestamp}.json"
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to: {filepath}")
        return str(filepath)

    def generate_comparison_report(
        self,
        previous_results: Dict[str, Any] = None,
    ) -> str:
        """
        Generate a comparison report between current and previous evaluation.
        """
        report_lines = []

        report_lines.append("# RAG Quality Re-Evaluation Report")
        report_lines.append("")
        report_lines.append(f"**SPEC:** SPEC-RAG-QUALITY-003")
        report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Phases Implemented:** 1, 2, 3, 4, 5")
        report_lines.append("")

        # Current Results
        report_lines.append("## Current Evaluation Results")
        report_lines.append("")

        agg = self.results.get("aggregates", {})

        report_lines.append("### Overall Metrics")
        report_lines.append("")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|-------|")
        report_lines.append(f"| Total Queries | {agg.get('total_queries', 0)} |")
        report_lines.append(f"| Passed | {agg.get('passed', 0)} |")
        report_lines.append(f"| Failed | {agg.get('failed', 0)} |")
        report_lines.append(f"| **Pass Rate** | **{agg.get('pass_rate', 0):.1%}** |")
        report_lines.append("")

        report_lines.append("### Metric Averages")
        report_lines.append("")
        report_lines.append(f"| Metric | Score | Threshold | Status |")
        report_lines.append(f"|--------|-------|-----------|--------|")

        thresholds = {"faithfulness": 0.60, "answer_relevancy": 0.70,
                      "contextual_precision": 0.65, "contextual_recall": 0.65}

        for metric, threshold in thresholds.items():
            score = agg.get("metric_averages", {}).get(metric, 0.0)
            status = "PASS" if score >= threshold else "FAIL"
            report_lines.append(f"| {metric.replace('_', ' ').title()} | {score:.3f} | {threshold} | {status} |")

        overall = agg.get("metric_averages", {}).get("overall_score", 0.0)
        overall_status = "PASS" if overall >= 0.70 else "FAIL"
        report_lines.append(f"| **Overall Score** | **{overall:.3f}** | 0.70 | **{overall_status}** |")
        report_lines.append("")

        # SPEC-RAG-QUALITY-003 Components
        report_lines.append("### SPEC-RAG-QUALITY-003 Component Statistics")
        report_lines.append("")
        report_lines.append(f"| Component | Statistic | Value |")
        report_lines.append(f"|-----------|-----------|-------|")
        report_lines.append(f"| Phase 1: Colloquial Transform | Transform Rate | {agg.get('colloquial_transform_rate', 0):.1%} |")
        report_lines.append(f"| Phase 5: Weight Optimization | Avg Formality Score | {agg.get('avg_formality_score', 0):.2f} |")
        report_lines.append(f"| Phase 1: Transformer Stats | Pattern Count | {self.colloquial_transformer.get_stats()['total_mappings']} |")
        report_lines.append(f"| Phase 2: Expander Stats | Mode | {self.morphological_expander.mode.value} |")
        report_lines.append(f"| Phase 3: Evaluator Stats | Similarity Threshold | {self.semantic_evaluator._similarity_threshold} |")
        report_lines.append("")

        # Comparison with Previous Results
        if previous_results:
            report_lines.append("## Comparison with Previous Evaluation")
            report_lines.append("")

            prev_agg = previous_results.get("aggregates", previous_results)

            report_lines.append(f"| Metric | Previous | Current | Change |")
            report_lines.append(f"|--------|----------|---------|--------|")

            prev_pass_rate = prev_agg.get("pass_rate", prev_agg.get("overall_pass_rate", 0))
            curr_pass_rate = agg.get("pass_rate", 0)
            pass_rate_change = curr_pass_rate - prev_pass_rate
            report_lines.append(f"| Pass Rate | {prev_pass_rate:.1%} | {curr_pass_rate:.1%} | {pass_rate_change:+.1%} |")

            prev_overall = prev_agg.get("overall_score", prev_agg.get("metric_averages", {}).get("overall_score", 0.502))
            curr_overall = agg.get("metric_averages", {}).get("overall_score", 0)
            overall_change = curr_overall - prev_overall
            report_lines.append(f"| Overall Score | {prev_overall:.3f} | {curr_overall:.3f} | {overall_change:+.3f} |")

            for metric in ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]:
                prev_val = prev_agg.get(metric, prev_agg.get("metric_averages", {}).get(metric, 0.5))
                curr_val = agg.get("metric_averages", {}).get(metric, 0)
                change = curr_val - prev_val
                report_lines.append(f"| {metric.replace('_', ' ').title()} | {prev_val:.3f} | {curr_val:.3f} | {change:+.3f} |")

            report_lines.append("")

        # Per-Persona Breakdown
        report_lines.append("## Per-Persona Breakdown")
        report_lines.append("")
        report_lines.append(f"| Persona | Queries | Pass Rate | Avg Score | Colloquial Rate |")
        report_lines.append(f"|---------|---------|-----------|-----------|-----------------|")

        for persona_id, persona_data in sorted(self.results["per_persona"].items()):
            pass_rate = persona_data.get("pass_rate", 0)
            avg_score = persona_data.get("avg_scores", {}).get("overall_score", 0)
            colloquial_count = persona_data.get("colloquial_transform_count", 0)
            total = persona_data.get("total_queries", 1)
            colloquial_rate = colloquial_count / total if total > 0 else 0
            report_lines.append(
                f"| {persona_id} | {total} | {pass_rate:.1%} | {avg_score:.3f} | {colloquial_rate:.1%} |"
            )

        report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")

        if agg.get("pass_rate", 0) >= 0.80:
            report_lines.append("SUCCESS: Pass rate target (80%) achieved!")
        else:
            report_lines.append(f"IMPROVEMENT NEEDED: Pass rate ({agg.get('pass_rate', 0):.1%}) below target (80%)")

        for metric, threshold in thresholds.items():
            score = agg.get("metric_averages", {}).get(metric, 0.0)
            if score < threshold:
                report_lines.append(f"- {metric.replace('_', ' ').title()}: {score:.3f} < {threshold} (NEEDS IMPROVEMENT)")

        report_lines.append("")

        return "\n".join(report_lines)


def main():
    """Run the integrated quality evaluation."""
    # Previous evaluation results for comparison
    previous_results = {
        "overall_pass_rate": 0.0,
        "overall_score": 0.502,
        "faithfulness": 0.500,
        "answer_relevancy": 0.501,
        "contextual_precision": 0.505,
        "contextual_recall": 0.501,
    }

    # Initialize integrated evaluator
    evaluator = IntegratedQualityEvaluator(
        judge_model="gpt-4o",
        similarity_threshold=0.75,
        use_ragas=True,
    )

    # Run full evaluation
    results = evaluator.run_full_evaluation(
        queries_per_persona=5,
        personas=[
            "freshman",
            "graduate",
            "professor",
            "staff",
            "parent",
            "international",
        ],
    )

    # Save results
    json_path = evaluator.save_results()

    # Generate comparison report
    report = evaluator.generate_comparison_report(previous_results)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"data/evaluations/comparison_report_SPEC-RAG-QUALITY-003_{timestamp}.md"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print report
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    print(f"\nResults saved to: {json_path}")
    print(f"Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
