#!/usr/bin/env python3
"""
Custom LLM-as-Judge Batch Evaluation Script.

This script evaluates RAG quality using a custom LLM-as-Judge implementation
that directly calls OpenAI API (GPT-4o) for evaluation.

Features:
- Loads pilot test results from JSON files
- Evaluates each scenario with custom LLM-as-Judge
- Generates comparison report (Mock vs Custom Judge)
- Calculates pass/fail per stage thresholds

Usage:
    python scripts/custom_judge_evaluation.py [--input FILE] [--output DIR]

Output:
- JSON results: data/evaluations/custom_judge_results_*.json
- Markdown report: data/evaluations/custom_judge_report_*.md
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.domain.evaluation import (
    CustomEvaluationResult,
    CustomJudgeConfig,
    CustomLLMJudge,
    EvaluationThresholds,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


class CustomJudgeBatchEvaluator:
    """
    Batch evaluator using Custom LLM-as-Judge.

    Evaluates RAG responses from pilot test results and generates
    comprehensive comparison reports.
    """

    def __init__(
        self,
        output_dir: str = "data/evaluations",
        stage: int = 1,
        model: str = "gpt-4o",
    ):
        """
        Initialize the batch evaluator.

        Args:
            output_dir: Directory for output files
            stage: Evaluation stage (1=initial, 2=intermediate, 3=target)
            model: OpenAI model to use for evaluation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.model = model

        # Initialize custom judge
        config = CustomJudgeConfig(model=model)
        thresholds = EvaluationThresholds.for_stage(stage=stage)

        self.judge = CustomLLMJudge(config=config, thresholds=thresholds)

        logger.info(
            f"Custom LLM Judge initialized: model={model}, stage={stage}, "
            f"thresholds={thresholds.get_current_stage_name()}"
        )

        # Storage for results
        self.evaluation_results: List[CustomEvaluationResult] = []
        self.comparison_data: List[Dict[str, Any]] = []

    async def load_pilot_test_results(self, input_file: str) -> Dict[str, Any]:
        """
        Load pilot test results from JSON file.

        Args:
            input_file: Path to pilot test JSON file

        Returns:
            Dictionary with pilot test data
        """
        logger.info(f"Loading pilot test results from: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            f"Loaded {data.get('summary', {}).get('total_scenarios', 0)} scenarios"
        )
        return data

    async def extract_test_cases(
        self, pilot_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract test cases from pilot test data.

        Args:
            pilot_data: Pilot test results dictionary

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        # Extract individual results from pilot data
        # The structure uses "results" array where each item has:
        # - persona, scenario_index, query, category, ground_truth, answer
        # - sources (array of contexts with text field)
        # - evaluation (mock scores)
        if "results" in pilot_data:
            scenarios = pilot_data["results"]
        elif "scenarios" in pilot_data:
            scenarios = pilot_data["scenarios"]
        else:
            logger.warning("No 'results' or 'scenarios' key found in pilot data")
            return []

        for scenario in scenarios:
            # Extract contexts from sources (use the 'text' field from each source)
            sources = scenario.get("sources", [])
            contexts = [
                source.get("text", "") for source in sources if source.get("text")
            ]

            # Extract mock evaluation scores
            evaluation = scenario.get("evaluation", {})
            mock_scores = {
                "faithfulness": evaluation.get("faithfulness", 0.5),
                "answer_relevancy": evaluation.get("answer_relevancy", 0.5),
                "contextual_precision": evaluation.get("contextual_precision", 0.5),
                "contextual_recall": evaluation.get("contextual_recall", 0.5),
                "overall": evaluation.get("overall_score", 0.5),
            }

            # Create scenario ID from persona and index
            persona = scenario.get("persona", "")
            scenario_index = scenario.get("scenario_index", "")
            scenario_id = (
                f"{persona}_{scenario_index:03d}" if persona and scenario_index else ""
            )

            test_case = {
                "scenario_id": scenario_id,
                "query": scenario.get("query", ""),
                "response": scenario.get(
                    "answer", ""
                ),  # Note: field is "answer" not "response"
                "contexts": contexts,  # Extracted from sources
                "expected_answer": scenario.get("ground_truth", ""),
                "persona": persona,
                "category": scenario.get("category", ""),
                "mock_scores": mock_scores,
            }
            test_cases.append(test_case)

        logger.info(f"Extracted {len(test_cases)} test cases")
        return test_cases

    async def run_evaluation(
        self, test_cases: List[Dict[str, Any]]
    ) -> List[CustomEvaluationResult]:
        """
        Run evaluation on all test cases.

        Args:
            test_cases: List of test case dictionaries

        Returns:
            List of CustomEvaluationResult objects
        """
        logger.info(f"Starting evaluation of {len(test_cases)} test cases")

        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(
                f"[{i + 1}/{len(test_cases)}] Evaluating: {test_case['query'][:50]}..."
            )

            try:
                result = await self.judge.evaluate(
                    query=test_case["query"],
                    response=test_case["response"],
                    contexts=test_case["contexts"],
                    expected_answer=test_case.get("expected_answer"),
                )

                # Add metadata
                result.scenario_id = test_case.get("scenario_id", "")
                result.persona = test_case.get("persona", "")
                result.category = test_case.get("category", "")
                result.mock_scores = test_case.get("mock_scores", {})

                results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating test case {i + 1}: {e}")
                # Create a failed result
                result = CustomEvaluationResult(
                    query=test_case.get("query", ""),
                    response=test_case.get("response", ""),
                    contexts=test_case.get("contexts", []),
                    overall_score=0.0,
                    passed=False,
                    failure_reasons=[f"Evaluation error: {str(e)}"],
                )
                results.append(result)

        logger.info(f"Evaluation complete: {len(results)} results")
        return results

    def generate_comparison_data(
        self, results: List[CustomEvaluationResult]
    ) -> List[Dict[str, Any]]:
        """
        Generate comparison data between mock and custom judge scores.

        Args:
            results: List of CustomEvaluationResult objects

        Returns:
            List of comparison dictionaries
        """
        comparison_data = []

        for result in results:
            mock_scores = getattr(result, "mock_scores", {})
            comparison = {
                "scenario_id": getattr(result, "scenario_id", ""),
                "query": result.query,
                "persona": getattr(result, "persona", ""),
                "category": getattr(result, "category", ""),
                "mock_scores": mock_scores,
                "custom_judge_scores": {
                    "faithfulness": result.faithfulness_score,
                    "answer_relevancy": result.answer_relevancy_score,
                    "contextual_precision": result.contextual_precision_score,
                    "contextual_recall": result.contextual_recall_score,
                    "overall": result.overall_score,
                },
                "differences": {
                    "faithfulness": result.faithfulness_score
                    - mock_scores.get("faithfulness", 0.5),
                    "answer_relevancy": result.answer_relevancy_score
                    - mock_scores.get("answer_relevancy", 0.5),
                    "contextual_precision": result.contextual_precision_score
                    - mock_scores.get("contextual_precision", 0.5),
                    "contextual_recall": result.contextual_recall_score
                    - mock_scores.get("contextual_recall", 0.5),
                    "overall": result.overall_score - mock_scores.get("overall", 0.5),
                },
                "passed": result.passed,
                "failure_reasons": result.failure_reasons,
            }
            comparison_data.append(comparison)

        return comparison_data

    def calculate_statistics(
        self, comparison_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from comparison data.

        Args:
            comparison_data: List of comparison dictionaries

        Returns:
            Statistics dictionary
        """
        total = len(comparison_data)
        if total == 0:
            return {}

        # Count passes and failures
        passed = sum(1 for c in comparison_data if c["passed"])
        pass_rate = passed / total

        # Calculate average scores for mock and custom judge
        mock_avg = {
            "faithfulness": sum(
                c["mock_scores"]["faithfulness"] for c in comparison_data
            )
            / total,
            "answer_relevancy": sum(
                c["mock_scores"]["answer_relevancy"] for c in comparison_data
            )
            / total,
            "contextual_precision": sum(
                c["mock_scores"]["contextual_precision"] for c in comparison_data
            )
            / total,
            "contextual_recall": sum(
                c["mock_scores"]["contextual_recall"] for c in comparison_data
            )
            / total,
            "overall": sum(c["mock_scores"]["overall"] for c in comparison_data)
            / total,
        }

        custom_avg = {
            "faithfulness": sum(
                c["custom_judge_scores"]["faithfulness"] for c in comparison_data
            )
            / total,
            "answer_relevancy": sum(
                c["custom_judge_scores"]["answer_relevancy"] for c in comparison_data
            )
            / total,
            "contextual_precision": sum(
                c["custom_judge_scores"]["contextual_precision"]
                for c in comparison_data
            )
            / total,
            "contextual_recall": sum(
                c["custom_judge_scores"]["contextual_recall"] for c in comparison_data
            )
            / total,
            "overall": sum(c["custom_judge_scores"]["overall"] for c in comparison_data)
            / total,
        }

        # Calculate score variations (standard deviation of differences)
        import statistics

        differences = [c["differences"]["overall"] for c in comparison_data]
        variation = statistics.stdev(differences) if len(differences) > 1 else 0.0

        # Score range (min and max differences)
        min_diff = min(differences) if differences else 0.0
        max_diff = max(differences) if differences else 0.0

        # Per-persona breakdown
        persona_stats = {}
        for comparison in comparison_data:
            persona = comparison.get("persona", "unknown")
            if persona not in persona_stats:
                persona_stats[persona] = {
                    "total": 0,
                    "passed": 0,
                    "scores": [],
                }
            persona_stats[persona]["total"] += 1
            if comparison["passed"]:
                persona_stats[persona]["passed"] += 1
            persona_stats[persona]["scores"].append(
                comparison["custom_judge_scores"]["overall"]
            )

        # Calculate persona averages and pass rates
        for persona, stats in persona_stats.items():
            stats["pass_rate"] = (
                stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            stats["avg_score"] = (
                sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            )

        # Per-category breakdown
        category_stats = {}
        for comparison in comparison_data:
            category = comparison.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "passed": 0,
                    "scores": [],
                }
            category_stats[category]["total"] += 1
            if comparison["passed"]:
                category_stats[category]["passed"] += 1
            category_stats[category]["scores"].append(
                comparison["custom_judge_scores"]["overall"]
            )

        # Calculate category averages and pass rates
        for category, stats in category_stats.items():
            stats["pass_rate"] = (
                stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            stats["avg_score"] = (
                sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            )

        return {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(pass_rate, 3),
            "mock_average_scores": {k: round(v, 3) for k, v in mock_avg.items()},
            "custom_judge_average_scores": {
                k: round(v, 3) for k, v in custom_avg.items()
            },
            "score_variation": round(variation, 3),
            "score_range": {"min": round(min_diff, 3), "max": round(max_diff, 3)},
            "persona_breakdown": persona_stats,
            "category_breakdown": category_stats,
        }

    def generate_markdown_report(
        self,
        statistics: Dict[str, Any],
        comparison_data: List[Dict[str, Any]],
        output_file: Path,
    ) -> None:
        """
        Generate comprehensive markdown report.

        Args:
            statistics: Calculated statistics
            comparison_data: Comparison data for all scenarios
            output_file: Output file path
        """
        report_lines = [
            "# Custom LLM-as-Judge Evaluation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model**: {self.model}",
            f"**Stage**: {self.stage} - {EvaluationThresholds.for_stage(self.stage).get_current_stage_name()}",
            "",
        ]

        # Summary Statistics
        report_lines.extend(
            [
                "## Summary Statistics",
                "",
                f"- **Total Scenarios**: {statistics['total_scenarios']}",
                f"- **Passed**: {statistics['passed']}",
                f"- **Failed**: {statistics['failed']}",
                f"- **Pass Rate**: {statistics['pass_rate']:.1%}",
                "",
                "### Average Score Comparison",
                "",
                "| Metric | Mock | Custom Judge | Difference |",
                "|--------|------|--------------|------------|",
            ]
        )

        for metric in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
            "overall",
        ]:
            mock = statistics["mock_average_scores"][metric]
            custom = statistics["custom_judge_average_scores"][metric]
            diff = custom - mock
            diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
            report_lines.append(
                f"| {metric.replace('_', ' ').title()} | {mock:.3f} | {custom:.3f} | {diff_str} |"
            )

        report_lines.extend(
            [
                "",
                f"**Score Variation (Std Dev)**: {statistics['score_variation']:.3f}",
                f"**Score Range**: [{statistics['score_range']['min']:.3f}, {statistics['score_range']['max']:.3f}]",
                "",
            ]
        )

        # Per-Persona Breakdown
        report_lines.extend(
            [
                "## Per-Persona Breakdown",
                "",
                "| Persona | Total | Passed | Failed | Pass Rate | Avg Score |",
                "|---------|-------|--------|--------|-----------|-----------|",
            ]
        )

        for persona, stats in sorted(statistics["persona_breakdown"].items()):
            report_lines.append(
                f"| {persona} | {stats['total']} | {stats['passed']} | "
                f"{stats['total'] - stats['passed']} | {stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
            )

        report_lines.append("")

        # Per-Category Breakdown
        report_lines.extend(
            [
                "## Per-Category Breakdown",
                "",
                "| Category | Total | Passed | Failed | Pass Rate | Avg Score |",
                "|----------|-------|--------|--------|-----------|-----------|",
            ]
        )

        for category, stats in sorted(statistics["category_breakdown"].items()):
            report_lines.append(
                f"| {category} | {stats['total']} | {stats['passed']} | "
                f"{stats['total'] - stats['passed']} | {stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
            )

        report_lines.append("")

        # Failure Analysis
        failures = [c for c in comparison_data if not c["passed"]]
        if failures:
            report_lines.extend(
                [
                    "## Failure Analysis",
                    "",
                    f"**Total Failures**: {len(failures)}",
                    "",
                    "### Failed Scenarios",
                    "",
                ]
            )

            for i, failure in enumerate(failures, 1):
                report_lines.extend(
                    [
                        f"#### {i}. {failure['query'][:60]}...",
                        "",
                        f"- **Persona**: {failure['persona']}",
                        f"- **Category**: {failure['category']}",
                        f"- **Overall Score**: {failure['custom_judge_scores']['overall']:.3f}",
                        "",
                        "**Failure Reasons**:",
                    ]
                )
                for reason in failure.get("failure_reasons", []):
                    report_lines.append(f"  - {reason}")
                report_lines.append("")

        # Score Distribution Analysis
        report_lines.extend(
            [
                "## Score Distribution Analysis",
                "",
                "### Custom Judge Score Ranges",
                "",
                "| Range | Faithfulness | Answer Relevancy | Contextual Precision | Contextual Recall |",
                "|-------|--------------|------------------|---------------------|-------------------|",
            ]
        )

        # Calculate score distributions
        ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
        ]:
            range_counts = []
            for min_r, max_r in ranges:
                count = sum(
                    1
                    for c in comparison_data
                    if min_r <= c["custom_judge_scores"][metric] < max_r
                )
                pct = count / len(comparison_data) * 100 if comparison_data else 0
                range_counts.append(f"{count} ({pct:.0f}%)")
            report_lines.append(
                f"| {metric.replace('_', ' ').title()} | {' | '.join(range_counts)} |"
            )

        report_lines.extend(
            [
                "",
                "## Recommendations",
                "",
                "### Key Findings",
                "",
            ]
        )

        # Generate insights based on statistics
        if statistics["pass_rate"] < 0.5:
            report_lines.append(
                "- **Critical**: Low pass rate indicates significant quality issues. "
                "Priority: Review retrieval and generation pipeline."
            )
        elif statistics["pass_rate"] < 0.7:
            report_lines.append(
                "- **Moderate**: Pass rate below target. Consider improving retrieval quality "
                "and response generation."
            )
        else:
            report_lines.append(
                "- **Good**: Pass rate meets minimum requirements. Focus on edge cases."
            )

        # Score variation insight
        if statistics["score_variation"] < 0.05:
            report_lines.append(
                "- **Low Variation**: Custom judge scores show minimal variation from mock scores. "
                "Consider adjusting evaluation criteria or checking model calibration."
            )
        elif statistics["score_variation"] > 0.2:
            report_lines.append(
                "- **High Variation**: Significant differences between mock and custom judge scores. "
                "This indicates the custom judge is providing meaningful, nuanced evaluations."
            )

        # Faithfulness insight
        faithfulness_avg = statistics["custom_judge_average_scores"]["faithfulness"]
        if faithfulness_avg < 0.7:
            report_lines.append(
                f"- **Hallucination Risk**: Low faithfulness score ({faithfulness_avg:.3f}) indicates "
                "potential hallucination issues. Review context retrieval and generation."
            )

        # Write report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Markdown report saved to: {output_file}")

    def save_results(
        self,
        comparison_data: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        output_file: Path,
    ) -> None:
        """
        Save results to JSON file.

        Args:
            comparison_data: Comparison data for all scenarios
            statistics: Calculated statistics
            output_file: Output file path
        """
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "stage": self.stage,
                "stage_name": EvaluationThresholds.for_stage(
                    self.stage
                ).get_current_stage_name(),
            },
            "statistics": statistics,
            "scenarios": comparison_data,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

    async def run(self, input_file: str) -> None:
        """
        Run the complete evaluation workflow.

        Args:
            input_file: Path to input pilot test JSON file
        """
        logger.info("=" * 60)
        logger.info("Custom LLM-as-Judge Batch Evaluation")
        logger.info("=" * 60)

        # Load pilot test results
        pilot_data = await self.load_pilot_test_results(input_file)

        # Extract test cases
        test_cases = await self.extract_test_cases(pilot_data)

        if not test_cases:
            logger.error("No test cases found in pilot data. Exiting.")
            return

        # Run evaluation
        results = await self.run_evaluation(test_cases)
        self.evaluation_results = results

        # Generate comparison data
        comparison_data = self.generate_comparison_data(results)
        self.comparison_data = comparison_data

        # Calculate statistics
        statistics = self.calculate_statistics(comparison_data)

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_output = self.output_dir / f"custom_judge_results_{timestamp}.json"
        md_output = self.output_dir / f"custom_judge_report_{timestamp}.md"

        # Save results
        self.save_results(comparison_data, statistics, json_output)
        self.generate_markdown_report(statistics, comparison_data, md_output)

        # Print summary
        logger.info("=" * 60)
        logger.info("Evaluation Complete")
        logger.info("=" * 60)
        logger.info(f"Total Scenarios: {statistics['total_scenarios']}")
        logger.info(f"Passed: {statistics['passed']} ({statistics['pass_rate']:.1%})")
        logger.info(f"Failed: {statistics['failed']}")
        logger.info(
            f"Mock Avg Overall: {statistics['mock_average_scores']['overall']:.3f}"
        )
        logger.info(
            f"Custom Judge Avg Overall: {statistics['custom_judge_average_scores']['overall']:.3f}"
        )
        logger.info(f"Score Variation: {statistics['score_variation']:.3f}")
        logger.info("")
        logger.info(f"JSON Results: {json_output}")
        logger.info(f"Markdown Report: {md_output}")


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Custom LLM-as-Judge Batch Evaluation for RAG Quality"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/evaluations/pilot_test_20260207_145323.json",
        help="Path to pilot test JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/evaluations",
        help="Output directory for results",
    )
    parser.add_argument(
        "--stage",
        "-s",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Evaluation stage (1=initial, 2=intermediate, 3=target)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for evaluation",
    )

    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Run evaluation
    evaluator = CustomJudgeBatchEvaluator(
        output_dir=args.output,
        stage=args.stage,
        model=args.model,
    )

    await evaluator.run(args.input)


if __name__ == "__main__":
    asyncio.run(main())
