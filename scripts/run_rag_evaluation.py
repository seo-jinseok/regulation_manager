#!/usr/bin/env python3
"""
Comprehensive RAG Quality Evaluation Script.

Executes queries through the RAG system and evaluates responses
using LLM-as-Judge methodology.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.evaluation.quality_evaluator import RAGQualityEvaluator
from src.rag.domain.evaluation.models import EvaluationThresholds
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGEvaluationRunner:
    """Runs comprehensive RAG quality evaluation across all personas."""

    def __init__(
        self,
        queries_dir: Path,
        output_dir: Path,
        judge_model: str = "gpt-4o",
        stage: int = 1,
    ):
        self.queries_dir = queries_dir
        self.output_dir = output_dir
        self.judge_model = judge_model
        self.stage = stage
        self.results: list[dict[str, Any]] = []

        # Initialize components
        self._search_usecase: RegulationSearchUseCase | None = None
        self._evaluator: RAGQualityEvaluator | None = None

    def _initialize_components(self) -> None:
        """Initialize RAG system and evaluator."""
        logger.info("Initializing RAG system and evaluator...")

        # Initialize vector store
        db_path = project_root / "data" / "chroma_db"
        store = ChromaVectorStore(persist_directory=str(db_path))

        if store.count() == 0:
            raise RuntimeError("Database is empty. Run 'regulation sync' first.")

        # Initialize LLM client for RAG
        llm_client = LLMClientAdapter(
            provider="local",
            model=None,
            base_url=None,
        )

        # Initialize search use case
        self._search_usecase = SearchUseCase(
            store=store,
            llm_client=llm_client,
            use_reranker=True,
        )

        # Initialize evaluator with thresholds for current stage
        thresholds = EvaluationThresholds.for_stage(self.stage)
        self._evaluator = RAGQualityEvaluator(
            judge_model=self.judge_model,
            thresholds=thresholds,
            use_ragas=True,  # Will fall back to mock if RAGAS not available
            stage=self.stage,
        )

        logger.info(f"Vector store initialized with {store.count()} documents")
        logger.info(f"Evaluator initialized with model: {self.judge_model}")
        logger.info(f"Stage: {self.stage} - {thresholds.get_current_stage_name()}")

    def load_queries(self, persona: str) -> list[dict[str, Any]]:
        """Load queries for a specific persona."""
        query_file = self.queries_dir / f"queries_{persona}.json"
        if not query_file.exists():
            logger.warning(f"Query file not found: {query_file}")
            return []

        with open(query_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("queries", [])

    async def execute_query(self, query: str) -> tuple[str, list[str], dict[str, Any]]:
        """Execute a single query through the RAG system."""
        if not self._search_usecase:
            raise RuntimeError("Search use case not initialized")

        try:
            # Use the ask method to get answer and sources
            result = self._search_usecase.ask(
                question=query,
                top_k=5,
            )

            # Answer is a dataclass with text, sources, confidence
            answer = result.text
            # Extract text from SearchResult objects
            sources = [s.text for s in result.sources if hasattr(s, 'text')]
            metadata = {
                "confidence": result.confidence,
                "num_sources": len(result.sources),
            }

            return answer, sources, metadata

        except Exception as e:
            logger.error(f"Error executing query '{query}': {e}")
            return f"Error: {str(e)}", [], {"error": str(e)}

    async def evaluate_response(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        persona: str,
        category: str,
        difficulty: str,
    ) -> dict[str, Any]:
        """Evaluate a single response using LLM-as-Judge."""
        if not self._evaluator:
            raise RuntimeError("Evaluator not initialized")

        try:
            result = await self._evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=None,  # No ground truth available
            )

            return {
                "persona": persona,
                "query": query,
                "category": category,
                "difficulty": difficulty,
                "answer": answer,
                "contexts": contexts[:2] if contexts else [],  # Store first 2 contexts
                "evaluation": {
                    "faithfulness": result.faithfulness,
                    "answer_relevancy": result.answer_relevancy,
                    "contextual_precision": result.contextual_precision,
                    "contextual_recall": result.contextual_recall,
                    "overall_score": result.overall_score,
                    "passed": result.passed,
                    "failure_reasons": result.failure_reasons,
                },
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            return {
                "persona": persona,
                "query": query,
                "category": category,
                "difficulty": difficulty,
                "answer": answer,
                "contexts": contexts[:2] if contexts else [],
                "evaluation": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "contextual_precision": 0.0,
                    "contextual_recall": 0.0,
                    "overall_score": 0.0,
                    "passed": False,
                    "failure_reasons": [f"Evaluation error: {str(e)}"],
                },
                "metadata": {"error": str(e)},
            }

    async def run_persona_evaluation(self, persona: str) -> list[dict[str, Any]]:
        """Run evaluation for a specific persona."""
        logger.info(f"Starting evaluation for persona: {persona}")

        queries = self.load_queries(persona)
        if not queries:
            logger.warning(f"No queries found for persona: {persona}")
            return []

        results = []
        total = len(queries)

        for i, query_data in enumerate(queries, 1):
            query = query_data["query"]
            category = query_data.get("category", "general")
            difficulty = query_data.get("difficulty", "medium")

            logger.info(f"[{persona}] Processing query {i}/{total}: {query[:50]}...")

            # Execute query through RAG system
            answer, contexts, exec_metadata = await self.execute_query(query)

            # Evaluate response
            result = await self.evaluate_response(
                query=query,
                answer=answer,
                contexts=contexts,
                persona=persona,
                category=category,
                difficulty=difficulty,
            )
            result["execution_metadata"] = exec_metadata

            results.append(result)

            # Log progress
            if result["evaluation"]["passed"]:
                logger.info(f"  PASSED (score: {result['evaluation']['overall_score']:.3f})")
            else:
                logger.info(f"  FAILED (score: {result['evaluation']['overall_score']:.3f})")

        logger.info(f"Completed evaluation for persona: {persona}")
        return results

    async def run_all_evaluations(self) -> dict[str, Any]:
        """Run evaluation for all personas."""
        self._initialize_components()

        personas = [
            "undergraduate",
            "graduate",
            "professor",
            "staff",
            "parent",
            "international",
        ]

        all_results = []
        start_time = datetime.now()

        for persona in personas:
            persona_results = await self.run_persona_evaluation(persona)
            all_results.extend(persona_results)

            # Save intermediate results
            self._save_persona_results(persona, persona_results)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate aggregate statistics
        summary = self._calculate_summary(all_results, duration)

        # Save final results
        self._save_final_results(all_results, summary)

        return {
            "results": all_results,
            "summary": summary,
        }

    def _save_persona_results(
        self, persona: str, results: list[dict[str, Any]]
    ) -> None:
        """Save results for a specific persona."""
        output_file = self.output_dir / f"results_{persona}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results for {persona} to {output_file}")

    def _calculate_summary(
        self, results: list[dict[str, Any]], duration: float
    ) -> dict[str, Any]:
        """Calculate aggregate statistics from all results."""
        if not results:
            return {"error": "No results to summarize"}

        total = len(results)
        passed = sum(1 for r in results if r["evaluation"]["passed"])

        # Calculate averages
        avg_faithfulness = sum(
            r["evaluation"]["faithfulness"] for r in results
        ) / total
        avg_relevancy = sum(
            r["evaluation"]["answer_relevancy"] for r in results
        ) / total
        avg_precision = sum(
            r["evaluation"]["contextual_precision"] for r in results
        ) / total
        avg_recall = sum(
            r["evaluation"]["contextual_recall"] for r in results
        ) / total
        avg_overall = sum(
            r["evaluation"]["overall_score"] for r in results
        ) / total

        # Per-persona breakdown
        persona_stats: dict[str, dict[str, Any]] = {}
        for persona in ["undergraduate", "graduate", "professor", "staff", "parent", "international"]:
            persona_results = [r for r in results if r["persona"] == persona]
            if persona_results:
                persona_passed = sum(
                    1 for r in persona_results if r["evaluation"]["passed"]
                )
                persona_stats[persona] = {
                    "total": len(persona_results),
                    "passed": persona_passed,
                    "pass_rate": persona_passed / len(persona_results),
                    "avg_score": sum(
                        r["evaluation"]["overall_score"] for r in persona_results
                    ) / len(persona_results),
                }

        # Per-category breakdown
        category_stats: dict[str, dict[str, Any]] = {}
        categories = set(r["category"] for r in results)
        for category in categories:
            cat_results = [r for r in results if r["category"] == category]
            if cat_results:
                cat_passed = sum(
                    1 for r in cat_results if r["evaluation"]["passed"]
                )
                category_stats[category] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "pass_rate": cat_passed / len(cat_results),
                    "avg_score": sum(
                        r["evaluation"]["overall_score"] for r in cat_results
                    ) / len(cat_results),
                }

        # Per-difficulty breakdown
        difficulty_stats: dict[str, dict[str, Any]] = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r["difficulty"] == difficulty]
            if diff_results:
                diff_passed = sum(
                    1 for r in diff_results if r["evaluation"]["passed"]
                )
                difficulty_stats[difficulty] = {
                    "total": len(diff_results),
                    "passed": diff_passed,
                    "pass_rate": diff_passed / len(diff_results),
                    "avg_score": sum(
                        r["evaluation"]["overall_score"] for r in diff_results
                    ) / len(diff_results),
                }

        # Failure pattern analysis
        failure_patterns: dict[str, int] = {}
        for result in results:
            for reason in result["evaluation"]["failure_reasons"]:
                # Extract key failure type
                if "faithfulness" in reason.lower():
                    failure_patterns["faithfulness"] = failure_patterns.get("faithfulness", 0) + 1
                elif "relevancy" in reason.lower():
                    failure_patterns["answer_relevancy"] = failure_patterns.get("answer_relevancy", 0) + 1
                elif "precision" in reason.lower():
                    failure_patterns["contextual_precision"] = failure_patterns.get("contextual_precision", 0) + 1
                elif "recall" in reason.lower():
                    failure_patterns["contextual_recall"] = failure_patterns.get("contextual_recall", 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": total,
            "total_passed": passed,
            "pass_rate": passed / total,
            "duration_seconds": duration,
            "metrics": {
                "faithfulness": round(avg_faithfulness, 3),
                "answer_relevancy": round(avg_relevancy, 3),
                "contextual_precision": round(avg_precision, 3),
                "contextual_recall": round(avg_recall, 3),
                "overall_score": round(avg_overall, 3),
            },
            "thresholds": {
                "faithfulness": 0.90,
                "answer_relevancy": 0.85,
                "contextual_precision": 0.80,
                "contextual_recall": 0.80,
            },
            "pass_threshold": 0.80,
            "passed_evaluation": (passed / total) >= 0.80,
            "persona_breakdown": persona_stats,
            "category_breakdown": category_stats,
            "difficulty_breakdown": difficulty_stats,
            "failure_patterns": failure_patterns,
        }

    def _save_final_results(
        self, results: list[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        """Save final evaluation results and summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = self.output_dir / f"evaluation_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results, "summary": summary},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Saved full evaluation results to {results_file}")

        # Generate markdown report
        report = self._generate_report(summary, results)
        report_file = self.output_dir / f"report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved evaluation report to {report_file}")

        # Generate SPEC template for failures
        if not summary["passed_evaluation"]:
            spec_template = self._generate_spec_template(summary, results)
            spec_file = self.output_dir / f"spec_{timestamp}.md"
            with open(spec_file, "w", encoding="utf-8") as f:
                f.write(spec_template)
            logger.info(f"Saved improvement SPEC template to {spec_file}")

    def _generate_report(
        self, summary: dict[str, Any], results: list[dict[str, Any]]
    ) -> str:
        """Generate markdown evaluation report."""
        lines = [
            "# RAG Quality Evaluation Report",
            "",
            f"**Generated:** {summary['timestamp']}",
            f"**Duration:** {summary['duration_seconds']:.1f} seconds",
            "",
            "## Summary",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| **Overall Pass Rate** | {summary['pass_rate']:.1%} | 80% | {'PASS' if summary['passed_evaluation'] else 'FAIL'} |",
            f"| **Total Queries** | {summary['total_queries']} | 150+ | {'PASS' if summary['total_queries'] >= 150 else 'FAIL'} |",
            f"| **Queries Passed** | {summary['total_passed']} | - | - |",
            "",
            "## Metric Scores",
            "",
            f"| Metric | Score | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| Faithfulness | {summary['metrics']['faithfulness']:.3f} | {summary['thresholds']['faithfulness']} | {'PASS' if summary['metrics']['faithfulness'] >= summary['thresholds']['faithfulness'] else 'FAIL'} |",
            f"| Answer Relevancy | {summary['metrics']['answer_relevancy']:.3f} | {summary['thresholds']['answer_relevancy']} | {'PASS' if summary['metrics']['answer_relevancy'] >= summary['thresholds']['answer_relevancy'] else 'FAIL'} |",
            f"| Contextual Precision | {summary['metrics']['contextual_precision']:.3f} | {summary['thresholds']['contextual_precision']} | {'PASS' if summary['metrics']['contextual_precision'] >= summary['thresholds']['contextual_precision'] else 'FAIL'} |",
            f"| Contextual Recall | {summary['metrics']['contextual_recall']:.3f} | {summary['thresholds']['contextual_recall']} | {'PASS' if summary['metrics']['contextual_recall'] >= summary['thresholds']['contextual_recall'] else 'FAIL'} |",
            f"| **Overall Score** | **{summary['metrics']['overall_score']:.3f}** | **0.80** | {'**PASS**' if summary['metrics']['overall_score'] >= 0.80 else '**FAIL**'} |",
            "",
            "## Per-Persona Results",
            "",
            f"| Persona | Total | Passed | Pass Rate | Avg Score |",
            f"|---------|-------|--------|-----------|-----------|",
        ]

        for persona, stats in summary.get("persona_breakdown", {}).items():
            lines.append(
                f"| {persona} | {stats['total']} | {stats['passed']} | "
                f"{stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
            )

        lines.extend([
            "",
            "## Per-Category Results",
            "",
            f"| Category | Total | Passed | Pass Rate | Avg Score |",
            f"|----------|-------|--------|-----------|-----------|",
        ])

        for category, stats in sorted(
            summary.get("category_breakdown", {}).items(),
            key=lambda x: x[1]["avg_score"],
        ):
            lines.append(
                f"| {category} | {stats['total']} | {stats['passed']} | "
                f"{stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
            )

        lines.extend([
            "",
            "## Per-Difficulty Results",
            "",
            f"| Difficulty | Total | Passed | Pass Rate | Avg Score |",
            f"|------------|-------|--------|-----------|-----------|",
        ])

        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in summary.get("difficulty_breakdown", {}):
                stats = summary["difficulty_breakdown"][difficulty]
                lines.append(
                    f"| {difficulty} | {stats['total']} | {stats['passed']} | "
                    f"{stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
                )

        # Add failure patterns
        if summary.get("failure_patterns"):
            lines.extend([
                "",
                "## Failure Patterns",
                "",
                f"| Failure Type | Count |",
                f"|--------------|-------|",
            ])
            for pattern, count in sorted(
                summary["failure_patterns"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                lines.append(f"| {pattern} | {count} |")

        # Add sample failures
        failures = [r for r in results if not r["evaluation"]["passed"]][:10]
        if failures:
            lines.extend([
                "",
                "## Sample Failures (First 10)",
                "",
            ])
            for i, failure in enumerate(failures, 1):
                lines.extend([
                    f"### Failure {i}: {failure['persona']} - {failure['category']}",
                    f"- **Query:** {failure['query']}",
                    f"- **Difficulty:** {failure['difficulty']}",
                    f"- **Score:** {failure['evaluation']['overall_score']:.3f}",
                    f"- **Reasons:** {', '.join(failure['evaluation']['failure_reasons'])}",
                    "",
                ])

        return "\n".join(lines)

    def _generate_spec_template(
        self, summary: dict[str, Any], results: list[dict[str, Any]]
    ) -> str:
        """Generate SPEC template for improvement recommendations."""
        timestamp = datetime.now().strftime("%Y%m%d")
        spec_id = f"SPEC-RAG-Q-{timestamp}"

        lines = [
            f"# {spec_id}: RAG Quality Improvement",
            "",
            "## Status",
            "",
            "- **State:** DRAFT",
            f"- **Created:** {summary['timestamp']}",
            "- **Priority:** HIGH",
            "",
            "## Problem Statement",
            "",
            f"The RAG system achieved a {summary['pass_rate']:.1%} pass rate, "
            f"which is {'above' if summary['passed_evaluation'] else 'below'} the 80% threshold.",
            "",
            "### Key Metrics",
            "",
        ]

        for metric, score in summary["metrics"].items():
            threshold = summary["thresholds"].get(metric, 0.80)
            status = "PASS" if score >= threshold else "FAIL"
            lines.append(f"- **{metric}:** {score:.3f} (threshold: {threshold}) - {status}")

        lines.extend([
            "",
            "## Improvement Recommendations",
            "",
        ])

        # Add recommendations based on failure patterns
        patterns = summary.get("failure_patterns", {})
        if patterns:
            lines.append("### Based on Failure Patterns")
            lines.append("")
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                if pattern == "faithfulness":
                    lines.extend([
                        f"#### Faithfulness ({count} failures)",
                        "- Implement stricter context grounding verification",
                        "- Add citation requirements for all factual claims",
                        "- Consider answer verification step before response",
                        "",
                    ])
                elif pattern == "answer_relevancy":
                    lines.extend([
                        f"#### Answer Relevancy ({count} failures)",
                        "- Improve query understanding and intent classification",
                        "- Add query expansion for ambiguous questions",
                        "- Implement answer completeness checking",
                        "",
                    ])
                elif pattern == "contextual_precision":
                    lines.extend([
                        f"#### Contextual Precision ({count} failures)",
                        "- Tune reranker model for better ranking",
                        "- Implement query-document relevance scoring",
                        "- Consider hybrid search approach",
                        "",
                    ])
                elif pattern == "contextual_recall":
                    lines.extend([
                        f"#### Contextual Recall ({count} failures)",
                        "- Increase retrieval top_k for complex queries",
                        "- Implement multi-hop retrieval for complex questions",
                        "- Add query decomposition for multi-part questions",
                        "",
                    ])

        # Add persona-specific recommendations
        weak_personas = [
            (p, s) for p, s in summary.get("persona_breakdown", {}).items()
            if s["pass_rate"] < 0.80
        ]
        if weak_personas:
            lines.extend([
                "### Persona-Specific Improvements",
                "",
            ])
            for persona, stats in sorted(weak_personas, key=lambda x: x[1]["pass_rate"]):
                lines.extend([
                    f"#### {persona} (Pass Rate: {stats['pass_rate']:.1%})",
                    f"- Focus on queries with lower scores",
                    f"- Review category-specific performance",
                    "",
                ])

        # Add difficulty-specific recommendations
        hard_stats = summary.get("difficulty_breakdown", {}).get("hard", {})
        if hard_stats and hard_stats.get("pass_rate", 1.0) < 0.80:
            lines.extend([
                "### Hard Query Improvements",
                f"- Current pass rate for hard queries: {hard_stats['pass_rate']:.1%}",
                "- Implement query clarification system",
                "- Add multi-turn conversation support",
                "- Consider ensemble approach for complex queries",
                "",
            ])

        lines.extend([
            "## Acceptance Criteria",
            "",
            "- [ ] Overall pass rate >= 80%",
            "- [ ] All metric scores meet thresholds",
            "- [ ] No persona with pass rate < 70%",
            "- [ ] Hard query pass rate >= 60%",
            "",
            "## Related SPECs",
            "",
            "- SPEC-RAG-QUALITY-001 (Initial RAG implementation)",
            "- SPEC-RAG-QUALITY-002 (Quality improvements)",
            "",
        ])

        return "\n".join(lines)


async def main():
    """Main entry point for RAG evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG quality evaluation")
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=Path("data/evaluations/persona_results_20260215_174915"),
        help="Directory containing persona query files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluations"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="LLM model to use for evaluation",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Evaluation stage (1=initial, 2=intermediate, 3=target)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    runner = RAGEvaluationRunner(
        queries_dir=args.queries_dir,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        stage=args.stage,
    )

    results = await runner.run_all_evaluations()

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Passed: {summary['total_passed']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Overall Score: {summary['metrics']['overall_score']:.3f}")
    print(f"Evaluation: {'PASSED' if summary['passed_evaluation'] else 'FAILED'}")
    print("=" * 60)

    return 0 if summary["passed_evaluation"] else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
