#!/usr/bin/env python3
"""
RAGAS-based LLM-as-Judge Evaluation Script for Regulation Manager RAG System.

This script evaluates the RAG system using RAGAS metrics:
- Faithfulness (환각 감지)
- Answer Relevancy (답변 관련성)
- Contextual Precision (검색 정밀도)
- Contextual Recall (검색 재현율)

Usage:
    # Evaluate on test dataset
    python scripts/evaluate_ragas.py --dataset test

    # Evaluate on specific sample count
    python scripts/evaluate_ragas.py --dataset test --samples 10

    # Evaluate with custom judge model
    python scripts/evaluate_ragas.py --dataset test --judge-model gpt-4o

    # Evaluate specific metrics only
    python scripts/evaluate_ragas.py --dataset test --metrics faithfulness answer_relevancy
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.config import get_config
from src.rag.domain.evaluation.models import EvaluationResult, EvaluationThresholds
from src.rag.domain.evaluation.quality_evaluator import RAGQualityEvaluator
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.interface.query_handler import QueryHandler, QueryOptions

# Load environment variables
load_dotenv()


@dataclass
class GroundTruthSample:
    """Ground truth sample from JSONL file."""

    id: str
    query: str
    answer: str
    context: List[str]
    category: str
    difficulty: str
    query_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthSample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            answer=data["answer"],
            context=data.get("context", []),
            category=data.get("category", ""),
            difficulty=data.get("difficulty", "중급"),
            query_type=data.get("query_type", "정확한 쿼리"),
            metadata=data.get("metadata", {}),
        )


class GroundTruthLoader:
    """Load ground truth data from JSONL files."""

    def __init__(self, data_dir: Path = Path("data/ground_truth")):
        self.data_dir = data_dir

    def load_dataset(self, split: str = "test") -> List[GroundTruthSample]:
        """
        Load ground truth dataset.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            List of ground truth samples
        """
        file_path = self.data_dir / split / f"{split}.jsonl"

        if not file_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(GroundTruthSample.from_dict(data))

        print(f"✅ Loaded {len(samples)} samples from {file_path}")
        return samples


class RAGASOrchestrator:
    """
    Orchestrates RAGAS evaluation for RAG system.

    Workflow:
    1. Load ground truth samples
    2. Generate RAG system answers
    3. Evaluate with RAGAS
    4. Save results and generate report
    """

    def __init__(
        self,
        db_path: str = "data/chroma_db",
        judge_model: str = "gpt-4o",
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        thresholds: Optional[EvaluationThresholds] = None,
    ):
        """Initialize orchestrator."""
        self.db_path = db_path
        self.config = get_config()

        # Initialize RAG components
        print("🔧 Initializing RAG system...")
        self.store = ChromaVectorStore(persist_directory=db_path)
        self.llm_client = LLMClientAdapter(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
        )
        self.query_handler = QueryHandler(
            store=self.store,
            llm_client=self.llm_client,
            use_reranker=True,
        )

        # Initialize RAGAS evaluator
        print(f"🧪 Initializing RAGAS evaluator with {judge_model}...")
        self.evaluator = RAGQualityEvaluator(
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            judge_base_url=judge_base_url,
            thresholds=thresholds,
        )

        # Default query options
        self.default_options = QueryOptions(
            top_k=5,
            use_rerank=True,
        )

        print("✅ Initialization complete\n")

    def generate_rag_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer using RAG system.

        Args:
            query: User query

        Returns:
            Dict with answer, contexts, and metadata
        """
        try:
            # Process query through RAG system
            options = QueryOptions(
                top_k=5,
                use_rerank=True,
                force_mode="ask",
            )

            result = self.query_handler.process_query(
                query=query,
                options=options,
            )

            # Extract answer and contexts
            answer_text = result.content if result.success else ""
            contexts = []

            # Extract contexts from result.data
            if result.data:
                # Try tool_results path (FunctionGemma)
                if "tool_results" in result.data:
                    for tool_result in result.data.get("tool_results", []):
                        if tool_result.get("tool_name") == "search_regulations":
                            result_data = tool_result.get("result")
                            if result_data and isinstance(result_data, dict):
                                search_results = result_data.get("results", [])
                                contexts = [
                                    r.get("text", "") or r.get("content", "")
                                    for r in search_results[:5]
                                    if isinstance(r, dict)
                                ]
                                break

                # Try direct sources path
                elif "sources" in result.data:
                    sources_data = result.data["sources"]
                    if sources_data:
                        contexts = [
                            s.get("text", "") or s.get("content", "")
                            for s in sources_data[:5]
                            if isinstance(s, dict)
                        ]

                # Try search_results path
                elif "search_results" in result.data:
                    search_results = result.data["search_results"]
                    if isinstance(search_results, list):
                        contexts = [
                            r.get("text", "") or r.get("content", "")
                            for r in search_results[:5]
                            if isinstance(r, dict)
                        ]

            return {
                "answer": answer_text,
                "contexts": contexts,
                "success": result.success,
                "confidence": getattr(result, "confidence", 0.0),
            }

        except Exception as e:
            import traceback

            return {
                "answer": f"Error: {str(e)}",
                "contexts": [],
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    async def evaluate_sample(
        self, sample: GroundTruthSample
    ) -> Optional[EvaluationResult]:
        """
        Evaluate a single ground truth sample.

        Args:
            sample: Ground truth sample

        Returns:
            EvaluationResult or None if evaluation failed
        """
        print(f"  📝 Query: {sample.query[:60]}...")

        # Generate RAG answer
        rag_result = self.generate_rag_answer(sample.query)

        if not rag_result["success"]:
            print(f"    ❌ RAG system failed: {rag_result.get('error', 'Unknown error')}")
            return None

        # Evaluate with RAGAS
        try:
            evaluation_result = await self.evaluator.evaluate(
                query=sample.query,
                answer=rag_result["answer"],
                contexts=rag_result["contexts"],
                ground_truth=sample.answer,
            )

            # Add metadata
            evaluation_result.metadata.update({
                "sample_id": sample.id,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "query_type": sample.query_type,
                "ground_truth_answer": sample.answer,
                "rag_confidence": rag_result.get("confidence", 0.0),
            })

            # Print quick summary
            status = "✅" if evaluation_result.passed else "❌"
            print(
                f"    {status} F={evaluation_result.faithfulness:.2f} "
                f"R={evaluation_result.answer_relevancy:.2f} "
                f"P={evaluation_result.contextual_precision:.2f} "
                f"Rec={evaluation_result.contextual_recall:.2f} "
                f"Overall={evaluation_result.overall_score:.2f}"
            )

            return evaluation_result

        except Exception as e:
            import traceback

            print(f"    ❌ Evaluation failed: {e}")
            print(f"    Traceback: {traceback.format_exc()[:200]}")
            return None

    async def evaluate_dataset(
        self,
        samples: List[GroundTruthSample],
        max_samples: Optional[int] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate entire dataset.

        Args:
            samples: Ground truth samples
            max_samples: Maximum number of samples to evaluate

        Returns:
            List of evaluation results
        """
        if max_samples:
            samples = samples[:max_samples]

        print(f"\n🚀 Starting evaluation on {len(samples)} samples...\n")

        results = []
        for i, sample in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}]", end=" ")
            result = await self.evaluate_sample(sample)
            if result:
                results.append(result)

        print(f"\n✅ Evaluation complete: {len(results)}/{len(samples)} successful\n")
        return results

    def save_results(
        self,
        results: List[EvaluationResult],
        output_dir: Path,
    ) -> Path:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results
            output_dir: Output directory

        Returns:
            Path to saved results file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ragas_evaluation_{timestamp}.json"

        # Convert results to dict
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "summary": self._calculate_summary(results),
            "results": [r.to_dict() for r in results],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Results saved to: {output_file}")
        return output_file

    def _calculate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        if not results:
            return {}

        total = len(results)
        passed = sum(1 for r in results if r.passed)

        # Calculate average scores
        avg_faithfulness = sum(r.faithfulness for r in results) / total
        avg_relevancy = sum(r.answer_relevancy for r in results) / total
        avg_precision = sum(r.contextual_precision for r in results) / total
        avg_recall = sum(r.contextual_recall for r in results) / total
        avg_overall = sum(r.overall_score for r in results) / total

        # Count by category
        by_category = {}
        for r in results:
            category = r.metadata.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"count": 0, "passed": 0, "overall": 0.0}
            by_category[category]["count"] += 1
            if r.passed:
                by_category[category]["passed"] += 1
            by_category[category]["overall"] += r.overall_score

        # Calculate category averages
        for category in by_category:
            count = by_category[category]["count"]
            by_category[category]["pass_rate"] = by_category[category]["passed"] / count
            by_category[category]["avg_overall"] = by_category[category]["overall"] / count

        return {
            "total_samples": total,
            "passed_samples": passed,
            "pass_rate": passed / total,
            "average_scores": {
                "faithfulness": avg_faithfulness,
                "answer_relevancy": avg_relevancy,
                "contextual_precision": avg_precision,
                "contextual_recall": avg_recall,
                "overall": avg_overall,
            },
            "by_category": by_category,
        }

    def generate_report(self, results: List[EvaluationResult]) -> str:
        """Generate human-readable evaluation report."""
        summary = self._calculate_summary(results)

        if not summary:
            return "No results to report."

        report_lines = [
            "=" * 80,
            "RAGAS Evaluation Report",
            "=" * 80,
            "",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Samples: {summary['total_samples']}",
            f"Passed: {summary['passed_samples']}/{summary['total_samples']} ({summary['pass_rate']:.1%})",
            "",
            "## Average Scores",
            "",
        ]

        avg_scores = summary["average_scores"]
        report_lines.extend([
            f"✦ Faithfulness (환각 감지):     {avg_scores['faithfulness']:.3f} {'✅' if avg_scores['faithfulness'] >= 0.90 else '❌'}",
            f"✦ Answer Relevancy (답변 관련성): {avg_scores['answer_relevancy']:.3f} {'✅' if avg_scores['answer_relevancy'] >= 0.85 else '❌'}",
            f"✦ Contextual Precision (검색 정밀도): {avg_scores['contextual_precision']:.3f} {'✅' if avg_scores['contextual_precision'] >= 0.80 else '❌'}",
            f"✦ Contextual Recall (검색 재현율):   {avg_scores['contextual_recall']:.3f} {'✅' if avg_scores['contextual_recall'] >= 0.80 else '❌'}",
            f"✦ Overall Score:                   {avg_scores['overall']:.3f}",
            "",
            "## Results by Category",
            "",
        ])

        # Sort categories by pass rate
        categories_sorted = sorted(
            summary["by_category"].items(),
            key=lambda x: x[1]["pass_rate"],
            reverse=True,
        )

        for category, stats in categories_sorted:
            report_lines.extend([
                f"### {category}",
                f"  Samples: {stats['count']}",
                f"  Pass Rate: {stats['pass_rate']:.1%}",
                f"  Avg Overall: {stats['avg_overall']:.3f}",
                "",
            ])

        # Failed samples
        failed_samples = [r for r in results if not r.passed]
        if failed_samples:
            report_lines.extend([
                "## Failed Samples",
                "",
            ])

            for i, result in enumerate(failed_samples[:10], 1):  # Show first 10
                report_lines.extend([
                    f"### {i}. {result.query[:60]}...",
                    "**Failures:**",
                ])
                for reason in result.failure_reasons:
                    report_lines.append(f"  - {reason}")
                report_lines.extend([
                    f"**Scores:** F={result.faithfulness:.2f} R={result.answer_relevancy:.2f} "
                    f"P={result.contextual_precision:.2f} Rec={result.contextual_recall:.2f}",
                    "",
                ])

            if len(failed_samples) > 10:
                report_lines.append(f"... and {len(failed_samples) - 10} more failed samples")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


# CLI Interface
@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["train", "val", "test"]),
    default="test",
    help="Dataset split to evaluate",
)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Maximum number of samples to evaluate",
)
@click.option(
    "--judge-model",
    type=str,
    default="gpt-4o",
    help="Judge LLM model for RAGAS evaluation",
)
@click.option(
    "--db-path",
    type=str,
    default="data/chroma_db",
    help="Path to ChromaDB database",
)
@click.option(
    "--output-dir",
    type=str,
    default="test_reports",
    help="Output directory for results",
)
@click.option(
    "--metrics",
    type=str,
    multiple=True,
    help="Specific metrics to evaluate (faithfulness, answer_relevancy, contextual_precision, contextual_recall)",
)
def main(
    dataset: str,
    samples: Optional[int],
    judge_model: str,
    db_path: str,
    output_dir: str,
    metrics: tuple,
):
    """
    Evaluate RAG system using RAGAS LLM-as-Judge methodology.

    This script evaluates the regulation manager RAG system across four core metrics:
    - Faithfulness: Detects hallucination in generated answers
    - Answer Relevancy: Measures how well the answer addresses the query
    - Contextual Precision: Evaluates retrieval ranking quality
    - Contextual Recall: Measures information completeness in retrieved contexts

    Example usage:
        python scripts/evaluate_ragas.py --dataset test --samples 10
        python scripts/evaluate_ragas.py --dataset val --judge-model gpt-4o
    """
    print("🎯 RAGAS Evaluation for Regulation Manager RAG System")
    print("=" * 80)
    print()

    # Load ground truth data
    print(f"📂 Loading ground truth data from '{dataset}' split...")
    loader = GroundTruthLoader()
    samples_list = loader.load_dataset(split=dataset)

    if samples and samples < len(samples_list):
        print(f"⚠️  Limiting evaluation to {samples} samples")
        samples_list = samples_list[:samples]

    print()

    # Initialize orchestrator
    orchestrator = RAGASOrchestrator(
        db_path=db_path,
        judge_model=judge_model,
        judge_api_key=None,  # Will use OPENAI_API_KEY from env
        judge_base_url=None,  # Will use default OpenAI endpoint
    )

    # Run evaluation
    results = asyncio.run(orchestrator.evaluate_dataset(samples_list))

    if not results:
        print("❌ No successful evaluations. Exiting.")
        sys.exit(1)

    # Save results
    output_path = Path(output_dir)
    results_file = orchestrator.save_results(results, output_path)

    # Generate and print report
    report = orchestrator.generate_report(results)
    print("\n" + report)

    # Save report
    report_file = output_path / f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Report saved to: {report_file}")

    # Exit with appropriate code
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    if pass_rate >= 0.8:
        print(f"\n✅ Evaluation PASSED with {pass_rate:.1%} pass rate")
        sys.exit(0)
    else:
        print(f"\n⚠️  Evaluation WARNING: {pass_rate:.1%} pass rate below 80%")
        sys.exit(1)


if __name__ == "__main__":
    main()
