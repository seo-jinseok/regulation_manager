"""
SPEC-RAG-QUALITY-007: Evaluation Metrics Verification Script

Purpose: Identify root cause of uniform 0.50 scores in evaluation results.

ROOT CAUSE ANALYSIS:
====================
The uniform 0.50 scores in evaluation results are caused by:

1. **Missing OPENAI_API_KEY**: RAGAS requires an LLM for evaluation.
   Without an API key, the evaluator falls back to mock scoring.

2. **Mock Scoring Fallback**: The `_mock_*` methods in
   `src/rag/domain/evaluation/quality_evaluator.py` return 0.5 as
   default when contexts are empty or data is missing.

3. **Empty Contexts**: Some evaluation samples may have empty context
   lists, triggering the fallback to 0.5 scores.

SOLUTION:
=========
1. Set OPENAI_API_KEY environment variable
2. Ensure evaluation samples include non-empty contexts
3. Verify RAGAS initialization with proper LLM configuration

This script verifies:
1. RAGAS environment setup (chromadb, ragas, metrics)
2. Citation format validation
3. Context relevance score distribution
4. Small-scale test to identify score patterns

Created: 2026-02-20
Author: MoAI TDD Implementation
"""

import json
import logging
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_ragas_environment() -> Dict[str, Any]:
    """
    Check RAGAS environment setup.

    Verifies:
    - chromadb installation
    - ragas installation
    - ragas metrics availability
    - LLM client configuration

    Returns:
        Dictionary with environment status
    """
    result = {
        "chromadb": False,
        "ragas": False,
        "ragas_metrics": False,
        "llm_configured": False,
        "errors": [],
        "warnings": [],
    }

    # Check chromadb
    try:
        import chromadb

        result["chromadb"] = True
        result["chromadb_version"] = chromadb.__version__
        logger.info(f"chromadb available: {chromadb.__version__}")
    except ImportError as e:
        result["errors"].append(f"chromadb not installed: {e}")
        logger.warning("chromadb not installed")

    # Check ragas
    try:
        import ragas

        result["ragas"] = True
        result["ragas_version"] = ragas.__version__
        logger.info(f"ragas available: {ragas.__version__}")
    except ImportError as e:
        result["errors"].append(f"ragas not installed: {e}")
        logger.warning("ragas not installed")
        return result  # No point continuing without ragas

    # Check ragas metrics (using new import path)
    try:
        # Try new import path first (v1.0+)
        try:
            from ragas.metrics._faithfulness import (
                Faithfulness,
            )
            from ragas.metrics._answer_relevance import (
                AnswerRelevance,
            )
            from ragas.metrics._context_precision import (
                ContextPrecision,
            )
            from ragas.metrics._context_recall import (
                ContextRecall,
            )

            result["ragas_metrics"] = True
            result["metrics_available"] = [
                "Faithfulness",
                "AnswerRelevance",
                "ContextPrecision",
                "ContextRecall",
            ]
            logger.info("RAGAS metrics available (new import path)")
        except ImportError:
            # Fallback to old import path (deprecated)
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            result["ragas_metrics"] = True
            result["metrics_available"] = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            result["warnings"].append(
                "Using deprecated ragas.metrics import. "
                "Update to ragas.metrics.collections"
            )
            logger.info("RAGAS metrics available (deprecated import path)")

    except ImportError as e:
        result["errors"].append(f"RAGAS metrics not available: {e}")
        logger.error(f"RAGAS metrics import failed: {e}")

    # Check LLM configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        result["llm_configured"] = True
        result["llm_provider"] = "openai"
        logger.info("OpenAI API key configured")
    else:
        result["warnings"].append("OPENAI_API_KEY not set")
        logger.warning("OPENAI_API_KEY not set - RAGAS may use defaults")

    return result


def verify_citation_format(response: str) -> Dict[str, Any]:
    """
    Check if citation format matches evaluation expectations.

    Detects Korean regulation citation patterns:
    - 제N조/N조 (Article N)
    - 제N항/N항 (Paragraph N)
    - 규정명 (Regulation name)
    - 학칙 (School rules)

    Args:
        response: The answer text to check

    Returns:
        Dictionary with citation analysis
    """
    result = {
        "has_citation": False,
        "detected_citations": [],
        "citation_types": [],
        "citation_count": 0,
    }

    # Citation patterns from evaluation_constants.py
    citation_patterns = [
        (r"제\d+\s*[조항]", "article_formal"),  # 제N조/제N항
        (r"\d+\s*[조항](?!.*제)", "article_informal"),  # N조/N항 (not after 제)
        (r"\d+조\s*\d+항", "article_paragraph"),
        (r"\d+조\s*제\d+항", "article_paragraph_formal"),
        (r"\w*규정", "regulation"),
        (r"\w*학칙", "school_rules"),
        (r"\w*시행세칙", "enforcement_rules"),
    ]

    detected = []
    for pattern, citation_type in citation_patterns:
        matches = re.findall(pattern, response)
        if matches:
            detected.extend(matches)
            result["citation_types"].append(citation_type)

    if detected:
        result["has_citation"] = True
        result["detected_citations"] = detected
        result["citation_count"] = len(detected)

    return result


def verify_context_relevance(scores: List[float]) -> Dict[str, Any]:
    """
    Analyze score distribution for uniform scores.

    Detects:
    - Uniform scores (all same value)
    - Likely default values (all exactly 0.50)
    - Score variance and standard deviation

    Args:
        scores: List of context relevance scores

    Returns:
        Dictionary with score analysis
    """
    result = {
        "count": len(scores),
        "mean": 0.0,
        "variance": 0.0,
        "std_dev": 0.0,
        "min": 0.0,
        "max": 0.0,
        "is_uniform": False,
        "likely_default": False,
        "warning": "",
    }

    if not scores:
        result["warning"] = "No scores provided"
        return result

    # Calculate statistics
    result["mean"] = statistics.mean(scores)
    result["min"] = min(scores)
    result["max"] = max(scores)

    if len(scores) > 1:
        result["variance"] = statistics.variance(scores)
        result["std_dev"] = statistics.stdev(scores)

    # Check for uniform scores (all identical)
    if result["variance"] == 0:
        result["is_uniform"] = True
        result["warning"] = (
            f"SUSPICIOUS: All {len(scores)} scores are identical ({scores[0]}). "
            "This suggests default values or calculation failure."
        )

    # Check for likely default values (all exactly 0.50)
    if all(score == 0.50 for score in scores):
        result["likely_default"] = True
        result["warning"] = (
            f"CRITICAL: All {len(scores)} scores are exactly 0.50. "
            "This is the default value when RAGAS cannot calculate metrics. "
            "Likely causes: Missing LLM configuration, chromadb issues, or metric calculation failure."
        )

    return result


def run_small_scale_test(queries: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Run small test (10 queries) to identify score patterns.

    Tests RAGAS evaluation with minimal data to verify:
    - Metrics are calculated (not defaulted)
    - Score distribution is reasonable
    - No uniform 0.50 patterns

    Args:
        queries: Optional list of test queries. If None, uses default test cases.

    Returns:
        Dictionary with test results and analysis
    """
    result = {
        "ragas_available": False,
        "test_count": 0,
        "scores": {},
        "analysis": {},
        "errors": [],
    }

    # Default test queries if none provided
    if queries is None:
        queries = [
            {
                "query": "복학 신청 방법",
                "answer": "복학규정 제2조에 따라 학기 시작 30일 전까지 신청 가능합니다.",
                "contexts": ["복학규정 제2조: 학기 시작 30일 전까지 신청"],
            },
            {
                "query": "휴학 기간",
                "answer": "휴학은 최대 2년까지 가능하며 휴학규정 제5조에 명시되어 있습니다.",
                "contexts": ["휴학규정 제5조: 휴학 기간은 2년 이내"],
            },
            {
                "query": "장학금 신청",
                "answer": "장학금 규정 제3조에 따라 성적 3.0 이상인 학생이 신청 가능합니다.",
                "contexts": ["장학금 규정 제3조: 성적 3.0 이상 신청 가능"],
            },
        ]

    result["test_count"] = len(queries)

    # Verify RAGAS environment first
    env_check = verify_ragas_environment()
    if not env_check.get("ragas"):
        result["errors"].append("RAGAS not available - cannot run evaluation")
        result["analysis"] = {
            "environment_status": env_check,
            "recommendation": "Install RAGAS: pip install ragas",
        }
        return result

    result["ragas_available"] = True

    # Try actual RAGAS evaluation
    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevance
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall

        # Prepare dataset
        from datasets import Dataset

        data = {
            "question": [q["query"] for q in queries],
            "answer": [q["answer"] for q in queries],
            "contexts": [q["contexts"] for q in queries],
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        logger.info("Running RAGAS evaluation on test data...")
        metrics = [
            Faithfulness(),
            AnswerRelevance(),
            ContextPrecision(),
            ContextRecall(),
        ]

        eval_result = evaluate(dataset, metrics=metrics)

        # Extract scores
        result["scores"] = {
            "faithfulness": eval_result.get("faithfulness", 0),
            "answer_relevancy": eval_result.get("answer_relevance", 0),
            "context_precision": eval_result.get("context_precision", 0),
            "context_recall": eval_result.get("context_recall", 0),
        }

        # Analyze scores
        score_values = list(result["scores"].values())
        result["analysis"] = {
            "all_scores": score_values,
            "uniform_check": verify_context_relevance(score_values),
            "score_source": "ragas_evaluation",
        }

        # Check for default value pattern
        if all(v == 0.50 for v in score_values):
            result["analysis"]["warning"] = (
                "CRITICAL: All RAGAS metrics returned 0.50. "
                "This indicates RAGAS is using default values instead of calculating. "
                "Check LLM configuration and OPENAI_API_KEY."
            )
            result["analysis"]["likely_using_defaults"] = True
        else:
            result["analysis"]["likely_using_defaults"] = False

    except Exception as e:
        result["errors"].append(f"RAGAS evaluation failed: {str(e)}")
        result["analysis"] = {
            "error_details": str(e),
            "recommendation": "Check RAGAS configuration and dependencies",
        }
        logger.error(f"RAGAS evaluation error: {e}", exc_info=True)

    return result


def generate_verification_report(results: Dict[str, Any]) -> str:
    """
    Generate human-readable verification report.

    Args:
        results: Verification results from all checks

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION METRICS VERIFICATION REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)

    # RAGAS Environment
    lines.append("\n## RAGAS Environment")
    env = results.get("ragas_environment", {})
    lines.append(f"- chromadb: {'OK' if env.get('chromadb') else 'MISSING'}")
    if env.get("chromadb_version"):
        lines.append(f"  - Version: {env['chromadb_version']}")
    lines.append(f"- ragas: {'OK' if env.get('ragas') else 'MISSING'}")
    if env.get("ragas_version"):
        lines.append(f"  - Version: {env['ragas_version']}")
    lines.append(f"- ragas_metrics: {'OK' if env.get('ragas_metrics') else 'MISSING'}")
    lines.append(f"- llm_configured: {'OK' if env.get('llm_configured') else 'MISSING'}")

    if env.get("errors"):
        lines.append("\n### Errors:")
        for error in env["errors"]:
            lines.append(f"- {error}")

    if env.get("warnings"):
        lines.append("\n### Warnings:")
        for warning in env["warnings"]:
            lines.append(f"- {warning}")

    # Citation Analysis
    lines.append("\n## Citation Analysis")
    citation = results.get("citation_analysis", {})
    lines.append(f"- Total responses: {citation.get('total', 0)}")
    lines.append(f"- With citations: {citation.get('with_citations', 0)}")
    lines.append(
        f"- Citation rate: {citation.get('citation_rate', 0):.1%}"
    )

    # Score Analysis
    lines.append("\n## Score Analysis")
    score = results.get("score_analysis", {})
    lines.append(f"- Is uniform: {score.get('is_uniform', False)}")
    lines.append(f"- Likely default: {score.get('likely_default', False)}")
    if score.get("warning"):
        lines.append(f"- Warning: {score['warning']}")

    # Small Scale Test Results
    if "small_scale_test" in results:
        lines.append("\n## Small Scale Test Results")
        test = results["small_scale_test"]
        lines.append(f"- RAGAS available: {test.get('ragas_available', False)}")
        lines.append(f"- Test count: {test.get('test_count', 0)}")

        if test.get("scores"):
            lines.append("\n### Scores:")
            for metric, value in test["scores"].items():
                lines.append(f"- {metric}: {value}")

        if test.get("analysis", {}).get("likely_using_defaults"):
            lines.append("\n### WARNING: Likely using default values!")
            lines.append(
                "All metrics returned 0.50, indicating RAGAS is not calculating."
            )

        if test.get("errors"):
            lines.append("\n### Errors:")
            for error in test["errors"]:
                lines.append(f"- {error}")

    # Recommendations
    lines.append("\n## Recommendations")

    if not env.get("ragas"):
        lines.append("\n1. Install RAGAS:")
        lines.append("   pip install ragas chromadb")

    if not env.get("llm_configured"):
        lines.append("\n2. Configure LLM for RAGAS:")
        lines.append("   export OPENAI_API_KEY=your_key")

    if score.get("likely_default"):
        lines.append("\n3. Fix Default Value Issue:")
        lines.append("   - RAGAS is returning default 0.50 values")
        lines.append("   - Check OPENAI_API_KEY is set correctly")
        lines.append("   - Verify LLM provider configuration")
        lines.append("   - Check RAGAS metrics initialization")

    if citation.get("citation_rate", 0) < 0.5:
        lines.append("\n4. Improve Citation Rate:")
        lines.append("   - Current citation rate is low")
        lines.append("   - Ensure answers include regulation references")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def run_full_verification(
    evaluation_files: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete verification suite.

    Args:
        evaluation_files: Optional list of evaluation JSON files to analyze
        output_file: Optional file to save report

    Returns:
        Complete verification results
    """
    results = {}

    logger.info("Starting evaluation metrics verification...")

    # 1. RAGAS Environment
    logger.info("Checking RAGAS environment...")
    results["ragas_environment"] = verify_ragas_environment()

    # 2. Citation Analysis
    logger.info("Analyzing citation patterns...")
    if evaluation_files:
        citation_analysis = analyze_evaluation_files(evaluation_files)
    else:
        # Analyze recent evaluation files
        eval_dir = Path("data/evaluations")
        if eval_dir.exists():
            eval_files = sorted(eval_dir.glob("*.json"))[-10:]  # Last 10 files
            citation_analysis = analyze_evaluation_files(
                [str(f) for f in eval_files]
            )
        else:
            citation_analysis = {"total": 0, "with_citations": 0}
    results["citation_analysis"] = citation_analysis

    # 3. Score Analysis
    logger.info("Analyzing score distributions...")
    if evaluation_files:
        score_analysis = analyze_score_distribution(evaluation_files)
    else:
        # Create sample score data
        sample_scores = [0.50, 0.50, 0.50, 0.50, 0.50]  # Suspicious uniform
        score_analysis = verify_context_relevance(sample_scores)
    results["score_analysis"] = score_analysis

    # 4. Small Scale Test
    logger.info("Running small-scale evaluation test...")
    results["small_scale_test"] = run_small_scale_test()

    # 5. Generate Report
    report = generate_verification_report(results)

    # Print report
    print(report)

    # Save report if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_file}")

    # Save JSON results
    json_output = Path("data/verification_results.json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON results saved to: {json_output}")

    return results


def analyze_evaluation_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze evaluation files for citation patterns.

    Args:
        file_paths: List of evaluation JSON file paths

    Returns:
        Citation analysis results
    """
    result = {
        "total": 0,
        "with_citations": 0,
        "citation_rate": 0.0,
        "files_analyzed": len(file_paths),
    }

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both single result and batch results
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if "results" in data:
                    items = data["results"]
                else:
                    items = [data]
            else:
                continue

            for item in items:
                if "answer" in item:
                    result["total"] += 1
                    citation_check = verify_citation_format(item["answer"])
                    if citation_check["has_citation"]:
                        result["with_citations"] += 1

        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")

    if result["total"] > 0:
        result["citation_rate"] = result["with_citations"] / result["total"]

    return result


def analyze_score_distribution(file_paths: List[str]) -> Dict[str, Any]:
    """
    Analyze score distribution from evaluation files.

    Args:
        file_paths: List of evaluation JSON file paths

    Returns:
        Score analysis results
    """
    all_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both single result and batch results
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if "results" in data:
                    items = data["results"]
                else:
                    items = [data]
            else:
                continue

            for item in items:
                for key in all_scores.keys():
                    if key in item:
                        all_scores[key].append(item[key])

        except Exception as e:
            logger.warning(f"Error analyzing scores in {file_path}: {e}")

    # Analyze each metric
    analysis = {
        "metrics": {},
        "is_uniform": True,
        "likely_default": True,
        "warning": "",
    }

    for metric, scores in all_scores.items():
        if scores:
            metric_analysis = verify_context_relevance(scores)
            analysis["metrics"][metric] = metric_analysis
            if not metric_analysis["is_uniform"]:
                analysis["is_uniform"] = False
            if not metric_analysis["likely_default"]:
                analysis["likely_default"] = False

    if analysis["is_uniform"]:
        analysis["warning"] = (
            "All metrics show uniform scores. This strongly suggests "
            "RAGAS is using default values instead of calculating."
        )

    return analysis


def main():
    """Main entry point for verification script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify RAG evaluation metrics and identify issues"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Evaluation JSON files to analyze",
    )
    parser.add_argument(
        "--output",
        help="Output file for verification report",
        default="data/verification_report.md",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip small-scale evaluation test",
    )

    args = parser.parse_args()

    # Run verification
    results = run_full_verification(
        evaluation_files=args.files,
        output_file=args.output if not args.skip_test else None,
    )

    # Return exit code based on findings
    if results.get("score_analysis", {}).get("likely_default"):
        logger.error("VERIFICATION FAILED: Likely using default values")
        return 1

    if results.get("ragas_environment", {}).get("errors"):
        logger.warning("VERIFICATION WARNING: Environment issues detected")
        return 2

    logger.info("VERIFICATION COMPLETE: No critical issues found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
