#!/usr/bin/env python3
"""
Comprehensive RAG Quality Evaluation Script using ParallelPersonaEvaluator.

Executes evaluation across all 6 personas with 5 queries per persona.
Generates comprehensive reports with metrics and failure pattern analysis.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.domain.evaluation.parallel_evaluator import ParallelPersonaEvaluator


def main():
    """Execute comprehensive RAG quality evaluation."""
    print("=" * 80)
    print("RAG Quality Evaluation - Parallel Persona Assessment")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize evaluator
    print("Initializing ParallelPersonaEvaluator...")
    evaluator = ParallelPersonaEvaluator()
    print("✓ Evaluator initialized")
    print()

    # Define personas to evaluate
    personas = [
        "student-undergraduate",
        "student-graduate",
        "professor",
        "staff-admin",
        "parent",
        "student-international",
    ]

    print(f"Personas to evaluate: {', '.join(personas)}")
    print(f"Queries per persona: 5")
    print(f"Total queries: {len(personas) * 5}")
    print()

    # Execute parallel evaluation
    print("-" * 80)
    print("Starting parallel evaluation...")
    print("-" * 80)

    results = evaluator.evaluate_parallel(
        queries_per_persona=5,
        personas=personas,
        max_workers=6,
    )

    print()
    print("✓ Evaluation completed")
    print()

    # Generate summary statistics
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    total_queries = sum(r.queries_tested for r in results.values())
    total_passed = sum(sum(1 for r in results[p].results if r.passed) for p in results)
    total_failed = total_queries - total_passed

    # Calculate overall metrics
    all_scores = []
    all_accuracy = []
    all_completeness = []
    all_citations = []
    all_context_relevance = []

    for persona_result in results.values():
        for result in persona_result.results:
            all_scores.append(result.overall_score)
            all_accuracy.append(result.accuracy)
            all_completeness.append(result.completeness)
            all_citations.append(result.citations)
            all_context_relevance.append(result.context_relevance)

    avg_overall = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_accuracy = sum(all_accuracy) / len(all_accuracy) if all_accuracy else 0
    avg_completeness = sum(all_completeness) / len(all_completeness) if all_completeness else 0
    avg_citations = sum(all_citations) / len(all_citations) if all_citations else 0
    avg_context = sum(all_context_relevance) / len(all_context_relevance) if all_context_relevance else 0
    pass_rate = total_passed / total_queries if total_queries > 0 else 0

    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Pass Rate: {pass_rate:.1%}")
    print()
    print("Average Scores:")
    print(f"  Overall:      {avg_overall:.3f}")
    print(f"  Accuracy:     {avg_accuracy:.3f}")
    print(f"  Completeness: {avg_completeness:.3f}")
    print(f"  Citations:    {avg_citations:.3f}")
    print(f"  Context Rel:  {avg_context:.3f}")
    print()

    # Per-persona breakdown
    print("-" * 80)
    print("PER-PERSONA BREAKDOWN")
    print("-" * 80)

    for persona_id, persona_result in sorted(results.items()):
        persona_name = {
            "student-undergraduate": "Undergraduate Student",
            "student-graduate": "Graduate Student",
            "professor": "Professor",
            "staff-admin": "Administrative Staff",
            "parent": "Parent",
            "student-international": "International Student",
        }.get(persona_id, persona_id)

        persona_avg = sum(r.overall_score for r in persona_result.results) / len(persona_result.results)
        persona_pass = sum(1 for r in persona_result.results if r.passed)
        persona_pass_rate = persona_pass / persona_result.queries_tested

        print(f"{persona_name:25} | Queries: {persona_result.queries_tested:2} | "
              f"Avg: {persona_avg:.3f} | Pass: {persona_pass_rate:.1%}")

    print()

    # Failure pattern analysis
    print("-" * 80)
    print("FAILURE PATTERN ANALYSIS")
    print("-" * 80)

    all_issues = {}
    for persona_result in results.values():
        for issue, count in persona_result.issues.items():
            all_issues[issue] = all_issues.get(issue, 0) + count

    if all_issues:
        print("Top Issues:")
        for issue, count in sorted(all_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  • {issue}: {count} occurrences")
    else:
        print("  No issues detected!")
    print()

    # Save results
    print("-" * 80)
    print("Saving results...")
    print("-" * 80)

    json_path = evaluator.save_results()
    print(f"✓ JSON results saved to: {json_path}")

    # Generate markdown report
    report = evaluator.generate_report()

    # Save markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = f"data/evaluations/comprehensive_report_{timestamp}.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✓ Markdown report saved to: {md_path}")
    print()

    # Print full report to console
    print("=" * 80)
    print("FULL EVALUATION REPORT")
    print("=" * 80)
    print()
    print(report)

    print()
    print("=" * 80)
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return {
        "json_path": json_path,
        "md_path": md_path,
        "total_queries": total_queries,
        "pass_rate": pass_rate,
        "avg_overall": avg_overall,
    }


if __name__ == "__main__":
    try:
        result = main()
        print()
        print("Evaluation Summary:")
        print(f"  Results: {result['json_path']}")
        print(f"  Report: {result['md_path']}")
        print(f"  Pass Rate: {result['pass_rate']:.1%}")
        print(f"  Avg Score: {result['avg_overall']:.3f}")
        sys.exit(0)
    except Exception as e:
        print()
        print(f"ERROR: Evaluation failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
