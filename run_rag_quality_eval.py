#!/usr/bin/env python3
"""
RAG Quality Evaluation Runner using Parallel Persona Agents.

Executes 6 persona sub-agents in parallel to evaluate RAG system quality.
Implements the rag-quality-local skill specification.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.config import get_config
from src.rag.domain.evaluation.llm_judge import EvaluationBatch, LLMJudge
from src.rag.domain.evaluation.parallel_evaluator import (
    PersonaEvaluationResult,
)
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.interface.query_handler import QueryHandler, QueryOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 6 personas with their test queries
PERSONA_TEST_QUERIES = {
    "student-undergraduate": [
        "휴학 방법 알려줘",
        "등록금 언제까지 납부해요?",
        "성적 조회 어떻게 해요?",
        "장학금 신청하는 법",
        "수강신청 기간이 언제인가요?",
    ],
    "student-graduate": [
        "연구년 신청 관련 규정 확인 부탁드립니다",
        "연구비 지원 요건이 어떻게 되나요?",
        "논문 심사 절차 상세히 알려주세요",
        "조교 신청 자격 설명해주세요",
        "등록금 납부 유예 관련 문의",
    ],
    "professor": [
        "휴직 관련 조항 확인 필요",
        "연구년 적용 기준 상세히",
        "교원 승진 관련 편/장/조 구체적 근거",
        "연구비 집행 규정 해석 부탁드립니다",
        "Sabbatical 관련 예외 사항 확인",
    ],
    "staff-admin": [
        "휴가 업무 처리 절차 확인",
        "급여 관련 서식 양식 알려주세요",
        "연수 승인 권한자가 누구인가요?",
        "사무용품 사용 규정 안내",
        "복무 처리 기한이 언제까지인가요?",
    ],
    "parent": [
        "자녀 등록금 관련해서 알고 싶어요",
        "장학금 부모님도 알아야 하나요?",
        "기숙사 비용이 어떻게 되나요?",
        "휴학 신청은 부모가 해야 하나요?",
        "졸업 관련 서류 뭐 필요한가요?",
    ],
    "student-international": [
        "How do I apply for leave of absence?",
        "Visa related tuition payment procedure",
        "Dormitory requirements for international students",
        "Tell me about scholarship in English if possible",
        "Course registration English version available?",
    ],
}


async def run_parallel_evaluation(
    db_path: str = "data/chroma_db",
    queries_per_persona: int = 5,
    max_workers: int = 6,
    use_reranker: bool = True,
) -> Dict[str, PersonaEvaluationResult]:
    """Run parallel persona evaluation.

    Args:
        db_path: Path to ChromaDB
        queries_per_persona: Number of queries to test per persona
        max_workers: Maximum parallel workers
        use_reranker: Whether to use reranker

    Returns:
        Dictionary mapping persona IDs to evaluation results
    """
    logger.info("Starting parallel RAG quality evaluation...")

    # Initialize components
    config = get_config()
    llm_client = LLMClientAdapter(
        provider=config.llm_provider,
        model=config.llm_model,
        base_url=config.llm_base_url,
    )

    store = ChromaVectorStore(persist_directory=db_path)
    query_handler = QueryHandler(
        store=store,
        llm_client=llm_client,
        use_reranker=use_reranker,
    )

    judge = LLMJudge(llm_client=llm_client)
    batch = EvaluationBatch(judge=judge)

    # Evaluate each persona
    results = {}

    for persona_id, queries in PERSONA_TEST_QUERIES.items():
        logger.info(f"Evaluating persona: {persona_id}")

        persona_results = []
        queries_to_test = queries[:queries_per_persona]

        for query in queries_to_test:
            try:
                # Execute query through RAG system
                options = QueryOptions(
                    top_k=5,
                    use_rerank=use_reranker,
                    force_mode="ask",
                )

                result = query_handler.process_query(
                    query=query,
                    options=options,
                )

                # Extract answer and sources
                answer_text = result.content if result.success else ""
                sources = []

                if result.data and "tool_results" in result.data:
                    for tool_result in result.data.get("tool_results", []):
                        if tool_result.get("tool_name") == "search_regulations":
                            result_data = tool_result.get("result")
                            if result_data and isinstance(result_data, dict):
                                search_results = result_data.get("results", [])
                                for r in search_results[:5]:
                                    if isinstance(r, dict):
                                        sources.append({
                                            "title": r.get("title", "") or r.get("regulation_title", ""),
                                            "text": (r.get("text", "") or r.get("content", ""))[:200],
                                            "rule_code": r.get("rule_code", ""),
                                            "score": r.get("score", 0.0) or r.get("similarity", 0.0),
                                        })
                                break

                # Evaluate with LLM judge
                judge_result = judge.evaluate(
                    query=query,
                    answer=answer_text,
                    sources=sources,
                )

                batch.add_result(judge_result)
                persona_results.append(judge_result)

                logger.info(f"  Query: {query[:30]}... Score: {judge_result.overall_score:.3f} ({'PASS' if judge_result.passed else 'FAIL'})")

            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")

        # Calculate persona summary
        if persona_results:
            avg_score = sum(r.overall_score for r in persona_results) / len(persona_results)
            pass_count = sum(1 for r in persona_results if r.passed)

            # Count issues
            issues = {}
            for r in persona_results:
                for issue in r.issues:
                    issues[issue] = issues.get(issue, 0) + 1

            results[persona_id] = PersonaEvaluationResult(
                persona=persona_id,
                queries_tested=len(persona_results),
                results=persona_results,
                avg_score=avg_score,
                pass_rate=pass_count / len(persona_results),
                issues=issues,
            )

    return results, batch


def save_evaluation_results(
    results: Dict[str, PersonaEvaluationResult],
    batch: EvaluationBatch,
    output_dir: str = "data/evaluations",
) -> str:
    """Save evaluation results to JSON file.

    Args:
        results: Persona evaluation results
        batch: Evaluation batch with all results
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"rag_quality_eval_{timestamp}.json")

    # Get batch summary
    summary = batch.get_summary()

    # Prepare data
    data = {
        "evaluation_id": f"rag_quality_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "parallel_persona_evaluation",
        "personas_tested": list(results.keys()),
        "summary": {
            "total_queries": summary.total_queries,
            "passed": summary.passed,
            "failed": summary.failed,
            "pass_rate": summary.pass_rate,
            "avg_accuracy": summary.avg_accuracy,
            "avg_completeness": summary.avg_completeness,
            "avg_citations": summary.avg_citations,
            "avg_context_relevance": summary.avg_context_relevance,
            "avg_overall_score": summary.avg_overall_score,
        },
        "persona_results": {},
        "failure_patterns": summary.failure_patterns,
    }

    # Add persona-specific results
    for persona_id, result in results.items():
        data["persona_results"][persona_id] = {
            "queries_tested": result.queries_tested,
            "avg_score": result.avg_score,
            "pass_rate": result.pass_rate,
            "issues": result.issues,
            "results": [
                {
                    "query": r.query,
                    "answer": r.answer[:200] + "..." if len(r.answer) > 200 else r.answer,
                    "accuracy": r.accuracy,
                    "completeness": r.completeness,
                    "citations": r.citations,
                    "context_relevance": r.context_relevance,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "issues": r.issues,
                    "strengths": r.strengths,
                }
                for r in result.results
            ],
        }

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved evaluation results to: {filepath}")
    return filepath


def generate_markdown_report(
    results: Dict[str, PersonaEvaluationResult],
    batch: EvaluationBatch,
    eval_id: str,
) -> str:
    """Generate markdown evaluation report.

    Args:
        results: Persona evaluation results
        batch: Evaluation batch
        eval_id: Evaluation ID

    Returns:
        Markdown report string
    """
    summary = batch.get_summary()

    lines = [
        "# RAG Quality Evaluation Report",
        f"**Evaluation ID:** {eval_id}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total Queries** | {summary.total_queries} |",
        f"| **Passed** | {summary.passed} |",
        f"| **Failed** | {summary.failed} |",
        f"| **Pass Rate** | {summary.pass_rate:.1%} |",
        "",
        "## Average Scores",
        "",
        "| Metric | Score | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| **Overall** | {summary.avg_overall_score:.3f} | 0.800 | {'✓' if summary.avg_overall_score >= 0.8 else '✗'} |",
        f"| **Accuracy** | {summary.avg_accuracy:.3f} | 0.850 | {'✓' if summary.avg_accuracy >= 0.85 else '✗'} |",
        f"| **Completeness** | {summary.avg_completeness:.3f} | 0.750 | {'✓' if summary.avg_completeness >= 0.75 else '✗'} |",
        f"| **Citations** | {summary.avg_citations:.3f} | 0.700 | {'✓' if summary.avg_citations >= 0.7 else '✗'} |",
        f"| **Context Relevance** | {summary.avg_context_relevance:.3f} | 0.750 | {'✓' if summary.avg_context_relevance >= 0.75 else '✗'} |",
        "",
        "## Results by Persona",
        "",
    ]

    # Add persona-specific results
    for persona_id, result in sorted(results.items(), key=lambda x: x[1].avg_score, reverse=True):
        lines.extend([
            f"### {persona_id}",
            "",
            f"- **Queries Tested:** {result.queries_tested}",
            f"- **Average Score:** {result.avg_score:.3f}",
            f"- **Pass Rate:** {result.pass_rate:.1%}",
            "",
        ])

        if result.issues:
            lines.extend([
                "**Issues:**",
                ""
            ])
            for issue, count in sorted(result.issues.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {issue}: {count}x")
            lines.append("")

    # Add failure patterns
    lines.extend([
        "## Top Failure Patterns",
        "",
    ])

    for issue, count in sorted(summary.failure_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"- {issue}: {count}x")

    lines.extend([
        "",
        "## Detailed Query Results",
        "",
    ])

    # Add detailed results
    for i, result in enumerate(batch.results, 1):
        lines.extend([
            f"### {i}. {result.query}",
            f"**Score:** {result.overall_score:.3f} ({'PASS' if result.passed else 'FAIL'})",
            f"**Metrics:** Acc={result.accuracy:.3f}, Comp={result.completeness:.3f}, Cit={result.citations:.3f}, Ctx={result.context_relevance:.3f}",
        ])

        if result.issues:
            lines.append(f"**Issues:** {', '.join(result.issues)}")
        if result.strengths:
            lines.append(f"**Strengths:** {', '.join(result.strengths)}")

        lines.extend([
            f"**Answer:** {result.answer[:200]}...",
            "",
        ])

    return "\n".join(lines)


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("RAG Quality Evaluation - Parallel Persona Agents")
    logger.info("=" * 60)

    # Run evaluation
    results, batch = await run_parallel_evaluation(
        db_path="data/chroma_db",
        queries_per_persona=5,
        max_workers=6,
        use_reranker=True,
    )

    # Save results
    filepath = save_evaluation_results(results, batch)

    # Generate report
    summary = batch.get_summary()
    eval_id = f"rag_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = generate_markdown_report(results, batch, eval_id)

    # Save report
    report_path = filepath.replace(".json", "_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Saved report to: {report_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Total Queries: {summary.total_queries}")
    logger.info(f"Passed: {summary.passed}")
    logger.info(f"Failed: {summary.failed}")
    logger.info(f"Pass Rate: {summary.pass_rate:.1%}")
    logger.info(f"Average Score: {summary.avg_overall_score:.3f}")
    logger.info("=" * 60)

    # Print report to console
    print("\n" + report)


if __name__ == "__main__":
    asyncio.run(main())
