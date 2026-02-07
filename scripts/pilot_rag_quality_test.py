#!/usr/bin/env python3
"""
Comprehensive Pilot Test for RAG Quality Evaluation System.

This script executes a pilot test of the RAG quality evaluation system
using all 6 persona agents with 5 scenarios each (30 total scenarios).

Test Scope:
- 6 Persona Agents: freshman, graduate, professor, staff, parent, international
- 30 Scenarios: 5 scenarios per persona
- Categories: Simple, Complex, Edge cases mixed

Output:
- Results saved to data/evaluations/pilot_test_YYYYMMDD_HHMMSS.json
- Human-readable report in same directory

Usage:
    python scripts/pilot_rag_quality_test.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.rag.config import get_config
from src.rag.domain.evaluation.personas import PersonaManager
from src.rag.domain.evaluation.quality_evaluator import RAGQualityEvaluator
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_client import OpenAIClient
from src.rag.interface.query_handler import QueryHandler, QueryOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Test scenarios for each persona (5 scenarios each)
PERSONA_TEST_SCENARIOS = {
    "freshman": [
        {
            "query": "휴학 어떻게 해요?",
            "category": "simple",
            "ground_truth": "휴학 신청은 학사팀에 서류 제출",
        },
        {
            "query": "장학금 신청 방법 알려주실까요?",
            "category": "simple",
            "ground_truth": "장학금 신청은 포털에서 가능",
        },
        {
            "query": "성적이 나쁘면 휴학해야 하나요?",
            "category": "complex",
            "ground_truth": "성적 경고는 15학점 이상 수강且 평점 1.7 미만",
        },
        {
            "query": "처음이라 수강 신청 절차를 잘 몰라요",
            "category": "simple",
            "ground_truth": "수강 신청은 포털에서 학기별로 진행",
        },
        {
            "query": "복학 신청도 따로 해야 하나요?",
            "category": "edge",
            "ground_truth": "복학 신청은 휴학 종료 1달 전",
        },
    ],
    "graduate": [
        {
            "query": "연구년 자격 요건이 어떻게 되나요?",
            "category": "simple",
            "ground_truth": "연구년은 재직 6년 이상 교원에게 허용",
        },
        {
            "query": "연구비 지급 규정 확인 부탁드립니다",
            "category": "simple",
            "ground_truth": "연구비는 연구년 교원에게 연구비 지급",
        },
        {
            "query": "논문 제출 기한 연장 가능한가요?",
            "category": "complex",
            "ground_truth": "논문 제출 연장은 1년 한도로 가능",
        },
        {
            "query": "조교 근무 시간과 장학금 혜택 관련하여",
            "category": "complex",
            "ground_truth": "조교는 주 10시간 근무, 장학금 지급",
        },
        {
            "query": "등록금 면제 기준이 대학원마다 달라요?",
            "category": "edge",
            "ground_truth": "등록금 면제는 대학원별 기준 적용",
        },
    ],
    "professor": [
        {
            "query": "교원인사규정 제8조 확인 필요",
            "category": "simple",
            "ground_truth": "제8조는 의원면직 규정",
        },
        {
            "query": "연구년 적용 기준 상세히",
            "category": "complex",
            "ground_truth": "연구년은 재직기간 6년 이상, 성과 평가 우수",
        },
        {
            "query": "승진 심의 기준과 편장조 구체적 근거",
            "category": "complex",
            "ground_truth": "승진은 연구실적, 교육성과, 봉사활동 종합 평가",
        },
        {
            "query": "휴직 시 급여 지급 규정 해석 부탁드립니다",
            "category": "complex",
            "ground_truth": "휴직 중 급여는 무급, 공적 연금 납부",
        },
        {
            "query": "Sabbatical leave 규정과 국내 연구년 차이점",
            "category": "edge",
            "ground_truth": "Sabbatical은 해외 연구, 연구년은 국내 연구",
        },
    ],
    "staff": [
        {
            "query": "복무 규정 확인 부탁드립니다",
            "category": "simple",
            "ground_truth": "직원 복무는 주 40시간",
        },
        {
            "query": "휴가 신청 서식 양식 알려주세요",
            "category": "simple",
            "ground_truth": "연차 휴가는 연간 15일",
        },
        {
            "query": "급여 지급일과 처리 기한이 언제까지인가요?",
            "category": "simple",
            "ground_truth": "급여는 매월 25일 지급",
        },
        {
            "query": "사무용품 사용 규정과 승인 권한자 확인",
            "category": "complex",
            "ground_truth": "비품 구매는 100만원까지 팀장 승인",
        },
        {
            "query": "연수 참가 절차와 경비 처리 방법",
            "category": "complex",
            "ground_truth": "연수는 사전 승인, 실비 정산",
        },
    ],
    "parent": [
        {
            "query": "자녀 장학금 관련해서 알고 싶어요",
            "category": "simple",
            "ground_truth": "성적 우수 장학금, 근로 장학금 등 다양",
        },
        {
            "query": "기숙사 비용이 어떻게 되나요?",
            "category": "simple",
            "ground_truth": "기숙사비는 방 유형별로 상이",
        },
        {
            "query": "휴학 비용도 내야 하나요?",
            "category": "complex",
            "ground_truth": "휴학 중 등록금 면제, 기숙사비는 별도",
        },
        {
            "query": "성적 저하 시 학교에서 알려주나요?",
            "category": "edge",
            "ground_truth": "성적 경고는 학부모에게 통보 의무 없음",
        },
        {
            "query": "자녀 졸업 요건이 무엇인가요?",
            "category": "simple",
            "ground_truth": "졸업 요건은 학점 130점 이상, 논문 통과",
        },
    ],
    "international": [
        {
            "query": "How do I apply for student visa?",
            "category": "simple",
            "ground_truth": "유학생 비자는 입학 허가 후 발급",
        },
        {
            "query": "Tell me about dormitory procedure for international students",
            "category": "simple",
            "ground_truth": "유학생 기숙사는 우선 배정",
        },
        {
            "query": "Korean language program requirements",
            "category": "complex",
            "ground_truth": "한국어 과정은 TOPIK 3급 이상 권장",
        },
        {
            "query": " tuition fee payment in English version available?",
            "category": "complex",
            "ground_truth": "등록금 납부는 외국계좌 가능",
        },
        {
            "query": "비자 연장 절차와 관련 서류 뭐 필요한가요?",
            "category": "complex",
            "ground_truth": "비자 연장은 출입국사무소 방문, 재학증명서 필요",
        },
    ],
}


class RAGQualityPilotTest:
    """
    Comprehensive pilot test for RAG quality evaluation system.

    Executes 30 test scenarios across 6 personas and generates detailed report.
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        use_reranker: bool = True,
        judge_model: str = "gpt-4o",
        output_dir: str = "data/evaluations",
    ):
        """
        Initialize the pilot test.

        Args:
            json_path: Path to regulations JSON file
            use_reranker: Whether to use reranker
            judge_model: Judge LLM model for evaluation
            output_dir: Output directory for results
        """
        self.json_path = json_path or os.getenv("REGULATIONS_JSON_PATH", "data/output")
        self.use_reranker = use_reranker
        self.judge_model = judge_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize config
        self.config = get_config()

        # Initialize components
        logger.info("Initializing RAG system components...")
        self._initialize_components()

        # Initialize persona manager
        self.persona_manager = PersonaManager()
        logger.info(f"Loaded {len(self.persona_manager.list_personas())} personas")

        # Initialize quality evaluator
        self.evaluator = RAGQualityEvaluator(
            judge_model=self.judge_model,
            judge_api_key=os.getenv("OPENAI_API_KEY"),
            use_ragas=True,
        )
        logger.info(f"Quality evaluator initialized with {self.judge_model}")

        # Test results storage
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def _initialize_components(self):
        """Initialize RAG system components."""
        try:
            # Initialize vector store
            self.store = ChromaVectorStore(
                collection_name="regulations",
                persist_directory=str(Path("data/chroma_db").absolute()),
            )

            if self.store.count() == 0:
                logger.warning(
                    "Vector store is empty. Please run 'regulation sync' first."
                )
                raise ValueError("Vector store is empty")

            logger.info(f"Vector store loaded with {self.store.count()} documents")

            # Initialize LLM client
            self.llm_client = OpenAIClient()
            logger.info("LLM client initialized")

            # Initialize reranker (optional)
            if self.use_reranker:
                try:
                    from src.rag.infrastructure.reranker import BGEReranker

                    self.reranker = BGEReranker(
                        model_name=os.getenv(
                            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
                        )
                    )
                    logger.info("Reranker initialized")
                except Exception as e:
                    logger.warning(
                        f"Reranker initialization failed: {e}. Continuing without reranker."
                    )
                    self.reranker = None
            else:
                self.reranker = None

            # Initialize query handler
            self.query_handler = QueryHandler(
                store=self.store,
                llm_client=self.llm_client,
                use_reranker=self.use_reranker,
                json_path=self.json_path,
            )
            logger.info("Query handler initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def execute_test_scenario(
        self,
        persona_name: str,
        scenario: Dict[str, str],
        scenario_index: int,
    ) -> Dict[str, Any]:
        """
        Execute a single test scenario.

        Args:
            persona_name: Name of the persona
            scenario: Scenario dict with query, category, ground_truth
            scenario_index: Index of the scenario (1-5)

        Returns:
            Test result dict
        """
        query = scenario["query"]
        category = scenario["category"]
        ground_truth = scenario.get("ground_truth", "")

        logger.info(f"[{persona_name}] Scenario {scenario_index}: {query[:50]}...")

        try:
            # Execute RAG query
            result = self.query_handler.ask(
                question=query,
                options=QueryOptions(top_k=5, use_rerank=self.use_reranker),
            )

            if not result.success:
                return {
                    "persona": persona_name,
                    "scenario_index": scenario_index,
                    "query": query,
                    "category": category,
                    "error": result.content,
                    "passed": False,
                }

            # Extract contexts from sources
            contexts = []
            sources = result.data.get("sources", [])
            for source in sources:
                contexts.append(source.get("text", ""))

            # Evaluate quality
            evaluation = asyncio.run(
                self.evaluator.evaluate(
                    query=query,
                    answer=result.data.get("answer", result.content),
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
            )

            return {
                "persona": persona_name,
                "scenario_index": scenario_index,
                "query": query,
                "category": category,
                "ground_truth": ground_truth,
                "answer": result.data.get("answer", result.content),
                "sources": sources,
                "evaluation": {
                    "faithfulness": evaluation.faithfulness,
                    "answer_relevancy": evaluation.answer_relevancy,
                    "contextual_precision": evaluation.contextual_precision,
                    "contextual_recall": evaluation.contextual_recall,
                    "overall_score": evaluation.overall_score,
                    "passed": evaluation.passed,
                    "failure_reasons": evaluation.failure_reasons,
                    "metadata": evaluation.metadata,
                },
                "passed": evaluation.passed,
            }

        except Exception as e:
            logger.error(f"Error executing scenario: {e}")
            return {
                "persona": persona_name,
                "scenario_index": scenario_index,
                "query": query,
                "category": category,
                "error": str(e),
                "passed": False,
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test scenarios across all personas.

        Returns:
            Summary dict with all results
        """
        logger.info("=" * 60)
        logger.info("Starting Comprehensive Pilot Test")
        logger.info("=" * 60)

        total_scenarios = sum(
            len(scenarios) for scenarios in PERSONA_TEST_SCENARIOS.values()
        )
        logger.info(f"Total scenarios to execute: {total_scenarios}")

        current_scenario = 0

        for persona_name in self.persona_manager.list_personas():
            scenarios = PERSONA_TEST_SCENARIOS.get(persona_name, [])
            logger.info(
                f"\n--- Testing Persona: {persona_name} ({len(scenarios)} scenarios) ---"
            )

            for i, scenario in enumerate(scenarios, 1):
                current_scenario += 1
                logger.info(
                    f"\n[{current_scenario}/{total_scenarios}] Testing {persona_name} - Scenario {i}"
                )

                result = self.execute_test_scenario(persona_name, scenario, i)
                self.test_results.append(result)

        # Calculate summary statistics
        summary = self._calculate_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.get("passed", False))

        # Calculate average scores
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []
        overall_scores = []

        for result in self.test_results:
            if "evaluation" in result:
                eval_data = result["evaluation"]
                faithfulness_scores.append(eval_data.get("faithfulness", 0))
                relevancy_scores.append(eval_data.get("answer_relevancy", 0))
                precision_scores.append(eval_data.get("contextual_precision", 0))
                recall_scores.append(eval_data.get("contextual_recall", 0))
                overall_scores.append(eval_data.get("overall_score", 0))

        # Calculate per-persona statistics
        persona_stats = {}
        for persona_name in self.persona_manager.list_personas():
            persona_results = [
                r for r in self.test_results if r["persona"] == persona_name
            ]
            persona_passed = sum(1 for r in persona_results if r.get("passed", False))
            persona_total = len(persona_results)

            if persona_results:
                persona_avg = sum(
                    r.get("evaluation", {}).get("overall_score", 0)
                    for r in persona_results
                    if "evaluation" in r
                ) / len([r for r in persona_results if "evaluation" in r])
            else:
                persona_avg = 0.0

            persona_stats[persona_name] = {
                "total": persona_total,
                "passed": persona_passed,
                "failed": persona_total - persona_passed,
                "pass_rate": persona_passed / persona_total if persona_total > 0 else 0,
                "avg_score": persona_avg,
            }

        # Calculate per-category statistics
        category_stats = {}
        for category in ["simple", "complex", "edge"]:
            category_results = [
                r for r in self.test_results if r.get("category") == category
            ]
            category_passed = sum(1 for r in category_results if r.get("passed", False))
            category_total = len(category_results)

            if category_results:
                category_avg = sum(
                    r.get("evaluation", {}).get("overall_score", 0)
                    for r in category_results
                    if "evaluation" in r
                ) / len([r for r in category_results if "evaluation" in r])
            else:
                category_avg = 0.0

            category_stats[category] = {
                "total": category_total,
                "passed": category_passed,
                "failed": category_total - category_passed,
                "pass_rate": category_passed / category_total
                if category_total > 0
                else 0,
                "avg_score": category_avg,
            }

        # Identify failures
        failures = [
            r
            for r in self.test_results
            if not r.get("passed", False) and "evaluation" in r
        ]

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        return {
            "total_scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "average_scores": {
                "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores)
                if faithfulness_scores
                else 0,
                "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores)
                if relevancy_scores
                else 0,
                "contextual_precision": sum(precision_scores) / len(precision_scores)
                if precision_scores
                else 0,
                "contextual_recall": sum(recall_scores) / len(recall_scores)
                if recall_scores
                else 0,
                "overall": sum(overall_scores) / len(overall_scores)
                if overall_scores
                else 0,
            },
            "persona_stats": persona_stats,
            "category_stats": category_stats,
            "failures": [
                {
                    "persona": f["persona"],
                    "query": f["query"],
                    "reasons": f.get("evaluation", {}).get("failure_reasons", []),
                }
                for f in failures
            ],
            "test_duration_seconds": duration,
            "timestamp": end_time.isoformat(),
        }

    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to JSON and generate human-readable report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = self.output_dir / f"pilot_test_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "results": self.test_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Results saved to {json_path}")

        # Generate human-readable report
        report_path = self.output_dir / f"pilot_test_report_{timestamp}.md"
        self._generate_report(summary, report_path)
        logger.info(f"Report saved to {report_path}")

    def _generate_report(self, summary: Dict[str, Any], report_path: Path):
        """Generate human-readable markdown report."""
        lines = []
        lines.append("# RAG Quality Pilot Test Report")
        lines.append("")
        lines.append(f"**Generated:** {summary['timestamp']}")
        lines.append(
            f"**Test Duration:** {summary['test_duration_seconds']:.2f} seconds"
        )
        lines.append("")

        # Summary Statistics
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"- **Total Scenarios:** {summary['total_scenarios']}")
        lines.append(f"- **Passed:** {summary['passed']}")
        lines.append(f"- **Failed:** {summary['failed']}")
        lines.append(f"- **Pass Rate:** {summary['pass_rate']:.1%}")
        lines.append("")

        # Average Scores
        lines.append("## Average Scores")
        lines.append("")
        lines.append("| Metric | Score | Threshold |")
        lines.append("|--------|-------|-----------|")
        for metric, score in summary["average_scores"].items():
            lines.append(f"| {metric.replace('_', ' ').title()} | {score:.3f} | 0.80 |")
        lines.append("")

        # Per-Persona Breakdown
        lines.append("## Per-Persona Breakdown")
        lines.append("")
        lines.append("| Persona | Total | Passed | Failed | Pass Rate | Avg Score |")
        lines.append("|---------|-------|--------|--------|-----------|-----------|")
        for persona, stats in summary["persona_stats"].items():
            lines.append(
                f"| {persona} | {stats['total']} | {stats['passed']} | "
                f"{stats['failed']} | {stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
            )
        lines.append("")

        # Per-Category Breakdown
        lines.append("## Per-Category Breakdown")
        lines.append("")
        lines.append("| Category | Total | Passed | Failed | Pass Rate | Avg Score |")
        lines.append("|----------|-------|--------|--------|-----------|-----------|")
        for category, stats in summary["category_stats"].items():
            lines.append(
                f"| {category.title()} | {stats['total']} | {stats['passed']} | "
                f"{stats['failed']} | {stats['pass_rate']:.1%} | {stats['avg_score']:.3f} |"
            )
        lines.append("")

        # Failure Analysis
        if summary["failures"]:
            lines.append("## Failure Analysis")
            lines.append("")
            for failure in summary["failures"]:
                lines.append(f"### Persona: {failure['persona']}")
                lines.append(f"**Query:** {failure['query']}")
                lines.append("**Failure Reasons:**")
                for reason in failure["reasons"]:
                    lines.append(f"- {reason}")
                lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        pass_rate = summary["pass_rate"]
        if pass_rate >= 0.9:
            lines.append("- Excellent quality! System is performing well.")
        elif pass_rate >= 0.7:
            lines.append("- Good quality with room for improvement.")
        elif pass_rate >= 0.5:
            lines.append("- Moderate quality. Significant improvements needed.")
        else:
            lines.append("- Poor quality. Major system improvements required.")
        lines.append("")

        # Save report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive pilot test for RAG quality evaluation"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to regulations JSON file",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranker",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge LLM model for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/evaluations",
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        pilot_test = RAGQualityPilotTest(
            json_path=args.json_path,
            use_reranker=not args.no_reranker,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
        )

        summary = pilot_test.run_all_tests()

        # Print summary to console
        print("\n" + "=" * 60)
        print("PILOT TEST COMPLETE")
        print("=" * 60)
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Average Score: {summary['average_scores']['overall']:.3f}")
        print("=" * 60)

        sys.exit(0 if summary["pass_rate"] >= 0.7 else 1)

    except Exception as e:
        logger.error(f"Pilot test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
