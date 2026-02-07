#!/usr/bin/env python3
"""
Real LLM-as-Judge Evaluation Test for RAG Quality Evaluation System.

This script executes real LLM-as-Judge evaluation using GPT-4o on all 30 pilot test scenarios.
Compares Mock vs Real evaluation results and generates comprehensive reports.

Test Scope:
- 6 Persona Agents: freshman, graduate, professor, staff, parent, international
- 30 Scenarios: 5 scenarios per persona
- Categories: Simple, Complex, Edge cases mixed

Output:
- Real evaluation results: data/evaluations/real_evaluation_YYYYMMDD_HHMMSS.json
- Human-readable report: data/evaluations/real_evaluation_report_YYYYMMDD_HHMMSS.md
- Mock vs Real comparison: data/evaluations/mock_vs_real_comparison_YYYYMMDD_HHMMSS.md

Usage:
    python scripts/real_evaluation_test.py [--stage STAGE] [--output-dir DIR]
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
from src.rag.domain.evaluation.models import EvaluationThresholds
from src.rag.domain.evaluation.personas import PersonaManager
from src.rag.domain.evaluation.quality_evaluator import RAGQualityEvaluator
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_client import OpenAIClient
from src.rag.interface.query_handler import QueryHandler, QueryOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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


class RealEvaluationTest:
    """
    Real LLM-as-Judge evaluation test for RAG quality evaluation system.

    Executes 30 test scenarios with real LLM-as-Judge evaluation and generates reports.
    """

    def __init__(
        self,
        json_path: Optional[str] = None,
        use_reranker: bool = True,
        judge_model: str = "gpt-4o",
        output_dir: str = "data/evaluations",
        stage: int = 1,
        use_ragas: bool = True,
    ):
        """
        Initialize the real evaluation test.

        Args:
            json_path: Path to regulations JSON file
            use_reranker: Whether to use reranker
            judge_model: Judge LLM model for evaluation
            output_dir: Output directory for results
            stage: Evaluation stage (1=initial, 2=intermediate, 3=target)
            use_ragas: Whether to use RAGAS library
        """
        self.json_path = json_path or os.getenv("REGULATIONS_JSON_PATH", "data/output")
        self.use_reranker = use_reranker
        self.judge_model = judge_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.use_ragas = use_ragas

        # Initialize config
        self.config = get_config()

        # Initialize components
        logger.info("Initializing RAG system components...")
        self._initialize_components()

        # Initialize persona manager
        self.persona_manager = PersonaManager()
        logger.info(f"Loaded {len(self.persona_manager.list_personas())} personas")

        # Initialize quality evaluator with stage-based thresholds
        thresholds = EvaluationThresholds.for_stage(stage)
        self.evaluator = RAGQualityEvaluator(
            judge_model=self.judge_model,
            judge_api_key=os.getenv("OPENAI_API_KEY"),
            thresholds=thresholds,
            use_ragas=self.use_ragas,
            stage=stage,
        )
        stage_name = thresholds.get_current_stage_name()
        logger.info(
            f"Quality evaluator initialized with {self.judge_model} "
            f"(Stage: {stage} - {stage_name})"
        )

        # Test results storage
        self.test_results: List[Dict[str, Any]] = []
        self.mock_results: List[Dict[str, Any]] = []
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
                logger.warning("Vector store is empty. Please run 'regulation sync' first.")
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
                        model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
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
        use_real_evaluation: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a single test scenario.

        Args:
            persona_name: Name of the persona
            scenario: Scenario dict with query, category, ground_truth
            scenario_index: Index of the scenario (1-5)
            use_real_evaluation: Whether to use real LLM-as-Judge evaluation

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
                    "evaluation_method": "real" if use_real_evaluation else "mock",
                }

            # Extract contexts from sources
            contexts = []
            sources = result.data.get("sources", [])
            for source in sources:
                contexts.append(source.get("text", ""))

            # Evaluate quality (real or mock)
            if use_real_evaluation:
                evaluation = asyncio.run(
                    self.evaluator.evaluate(
                        query=query,
                        answer=result.data.get("answer", result.content),
                        contexts=contexts,
                        ground_truth=ground_truth,
                    )
                )
            else:
                # Mock evaluation
                from src.rag.domain.evaluation.models import EvaluationResult

                evaluation = EvaluationResult(
                    query=query,
                    answer=result.data.get("answer", result.content),
                    contexts=contexts,
                    faithfulness=0.502,
                    answer_relevancy=0.557,
                    contextual_precision=0.500,
                    contextual_recall=0.500,
                    overall_score=0.515,
                    passed=False,
                    failure_reasons=["Mock evaluation - thresholds not met"],
                    metadata={"evaluation_method": "mock", "evaluation_type": "mock"},
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
                "evaluation_method": "real" if use_real_evaluation else "mock",
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
                "evaluation_method": "real" if use_real_evaluation else "mock",
            }

    def run_all_tests(self, run_mock_comparison: bool = True) -> Dict[str, Any]:
        """
        Run all test scenarios across all personas.

        Args:
            run_mock_comparison: Whether to also run mock evaluation for comparison

        Returns:
            Summary dict with all results
        """
        logger.info("=" * 60)
        logger.info("Starting Real LLM-as-Judge Evaluation Test")
        logger.info("=" * 60)

        total_scenarios = sum(
            len(scenarios) for scenarios in PERSONA_TEST_SCENARIOS.values()
        )
        logger.info(f"Total scenarios to execute: {total_scenarios}")
        logger.info(f"Evaluation Stage: {self.stage}")
        logger.info(f"Using RAGAS: {self.use_ragas}")

        current_scenario = 0

        # Run real evaluation
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

                result = self.execute_test_scenario(
                    persona_name, scenario, i, use_real_evaluation=True
                )
                self.test_results.append(result)

        # Run mock evaluation for comparison (if requested)
        if run_mock_comparison:
            logger.info("\n" + "=" * 60)
            logger.info("Running Mock Evaluation for Comparison")
            logger.info("=" * 60)

            current_scenario = 0
            for persona_name in self.persona_manager.list_personas():
                scenarios = PERSONA_TEST_SCENARIOS.get(persona_name, [])

                for i, scenario in enumerate(scenarios, 1):
                    current_scenario += 1

                    result = self.execute_test_scenario(
                        persona_name, scenario, i, use_real_evaluation=False
                    )
                    self.mock_results.append(result)

        # Calculate summary statistics
        summary = self._calculate_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.get("passed", False))

        # Calculate average scores for real evaluation
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

        # Mock vs Real comparison
        comparison = self._calculate_mock_vs_real_comparison()

        # Identify failures
        failures = [
            r
            for r in self.test_results
            if not r.get("passed", False) and "evaluation" in r
        ]

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        return {
            "evaluation_type": "real_llm_judge",
            "stage": self.stage,
            "stage_name": EvaluationThresholds.for_stage(self.stage).get_current_stage_name(),
            "thresholds": EvaluationThresholds.for_stage(self.stage).get_thresholds_for_stage(
                self.stage
            ),
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
            "mock_vs_real_comparison": comparison,
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
            "judge_model": self.judge_model,
            "use_ragas": self.use_ragas,
        }

    def _calculate_mock_vs_real_comparison(self) -> Dict[str, Any]:
        """Calculate comparison between mock and real evaluation results."""
        if not self.mock_results:
            return {}

        # Mock averages
        mock_faithfulness = []
        mock_relevancy = []
        mock_precision = []
        mock_recall = []
        mock_overall = []

        for result in self.mock_results:
            if "evaluation" in result:
                eval_data = result["evaluation"]
                mock_faithfulness.append(eval_data.get("faithfulness", 0))
                mock_relevancy.append(eval_data.get("answer_relevancy", 0))
                mock_precision.append(eval_data.get("contextual_precision", 0))
                mock_recall.append(eval_data.get("contextual_recall", 0))
                mock_overall.append(eval_data.get("overall_score", 0))

        # Real averages (already calculated in main summary)
        real_faithfulness = []
        real_relevancy = []
        real_precision = []
        real_recall = []
        real_overall = []

        for result in self.test_results:
            if "evaluation" in result:
                eval_data = result["evaluation"]
                real_faithfulness.append(eval_data.get("faithfulness", 0))
                real_relevancy.append(eval_data.get("answer_relevancy", 0))
                real_precision.append(eval_data.get("contextual_precision", 0))
                real_recall.append(eval_data.get("contextual_recall", 0))
                real_overall.append(eval_data.get("overall_score", 0))

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "mock": {
                "faithfulness": avg(mock_faithfulness),
                "answer_relevancy": avg(mock_relevancy),
                "contextual_precision": avg(mock_precision),
                "contextual_recall": avg(mock_recall),
                "overall": avg(mock_overall),
            },
            "real": {
                "faithfulness": avg(real_faithfulness),
                "answer_relevancy": avg(real_relevancy),
                "contextual_precision": avg(real_precision),
                "contextual_recall": avg(real_recall),
                "overall": avg(real_overall),
            },
            "diff": {
                "faithfulness": avg(real_faithfulness) - avg(mock_faithfulness),
                "answer_relevancy": avg(real_relevancy) - avg(mock_relevancy),
                "contextual_precision": avg(real_precision) - avg(mock_precision),
                "contextual_recall": avg(real_recall) - avg(mock_recall),
                "overall": avg(real_overall) - avg(mock_overall),
            },
        }

    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to JSON and generate human-readable reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = self.output_dir / f"real_evaluation_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "real_results": self.test_results,
                    "mock_results": self.mock_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Results saved to {json_path}")

        # Generate human-readable report
        report_path = self.output_dir / f"real_evaluation_report_{timestamp}.md"
        self._generate_report(summary, report_path)
        logger.info(f"Report saved to {report_path}")

        # Generate comparison report
        if self.mock_results:
            comparison_path = self.output_dir / f"mock_vs_real_comparison_{timestamp}.md"
            self._generate_comparison_report(summary, comparison_path)
            logger.info(f"Comparison report saved to {comparison_path}")

    def _generate_report(self, summary: Dict[str, Any], report_path: Path):
        """Generate human-readable markdown report."""
        lines = []
        lines.append("# Real LLM-as-Judge Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {summary['timestamp']}")
        lines.append(
            f"**Test Duration:** {summary['test_duration_seconds']:.2f} seconds"
        )
        lines.append(f"**Evaluation Stage:** {summary['stage']} - {summary['stage_name']}")
        lines.append(f"**Judge Model:** {summary.get('judge_model', 'N/A')}")
        lines.append(f"**Using RAGAS:** {summary.get('use_ragas', False)}")
        lines.append("")

        # Threshold Information
        lines.append("## Evaluation Thresholds")
        lines.append("")
        thresholds = summary.get("thresholds", {})
        lines.append(f"- **Faithfulness:** {thresholds.get('faithfulness', 'N/A')}")
        lines.append(f"- **Answer Relevancy:** {thresholds.get('answer_relevancy', 'N/A')}")
        lines.append(
            f"- **Contextual Precision:** {thresholds.get('contextual_precision', 'N/A')}"
        )
        lines.append(
            f"- **Contextual Recall:** {thresholds.get('contextual_recall', 'N/A')}"
        )
        lines.append(f"- **Overall Pass:** {thresholds.get('overall_pass', 'N/A')}")
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
        lines.append("| Metric | Score | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")
        avg_scores = summary["average_scores"]
        for metric, score in avg_scores.items():
            metric_name = metric.replace("_", " ").title()
            threshold = thresholds.get(metric, "N/A")
            status = "PASS" if score >= threshold else "FAIL"
            lines.append(f"| {metric_name} | {score:.3f} | {threshold} | {status} |")
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

        # Mock vs Real Comparison
        if summary.get("mock_vs_real_comparison"):
            lines.append("## Mock vs Real Comparison")
            lines.append("")
            comparison = summary["mock_vs_real_comparison"]
            lines.append("| Metric | Mock | Real | Diff |")
            lines.append("|--------|------|------|------|")
            for metric in ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "overall"]:
                mock_val = comparison["mock"].get(metric, 0)
                real_val = comparison["real"].get(metric, 0)
                diff = comparison["diff"].get(metric, 0)
                diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
                lines.append(
                    f"| {metric.replace('_', ' ').title()} | {mock_val:.3f} | {real_val:.3f} | {diff_str} |"
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

        # Improvement Recommendations
        lines.append("## Improvement Recommendations")
        lines.append("")
        self._add_improvement_recommendations(lines, summary)
        lines.append("")

        # Save report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _generate_comparison_report(self, summary: Dict[str, Any], comparison_path: Path):
        """Generate detailed mock vs real comparison report."""
        lines = []
        lines.append("# Mock vs Real Evaluation Comparison Report")
        lines.append("")
        lines.append(f"**Generated:** {summary['timestamp']}")
        lines.append("")

        comparison = summary.get("mock_vs_real_comparison", {})
        if not comparison:
            lines.append("No comparison data available.")
            with open(comparison_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

        # Overall Comparison
        lines.append("## Overall Score Comparison")
        lines.append("")
        lines.append("| Metric | Mock Score | Real Score | Difference | Improvement |")
        lines.append("|--------|------------|------------|------------|-------------|")

        for metric in ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "overall"]:
            mock_val = comparison["mock"].get(metric, 0)
            real_val = comparison["real"].get(metric, 0)
            diff = comparison["diff"].get(metric, 0)
            improvement = "Yes" if diff > 0 else "No"
            diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
            lines.append(
                f"| {metric.replace('_', ' ').title()} | {mock_val:.3f} | {real_val:.3f} | {diff_str} | {improvement} |"
            )
        lines.append("")

        # Key Findings
        lines.append("## Key Findings")
        lines.append("")
        overall_diff = comparison["diff"].get("overall", 0)
        if overall_diff > 0:
            lines.append(f"- Real evaluation is **better** than mock by +{overall_diff:.3f}")
        elif overall_diff < 0:
            lines.append(f"- Real evaluation is **worse** than mock by {overall_diff:.3f}")
        else:
            lines.append("- Real evaluation and mock scores are **similar**")
        lines.append("")

        # Analysis by metric
        lines.append("## Analysis by Metric")
        lines.append("")

        # Faithfulness
        faith_diff = comparison["diff"].get("faithfulness", 0)
        lines.append(f"### Faithfulness (Diff: {faith_diff:+.3f})")
        if faith_diff > 0.1:
            lines.append("- Real LLM-as-Judge evaluation shows **significantly better** factual consistency")
        elif faith_diff < -0.1:
            lines.append("- Real evaluation shows **worse** factual consistency - investigate hallucination")
        else:
            lines.append("- Minimal difference between mock and real evaluation")
        lines.append("")

        # Answer Relevancy
        relevancy_diff = comparison["diff"].get("answer_relevancy", 0)
        lines.append(f"### Answer Relevancy (Diff: {relevancy_diff:+.3f})")
        if relevancy_diff > 0.1:
            lines.append("- Real evaluation shows **better** query response quality")
        elif relevancy_diff < -0.1:
            lines.append("- Real evaluation shows **worse** response quality - improve generation prompts")
        else:
            lines.append("- Minimal difference in answer relevancy")
        lines.append("")

        # Contextual Precision
        precision_diff = comparison["diff"].get("contextual_precision", 0)
        lines.append(f"### Contextual Precision (Diff: {precision_diff:+.3f})")
        if precision_diff > 0.1:
            lines.append("- Real evaluation shows **better** retrieval ranking")
        elif precision_diff < -0.1:
            lines.append("- Real evaluation shows **worse** retrieval ranking - tune reranker")
        else:
            lines.append("- Minimal difference in retrieval ranking")
        lines.append("")

        # Contextual Recall
        recall_diff = comparison["diff"].get("contextual_recall", 0)
        lines.append(f"### Contextual Recall (Diff: {recall_diff:+.3f})")
        if recall_diff > 0.1:
            lines.append("- Real evaluation shows **better** information completeness")
        elif recall_diff < -0.1:
            lines.append("- Real evaluation shows **worse** completeness - increase top_k")
        else:
            lines.append("- Minimal difference in information completeness")
        lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        self._add_improvement_recommendations(lines, summary)
        lines.append("")

        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _add_improvement_recommendations(self, lines: List[str], summary: Dict[str, Any]):
        """Add improvement recommendations to report."""
        avg_scores = summary.get("average_scores", {})
        thresholds = summary.get("thresholds", {})

        # Faithfulness recommendations
        faith_score = avg_scores.get("faithfulness", 0)
        faith_threshold = thresholds.get("faithfulness", 0.8)
        if faith_score < faith_threshold:
            lines.append(
                f"- **Faithfulness ({faith_score:.3f} < {faith_threshold}):** "
                "Consider reducing LLM temperature, strengthening anti-hallucination prompts"
            )

        # Answer Relevancy recommendations
        relevancy_score = avg_scores.get("answer_relevancy", 0)
        relevancy_threshold = thresholds.get("answer_relevancy", 0.8)
        if relevancy_score < relevancy_threshold:
            lines.append(
                f"- **Answer Relevancy ({relevancy_score:.3f} < {relevancy_threshold}):** "
                "Improve answer generation prompts, ensure direct query addressing"
            )

        # Contextual Precision recommendations
        precision_score = avg_scores.get("contextual_precision", 0)
        precision_threshold = thresholds.get("contextual_precision", 0.75)
        if precision_score < precision_threshold:
            lines.append(
                f"- **Contextual Precision ({precision_score:.3f} < {precision_threshold}):** "
                "Tune reranker threshold, improve retrieval ranking algorithm"
            )

        # Contextual Recall recommendations
        recall_score = avg_scores.get("contextual_recall", 0)
        recall_threshold = thresholds.get("contextual_recall", 0.75)
        if recall_score < recall_threshold:
            lines.append(
                f"- **Contextual Recall ({recall_score:.3f} < {recall_threshold}):** "
                "Increase top_k for retrieval, expand context window"
            )

        # Stage progression recommendation
        pass_rate = summary.get("pass_rate", 0)
        if pass_rate >= 0.8:
            current_stage = summary.get("stage", 1)
            if current_stage < 3:
                next_stage = current_stage + 1
                lines.append(
                    f"- **Stage Progression:** Current pass rate ({pass_rate:.1%}) exceeds 80%. "
                    f"Consider progressing to Stage {next_stage} for higher thresholds."
                )

        if not any(
            avg_scores.get(k, 0) < thresholds.get(k, 1.0)
            for k in ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]
        ):
            lines.append("- **Excellent Quality!** All metrics meet or exceed thresholds.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run real LLM-as-Judge evaluation for RAG quality testing"
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
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Evaluation stage (1=initial, 2=intermediate, 3=target)",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Disable RAGAS library use (fallback to custom implementation)",
    )
    parser.add_argument(
        "--skip-mock-comparison",
        action="store_true",
        help="Skip mock evaluation comparison",
    )

    args = parser.parse_args()

    try:
        evaluation_test = RealEvaluationTest(
            json_path=args.json_path,
            use_reranker=not args.no_reranker,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
            stage=args.stage,
            use_ragas=not args.no_ragas,
        )

        summary = evaluation_test.run_all_tests(
            run_mock_comparison=not args.skip_mock_comparison
        )

        # Print summary to console
        print("\n" + "=" * 60)
        print("REAL EVALUATION TEST COMPLETE")
        print("=" * 60)
        print(f"Stage: {args.stage} - {summary['stage_name']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Average Score: {summary['average_scores']['overall']:.3f}")
        print("=" * 60)

        # Show mock vs real comparison if available
        if summary.get("mock_vs_real_comparison"):
            comparison = summary["mock_vs_real_comparison"]
            print("\nMock vs Real Comparison:")
            for metric in ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "overall"]:
                mock_val = comparison["mock"].get(metric, 0)
                real_val = comparison["real"].get(metric, 0)
                diff = comparison["diff"].get(metric, 0)
                print(f"  {metric}: Mock={mock_val:.3f}, Real={real_val:.3f}, Diff={diff:+.3f}")
            print("=" * 60)

        sys.exit(0 if summary["pass_rate"] >= 0.6 else 1)

    except Exception as e:
        logger.error(f"Real evaluation test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
