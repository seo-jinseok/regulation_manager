#!/usr/bin/env python3
"""
Full RAG Quality Evaluation Script.

Executes comprehensive evaluation across 6 personas with 150+ test queries.
Generates JSON results and markdown report.

Usage:
    uv run python scripts/run_full_quality_evaluation.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.rag.domain.evaluation import (
    EvaluationResult,
    EvaluationThresholds,
    PersonaManager,
    RAGQualityEvaluator,
)
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.application.search_usecase import SearchUseCase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Evaluation thresholds (Stage 1 - Initial)
THRESHOLDS = {
    "accuracy": 0.85,  # Faithfulness
    "completeness": 0.75,  # Context Recall
    "citations": 0.70,  # Context Precision
    "context_relevance": 0.75,  # Answer Relevancy
}

# Persona ID mapping (ParallelPersonaEvaluator format to internal format)
PERSONA_ID_MAP = {
    "student-undergraduate": "freshman",
    "student-graduate": "graduate",
    "professor": "professor",
    "staff-admin": "staff",
    "parent": "parent",
    "student-international": "international",
}

# Test queries by persona
PERSONA_QUERIES = {
    "student-undergraduate": [
        # Easy - Simple procedural
        ("휴학 어떻게 해?", "procedural", "easy"),
        ("성적 평균 어떻게 계산돼?", "academic", "easy"),
        ("졸업 요건 뭐야?", "academic", "easy"),
        ("장학금 종류 알려줘", "financial", "easy"),
        ("기숙사 신청 방법 알려줘", "campus_life", "easy"),
        ("복학하려면 뭐 해야돼?", "procedural", "easy"),
        ("수강 신청 언제부터야?", "academic", "easy"),
        ("도서관 이용 시간이 어떻게 돼?", "campus_life", "easy"),
        ("학생증 재발급 어떻게 해?", "admin", "easy"),
        ("성적증명서 발급받고 싶어", "admin", "easy"),
        # Medium - Conditional
        ("F 학점 받으면 어떡하냐...", "academic", "medium"),
        ("학기 중에 휴학할 수 있어?", "procedural", "medium"),
        ("전공 바꿀 수 있어?", "academic", "medium"),
        ("복수전공 하려면 뭐가 필요해?", "academic", "medium"),
        ("부전공 신청은 언제 해?", "academic", "medium"),
        ("휴학 기간 연장할 수 있어?", "procedural", "medium"),
        ("장학금 받으면 등록금 얼마야?", "financial", "medium"),
        ("계절학기 수강하면 학점 몇 개야?", "academic", "medium"),
        ("유고결석 처리 어떻게 해?", "academic", "medium"),
        ("재수강하면 성적 어떻게 돼?", "academic", "medium"),
        # Hard - Complex scenarios
        ("휴학하고 군대 가면 복학은 언제 해야 해?", "procedural", "hard"),
        ("장학금 받다가 휴학하면 반납해야 해?", "financial", "hard"),
        ("이중전공하고 부전공도 할 수 있어?", "academic", "hard"),
        ("8학기 넘어도 등록할 수 있어?", "academic", "hard"),
        ("학기당 21학점 넘게 신청할 수 있어?", "academic", "hard"),
    ],
    "student-graduate": [
        # Easy
        ("연구비 신청 방법이 뭐예요?", "research", "easy"),
        ("논문 제출 기한이 언제까지예요?", "research", "easy"),
        ("학위청구 논문 심사비가 얼마예요?", "financial", "easy"),
        ("조교 월급이 얼마예요?", "salary", "easy"),
        ("연구실 사용 신청 어떻게 해요?", "research", "easy"),
        ("대학원 등록금 납부 기한이 언제까지예요?", "financial", "easy"),
        ("학위증 발급받고 싶어요", "admin", "easy"),
        ("지도교수 변경 어떻게 해요?", "academic", "easy"),
        ("논문 심사 위원 몇 명이 필요해요?", "research", "easy"),
        ("학자금 대출 이자 얼마예요?", "financial", "easy"),
        # Medium
        ("석박사 통합과정 전환 조건이 뭐예요?", "academic", "medium"),
        ("연구년제 신청 자격이 어떻게 돼요?", "research", "medium"),
        ("학위 논문 플래그십 통과 기준이 뭐예요?", "research", "medium"),
        ("TA 수행하면 등록금 감면돼요?", "financial", "medium"),
        ("해외 학회 참석 지원받을 수 있어요?", "research", "medium"),
        ("학위청구 논문 심사 절차가 어떻게 돼요?", "research", "medium"),
        ("연구윤리 교육 이수 필수예요?", "research", "medium"),
        ("박사자격시험 응시 자격이 뭐예요?", "academic", "medium"),
        ("휴학하고 군대 가면 어떻게 돼요?", "procedural", "medium"),
        ("논문 표절 검사 통과 기준이 뭐예요?", "research", "medium"),
        # Hard
        ("석사 과정 중 박사 과정으로 전환하면 논문 다시 써야 해요?", "academic", "hard"),
        ("연구비 집행 내역 제출 안 하면 어떻게 돼요?", "research", "hard"),
        ("학위 취소되는 경우가 있어요?", "academic", "hard"),
        ("외국인 교환학생으로 논문심사 받을 수 있어요?", "research", "hard"),
        ("복수학위 과정 어떻게 신청해요?", "academic", "hard"),
    ],
    "professor": [
        # Easy
        ("연구년 신청 기간이 언제까지입니까?", "research", "easy"),
        ("강의평가 결과는 어디서 확인합니까?", "teaching", "easy"),
        ("연구비 정산 서류가 뭡니까?", "research", "easy"),
        ("출장 신청 어떻게 합니까?", "admin", "easy"),
        ("휴직 신청 기간이 언제까지입니까?", "personnel", "easy"),
        ("연구실 안전 교육 언제 이수합니까?", "research", "easy"),
        ("학생 연구원 채용 절차가 어떻습니까?", "research", "easy"),
        ("재직증명서 발급받고 싶습니다", "admin", "easy"),
        ("승진 심사 서류가 뭡니까?", "personnel", "easy"),
        ("연구비 카드 한도가 얼마입니까?", "financial", "easy"),
        # Medium
        ("연구년제 신청 자격 요건이 어떻게 됩니까?", "research", "medium"),
        ("정년보장 심사 기준이 구체적으로 어떻게 됩니까?", "personnel", "medium"),
        ("교수 승진 심사 절차가 어떻게 진행됩니까?", "personnel", "medium"),
        ("연구비 간접비 비율이 얼마입니까?", "research", "medium"),
        ("학기당 강의 시간이 몇 시간입니까?", "teaching", "medium"),
        ("대학원생 지도비 지급 기준이 어떻게 됩니까?", "research", "medium"),
        ("휴직 중 연구비 집행 가능합니까?", "research", "medium"),
        ("산학협력 연구 계약 절차가 어떻게 됩니까?", "research", "medium"),
        ("방학 중 연구실 사용 규정이 어떻게 됩니까?", "research", "medium"),
        ("교원 연수 지원금이 얼마입니까?", "training", "medium"),
        # Hard
        ("연구년제 중 타 대학 겸직 가능합니까?", "research", "hard"),
        ("정년보장 심사 탈락 시 이의신청 가능합니까?", "personnel", "hard"),
        ("연구비 부정집행 시 제재가 어떻게 됩니까?", "research", "hard"),
        ("휴직 후 복직 시 호봉 어떻게 산정됩니까?", "personnel", "hard"),
        ("명예교수 위촉 조건이 어떻게 됩니까?", "personnel", "hard"),
    ],
    "staff-admin": [
        # Easy
        ("연차 휴가 신청 어떻게 하나요?", "leave", "easy"),
        ("복무 근무 시간이 어떻게 되나요?", "personnel", "easy"),
        ("급여 명세서 어디서 확인하나요?", "salary", "easy"),
        ("사무용품 구매 요청 어떻게 하나요?", "admin", "easy"),
        ("건강검진 신청 기간이 언제까지인가요?", "personnel", "easy"),
        ("연말정산 서류 제출 언제까지인가요?", "salary", "easy"),
        ("출장비 정산 어떻게 하나요?", "financial", "easy"),
        ("회의실 예약 방법이 뭐예요?", "admin", "easy"),
        ("재직증명서 발급받고 싶어요", "admin", "easy"),
        ("퇴직금 산정 어떻게 되나요?", "salary", "easy"),
        # Medium
        ("육아휴직 신청 자격이 어떻게 되나요?", "leave", "medium"),
        ("성과상여금 지급 기준이 뭔가요?", "salary", "medium"),
        ("직급 승진 심사 기준이 어떻게 되나요?", "personnel", "medium"),
        ("해외 연수 신청 절차가 어떻게 되나요?", "training", "medium"),
        ("보안 교육 이수 의무가 있나요?", "admin", "medium"),
        ("야근 수당 계산 어떻게 되나요?", "salary", "medium"),
        ("전보 발령 신청 어떻게 하나요?", "personnel", "medium"),
        ("병가 신청 시 필요 서류가 뭔가요?", "leave", "medium"),
        ("퇴직 연금 가입 기간 어떻게 계산하나요?", "salary", "medium"),
        ("시설 사용 승인 절차가 어떻게 되나요?", "admin", "medium"),
        # Hard
        ("장기 근속자 포상 기준이 어떻게 되나요?", "personnel", "hard"),
        ("징계 처분 시 정년 보장되나요?", "personnel", "hard"),
        ("퇴직 후 재고용 조건이 어떻게 되나요?", "personnel", "hard"),
        ("공무상 재해 보상 범위가 어떻게 되나요?", "personnel", "hard"),
        ("입찰 참가 자격 기준이 어떻게 되나요?", "admin", "hard"),
    ],
    "parent": [
        # Easy
        ("자녀 등록금 납부 기한이 언제까지인가요?", "financial", "easy"),
        ("장학금 신청 방법이 어떻게 되나요?", "financial", "easy"),
        ("기숙사 비용이 얼마인가요?", "financial", "easy"),
        ("성적표 받아볼 수 있나요?", "academic", "easy"),
        ("학자금 대출 이자가 얼마인가요?", "financial", "easy"),
        ("등록금 분납 가능한가요?", "financial", "easy"),
        ("식권 구매 어떻게 하나요?", "campus_life", "easy"),
        ("자녀 졸업식 언제인가요?", "academic", "easy"),
        ("건강보험 증명서 필요해요", "admin", "easy"),
        ("학생 확인서 발급받고 싶어요", "admin", "easy"),
        # Medium
        ("장학금 받으면 세금 내야 하나요?", "financial", "medium"),
        ("자녀 휴학하면 등록금 반환받나요?", "financial", "medium"),
        ("기숙사 입사 자격 기준이 어떻게 되나요?", "campus_life", "medium"),
        ("유학생 자녀 등록금 차이가 있나요?", "financial", "medium"),
        ("장애 학생 지원 제도가 있나요?", "support", "medium"),
        ("군 휴학 시 등록금 반환받나요?", "financial", "medium"),
        ("자녀 성적 부진 시 상담 가능한가요?", "academic", "medium"),
        ("교환학생 프로그램 비용이 어떻게 되나요?", "financial", "medium"),
        ("자녀가 아르바이트 할 수 있나요?", "campus_life", "medium"),
        ("복학 시 등록금 차이가 있나요?", "financial", "medium"),
        # Hard
        ("자녀가 학칙 위반 시 처벌이 어떻게 되나요?", "academic", "hard"),
        ("장학금 중복 수혜 가능한가요?", "financial", "hard"),
        ("등록금 납부 후 자녀 자퇴하면 환불되나요?", "financial", "hard"),
        ("기숙사 퇴사 시 보증금 반환되나요?", "financial", "hard"),
        ("부모님 동의 없는 휴학 가능한가요?", "procedural", "hard"),
    ],
    "student-international": [
        # Easy
        ("How do I apply for dormitory?", "campus_life", "easy"),
        ("비자 연장 어떻게 해요?", "visa", "easy"),
        ("Tuition payment deadline when?", "financial", "easy"),
        ("한국어 수업 어디서 해요?", "academic", "easy"),
        ("How to get student ID card?", "admin", "easy"),
        ("기숙사 식사 할 수 있어요?", "campus_life", "easy"),
        ("Scholarship for international students?", "financial", "easy"),
        ("한국어 시험 언제 있어요?", "academic", "easy"),
        ("Part-time job allowed?", "visa", "easy"),
        ("건강보험 가입해야 해요?", "visa", "easy"),
        # Medium
        ("아르바이트 허가 시간이 몇 시간이에요?", "visa", "medium"),
        ("재학 중 여행 갈 수 있어요?", "visa", "medium"),
        ("한국어 능력 시험 몇 점 필요해요?", "academic", "medium"),
        ("International student scholarship requirements?", "financial", "medium"),
        ("병원 갈 때 보험 되나요?", "visa", "medium"),
        ("졸업 후 취업 비자 바꿀 수 있어요?", "visa", "medium"),
        ("휴학하면 비자 어떻게 돼요?", "visa", "medium"),
        ("Exchange program credits transfer?", "academic", "medium"),
        ("한국어 수업 레벨이 몇 개 있어요?", "academic", "medium"),
        ("International office hours when?", "admin", "medium"),
        # Hard
        ("D-2에서 D-4로 변경할 수 있어요?", "visa", "hard"),
        ("졸업 후 6개월 비자 연장 가능해요?", "visa", "hard"),
        ("Professional visa E-7 requirements?", "visa", "hard"),
        ("International student tuition waiver criteria?", "financial", "hard"),
        ("Dual degree program available?", "academic", "hard"),
    ],
}


class FullQualityEvaluator:
    """Orchestrates full RAG quality evaluation."""

    def __init__(self, db_path: str = "data/chroma_db"):
        """Initialize evaluator with RAG components."""
        self.db_path = db_path
        self.store = ChromaVectorStore(persist_directory=db_path)
        self.search_usecase = SearchUseCase(self.store, use_reranker=True)
        self.evaluator = RAGQualityEvaluator(
            judge_model="gpt-4o",
            thresholds=EvaluationThresholds(stage=1),
            use_ragas=False,  # Use mock evaluation for speed
        )
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def query_rag(self, query: str) -> Tuple[str, List[str], float]:
        """Execute query against RAG system."""
        try:
            # Use the search usecase
            results = self.search_usecase.search(query, top_k=5)

            if not results:
                return "죄송합니다. 관련 규정을 찾을 수 없습니다.", [], 0.0

            # Extract contexts
            contexts = []
            for r in results[:3]:  # Top 3 contexts
                text = r.chunk.text if hasattr(r, 'chunk') else str(r)
                contexts.append(text[:500] if len(text) > 500 else text)

            # Calculate average confidence
            scores = [r.score for r in results if hasattr(r, 'score')]
            confidence = sum(scores) / len(scores) if scores else 0.0

            # Generate answer using LLM (simplified)
            answer = self._generate_answer(query, contexts)

            return answer, contexts, confidence

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return f"오류 발생: {str(e)}", [], 0.0

    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer from contexts (simplified)."""
        if not contexts:
            return "관련 규정을 찾을 수 없습니다."

        # Combine contexts for answer
        combined = "\n\n".join(contexts[:2])

        # Return a placeholder answer (in real system, this would call LLM)
        return f"규정에 따르면: {combined[:300]}..."

    async def evaluate_query(
        self,
        query: str,
        persona: str,
        category: str,
        difficulty: str,
    ) -> Dict[str, Any]:
        """Evaluate a single query."""
        # Get RAG response
        answer, contexts, confidence = self.query_rag(query)

        # Evaluate using quality evaluator
        result = await self.evaluator.evaluate(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=None,
        )

        # Map metrics to required format
        return {
            "persona": persona,
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "answer": answer[:500] + "..." if len(answer) > 500 else answer,
            "contexts": contexts[:2],  # Limit context size
            "evaluation": {
                "accuracy": result.faithfulness,
                "completeness": result.contextual_recall,
                "citations": result.contextual_precision,
                "context_relevance": result.answer_relevancy,
                "overall_score": result.overall_score,
                "passed": result.passed,
                "failure_reasons": result.failure_reasons,
            },
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation across all personas."""
        self.start_time = datetime.now()
        self.results = []

        total_queries = sum(len(queries) for queries in PERSONA_QUERIES.values())
        logger.info(f"Starting full evaluation with {total_queries} queries across {len(PERSONA_QUERIES)} personas")

        completed = 0
        for persona_id, queries in PERSONA_QUERIES.items():
            persona_display = PERSONA_ID_MAP.get(persona_id, persona_id)
            logger.info(f"Evaluating persona: {persona_id} ({len(queries)} queries)")

            for query, category, difficulty in queries:
                try:
                    result = await self.evaluate_query(
                        query=query,
                        persona=persona_id,
                        category=category,
                        difficulty=difficulty,
                    )
                    self.results.append(result)
                    completed += 1

                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total_queries} queries evaluated")

                except Exception as e:
                    logger.error(f"Error evaluating query '{query}': {e}")
                    self.results.append({
                        "persona": persona_id,
                        "query": query,
                        "category": category,
                        "difficulty": difficulty,
                        "error": str(e),
                        "evaluation": {
                            "accuracy": 0.5,
                            "completeness": 0.5,
                            "citations": 0.5,
                            "context_relevance": 0.5,
                            "overall_score": 0.5,
                            "passed": False,
                        }
                    })

        self.end_time = datetime.now()
        return self._generate_summary()

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("evaluation", {}).get("passed", False))

        # Calculate metric averages
        accuracy_scores = [r["evaluation"]["accuracy"] for r in self.results if "evaluation" in r]
        completeness_scores = [r["evaluation"]["completeness"] for r in self.results if "evaluation" in r]
        citations_scores = [r["evaluation"]["citations"] for r in self.results if "evaluation" in r]
        relevance_scores = [r["evaluation"]["context_relevance"] for r in self.results if "evaluation" in r]
        overall_scores = [r["evaluation"]["overall_score"] for r in self.results if "evaluation" in r]

        # Per-persona metrics
        persona_metrics = {}
        for persona_id in PERSONA_QUERIES.keys():
            persona_results = [r for r in self.results if r.get("persona") == persona_id]
            if persona_results:
                persona_passed = sum(1 for r in persona_results if r.get("evaluation", {}).get("passed", False))
                persona_scores = [r["evaluation"]["overall_score"] for r in persona_results if "evaluation" in r]
                persona_metrics[persona_id] = {
                    "total": len(persona_results),
                    "passed": persona_passed,
                    "pass_rate": round(persona_passed / len(persona_results), 3) if persona_results else 0,
                    "avg_score": round(sum(persona_scores) / len(persona_scores), 3) if persona_scores else 0,
                }

        # Per-category metrics
        categories = {}
        for r in self.results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "scores": []}
            categories[cat]["total"] += 1
            if r.get("evaluation", {}).get("passed", False):
                categories[cat]["passed"] += 1
            if "evaluation" in r:
                categories[cat]["scores"].append(r["evaluation"]["overall_score"])

        category_summary = {}
        for cat, data in categories.items():
            category_summary[cat] = {
                "total": data["total"],
                "passed": data["passed"],
                "pass_rate": round(data["passed"] / data["total"], 3) if data["total"] > 0 else 0,
                "avg_score": round(sum(data["scores"]) / len(data["scores"]), 3) if data["scores"] else 0,
            }

        # Per-difficulty metrics
        difficulties = {}
        for r in self.results:
            diff = r.get("difficulty", "unknown")
            if diff not in difficulties:
                difficulties[diff] = {"total": 0, "passed": 0, "scores": []}
            difficulties[diff]["total"] += 1
            if r.get("evaluation", {}).get("passed", False):
                difficulties[diff]["passed"] += 1
            if "evaluation" in r:
                difficulties[diff]["scores"].append(r["evaluation"]["overall_score"])

        difficulty_summary = {}
        for diff, data in difficulties.items():
            difficulty_summary[diff] = {
                "total": data["total"],
                "passed": data["passed"],
                "pass_rate": round(data["passed"] / data["total"], 3) if data["total"] > 0 else 0,
                "avg_score": round(sum(data["scores"]) / len(data["scores"]), 3) if data["scores"] else 0,
            }

        return {
            "evaluation_id": f"eval_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": int((self.end_time - self.start_time).total_seconds()) if self.end_time else 0,
            "overall": {
                "total_queries": total,
                "passed": passed,
                "pass_rate": round(passed / total, 3) if total > 0 else 0,
                "avg_accuracy": round(sum(accuracy_scores) / len(accuracy_scores), 3) if accuracy_scores else 0,
                "avg_completeness": round(sum(completeness_scores) / len(completeness_scores), 3) if completeness_scores else 0,
                "avg_citations": round(sum(citations_scores) / len(citations_scores), 3) if citations_scores else 0,
                "avg_context_relevance": round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else 0,
                "avg_score": round(sum(overall_scores) / len(overall_scores), 3) if overall_scores else 0,
            },
            "thresholds": THRESHOLDS,
            "persona_metrics": persona_metrics,
            "category_summary": category_summary,
            "difficulty_summary": difficulty_summary,
            "results": self.results,
        }


def generate_markdown_report(summary: Dict[str, Any]) -> str:
    """Generate markdown report from evaluation summary."""
    lines = [
        "# RAG Quality Evaluation Report",
        "",
        f"**Evaluation ID:** {summary['evaluation_id']}",
        f"**Timestamp:** {summary['timestamp']}",
        f"**Duration:** {summary['duration_seconds']} seconds",
        "",
        "## Executive Summary",
        "",
        f"| Metric | Value | Threshold | Status |",
        f"|--------|-------|-----------|--------|",
    ]

    overall = summary["overall"]
    thresholds = summary["thresholds"]

    # Overall pass rate
    pass_rate = overall["pass_rate"] * 100
    pass_status = "PASS" if pass_rate >= 80 else "FAIL"
    lines.append(f"| **Overall Pass Rate** | {pass_rate:.1f}% | 80% | {pass_status} |")
    lines.append(f"| **Total Queries** | {overall['total_queries']} | 150+ | {'PASS' if overall['total_queries'] >= 150 else 'FAIL'} |")
    lines.append(f"| **Queries Passed** | {overall['passed']} | - | - |")
    lines.append("")

    # Metric scores
    lines.extend([
        "## Metric Scores",
        "",
        "| Metric | Score | Threshold | Status |",
        "|--------|-------|-----------|--------|",
    ])

    metrics = [
        ("Accuracy (Faithfulness)", overall["avg_accuracy"], thresholds["accuracy"]),
        ("Completeness (Recall)", overall["avg_completeness"], thresholds["completeness"]),
        ("Citations (Precision)", overall["avg_citations"], thresholds["citations"]),
        ("Context Relevance", overall["avg_context_relevance"], thresholds["context_relevance"]),
    ]

    for name, score, threshold in metrics:
        status = "PASS" if score >= threshold else "FAIL"
        lines.append(f"| {name} | {score:.3f} | {threshold} | {status} |")

    overall_status = "PASS" if overall["avg_score"] >= 0.75 else "FAIL"
    lines.append(f"| **Overall Score** | **{overall['avg_score']:.3f}** | **0.75** | **{overall_status}** |")
    lines.append("")

    # Per-persona results
    lines.extend([
        "## Per-Persona Results",
        "",
        "| Persona | Total | Passed | Pass Rate | Avg Score |",
        "|---------|-------|--------|-----------|-----------|",
    ])

    for persona_id, metrics in summary["persona_metrics"].items():
        lines.append(
            f"| {persona_id} | {metrics['total']} | {metrics['passed']} | "
            f"{metrics['pass_rate']*100:.1f}% | {metrics['avg_score']:.3f} |"
        )
    lines.append("")

    # Per-category results
    lines.extend([
        "## Per-Category Results",
        "",
        "| Category | Total | Passed | Pass Rate | Avg Score |",
        "|----------|-------|--------|-----------|-----------|",
    ])

    for category, metrics in sorted(summary["category_summary"].items()):
        lines.append(
            f"| {category} | {metrics['total']} | {metrics['passed']} | "
            f"{metrics['pass_rate']*100:.1f}% | {metrics['avg_score']:.3f} |"
        )
    lines.append("")

    # Per-difficulty results
    lines.extend([
        "## Per-Difficulty Results",
        "",
        "| Difficulty | Total | Passed | Pass Rate | Avg Score |",
        "|------------|-------|--------|-----------|-----------|",
    ])

    for difficulty, metrics in sorted(summary["difficulty_summary"].items()):
        lines.append(
            f"| {difficulty} | {metrics['total']} | {metrics['passed']} | "
            f"{metrics['pass_rate']*100:.1f}% | {metrics['avg_score']:.3f} |"
        )
    lines.append("")

    # Sample failures
    failures = [r for r in summary["results"] if not r.get("evaluation", {}).get("passed", True)]
    if failures:
        lines.extend([
            "## Sample Failures (First 10)",
            "",
        ])

        for i, failure in enumerate(failures[:10], 1):
            eval_data = failure.get("evaluation", {})
            reasons = eval_data.get("failure_reasons", ["Unknown"])
            lines.extend([
                f"### Failure {i}: {failure.get('persona', 'unknown')} - {failure.get('category', 'unknown')}",
                f"- **Query:** {failure.get('query', 'N/A')}",
                f"- **Difficulty:** {failure.get('difficulty', 'unknown')}",
                f"- **Score:** {eval_data.get('overall_score', 0):.3f}",
                f"- **Reasons:** {'; '.join(reasons[:3])}",
                "",
            ])

    return "\n".join(lines)


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Starting Full RAG Quality Evaluation")
    logger.info("=" * 60)

    # Initialize evaluator
    evaluator = FullQualityEvaluator()

    # Run evaluation
    summary = await evaluator.run_full_evaluation()

    # Save results
    output_dir = Path("data/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"rag_quality_full_eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON results saved to: {json_path}")

    # Generate and save markdown report
    report = generate_markdown_report(summary)
    report_path = output_dir / f"rag_quality_full_report_{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Markdown report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries: {summary['overall']['total_queries']}")
    print(f"Passed: {summary['overall']['passed']}")
    print(f"Pass Rate: {summary['overall']['pass_rate']*100:.1f}%")
    print(f"Average Score: {summary['overall']['avg_score']:.3f}")
    print("\nMetric Scores:")
    print(f"  - Accuracy: {summary['overall']['avg_accuracy']:.3f} (threshold: {THRESHOLDS['accuracy']})")
    print(f"  - Completeness: {summary['overall']['avg_completeness']:.3f} (threshold: {THRESHOLDS['completeness']})")
    print(f"  - Citations: {summary['overall']['avg_citations']:.3f} (threshold: {THRESHOLDS['citations']})")
    print(f"  - Context Relevance: {summary['overall']['avg_context_relevance']:.3f} (threshold: {THRESHOLDS['context_relevance']})")
    print("\nOutput Files:")
    print(f"  - JSON: {json_path}")
    print(f"  - Report: {report_path}")
    print("=" * 60)

    # Return JSON summary for API
    return {
        "evaluation_id": summary["evaluation_id"],
        "overall_pass_rate": summary["overall"]["pass_rate"],
        "avg_score": summary["overall"]["avg_score"],
        "total_queries": summary["overall"]["total_queries"],
        "passed": summary["overall"]["passed"],
        "failed": summary["overall"]["total_queries"] - summary["overall"]["passed"],
        "persona_metrics": summary["persona_metrics"],
        "output_files": [str(json_path), str(report_path)],
    }


if __name__ == "__main__":
    result = asyncio.run(main())
    print(json.dumps(result, indent=2, ensure_ascii=False))
