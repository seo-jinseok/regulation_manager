"""
Comprehensive RAG Quality Evaluator for University Regulation Manager.

This script executes diverse test queries through the RAG system and evaluates
the quality of responses across multiple dimensions:
- Intent Recognition
- Answer Accuracy
- Completeness
- Clarity
- Citation Quality
- Hallucination Detection

Executes queries simulating different user personas and query styles.
"""

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file before importing project modules
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class IntentRecognitionScore(Enum):
    """Intent recognition quality levels."""

    PERFECT = 5
    GOOD = 4
    ACCEPTABLE = 3
    PARTIAL = 2
    POOR = 1


class AnswerQualityScore(Enum):
    """Answer quality levels."""

    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    INCORRECT = 1


class UserExperienceScore(Enum):
    """User experience quality levels."""

    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FRUSTRATING = 1


@dataclass
class TestQuery:
    """A test query with metadata."""

    query: str
    persona: str
    query_style: str
    expertise: str
    expected_intent: str
    expected_keywords: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""

    query: str
    persona: str
    query_style: str
    answer_text: str
    sources: List[Dict[str, Any]]
    confidence: float
    intent_score: int
    answer_score: int
    ux_score: int
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# Diverse Test Queries for Quality Evaluation
TEST_QUERIES = [
    # ===== Freshman Student Queries =====
    TestQuery(
        query="수강 신청 언제까지야?",
        persona="신입생",
        query_style="구어체",
        expertise="초급",
        expected_intent="registration_deadline",
        expected_keywords=["수강신청", "기간"],
    ),
    TestQuery(
        query="졸업하려면 학점 몇 점 필요해?",
        persona="신입생",
        query_style="정확",
        expertise="초급",
        expected_intent="graduation_requirements",
        expected_keywords=["졸업", "학점"],
    ),
    TestQuery(
        query="장학금 신청하는 법",
        persona="신입생",
        query_style="모호",
        expertise="초급",
        expected_intent="scholarship_application",
        expected_keywords=["장학금", "신청"],
    ),
    TestQuery(
        query="아 그게 뭐냐면 학생회비 납부하는 거 어디서 해?",
        persona="신입생",
        query_style="구어체/긴",
        expertise="초급",
        expected_intent="student_council_fee",
        expected_keywords=["학생회비", "납부"],
    ),
    TestQuery(
        query="휴학하고 싶은데 어떻게 해?",
        persona="신입생",
        query_style="모호",
        expertise="초급",
        expected_intent="leave_of_absence",
        expected_keywords=["휴학", "절차"],
    ),
    # ===== Graduate Student Queries =====
    TestQuery(
        query="박사과정 연구장려금 지급 기준과 신청 서류가 궁금합니다.",
        persona="대학원생",
        query_style="정확",
        expertise="중급",
        expected_intent="research_grant",
        expected_keywords=["연구장려금", "지급기준", "신청서류"],
    ),
    TestQuery(
        query="논문 심사 위원 위촉 절차와 기간을 알고 싶습니다.",
        persona="대학원생",
        query_style="정확",
        expertise="중급",
        expected_intent="thesis_committee",
        expected_keywords=["논문심사", "위원", "위촉"],
    ),
    TestQuery(
        query="졸업 요건 중 외국어 성적 제출에 관한 규정",
        persona="대학원생",
        query_style="정확",
        expertise="전문가",
        expected_intent="graduation_requirements_language",
        expected_keywords=["졸업요건", "외국어성적"],
    ),
    # ===== Professor Queries =====
    TestQuery(
        query="교원 인사 평가 정책 중 연구 성과 평가 기준",
        persona="교수",
        query_style="정확",
        expertise="전문가",
        expected_intent="faculty_evaluation",
        expected_keywords=["인사평가", "연구성과", "평가기준"],
    ),
    TestQuery(
        query="학부생 연구원 채용 시 행정 절차",
        persona="교수",
        query_style="정확",
        expertise="중급",
        expected_intent="undergraduate_researcher",
        expected_keywords=["학부생연구원", "채용", "행정절차"],
    ),
    TestQuery(
        query="연구비 집행 시 유의해야 할 규정 사항",
        persona="교수",
        query_style="정확",
        expertise="전문가",
        expected_intent="research_expenditure",
        expected_keywords=["연구비", "집행", "규정"],
    ),
    # ===== Staff Queries =====
    TestQuery(
        query="직원 복무 규정 중 연차 사용에 관한 규정",
        persona="교직원",
        query_style="정확",
        expertise="중급",
        expected_intent="annual_leave",
        expected_keywords=["복무규정", "연차", "사용"],
    ),
    TestQuery(
        query="구매 입찰 진행 절차와 필요 서류",
        persona="교직원",
        query_style="정확",
        expertise="전문가",
        expected_intent="procurement_procedure",
        expected_keywords=["구매입찰", "절차", "서류"],
    ),
    # ===== Parent Queries =====
    TestQuery(
        query="학생 복지 카드 사용 가능한 곳과 할인 혜택",
        persona="학부모",
        query_style="구어체",
        expertise="초급",
        expected_intent="student_welfare",
        expected_keywords=["복지카드", "할인"],
    ),
    TestQuery(
        query="기숙사 비용과 납부 방법",
        persona="학부모",
        query_style="정확",
        expertise="초급",
        expected_intent="dormitory_fee",
        expected_keywords=["기숙사", "비용", "납부"],
    ),
    TestQuery(
        query="학생이 휴학하면 등록금 환불되나요?",
        persona="학부모",
        query_style="구어체",
        expertise="초급",
        expected_intent="tuition_refund",
        expected_keywords=["휴학", "등록금", "환불"],
    ),
    # ===== Ambiguous Queries =====
    TestQuery(
        query="졸업",
        persona="신입생",
        query_style="모호",
        expertise="초급",
        expected_intent="graduation_requirements",
        expected_keywords=["졸업"],
    ),
    TestQuery(
        query="등록",
        persona="신입생",
        query_style="모호",
        expertise="초급",
        expected_intent="registration",
        expected_keywords=["등록"],
    ),
    # ===== Multi-part Queries =====
    TestQuery(
        query="수강 신청 기간과 정정 기간, 그리고 취소 기간을 알려주세요.",
        persona="신입생",
        query_style="복합",
        expertise="초급",
        expected_intent="registration_periods",
        expected_keywords=["수강신청", "정정", "취소", "기간"],
    ),
    TestQuery(
        query="연구장려금 신청 자격과 절차, 그리고 제출 서류가 무엇인가요?",
        persona="대학원생",
        query_style="복합",
        expertise="중급",
        expected_intent="research_grant_details",
        expected_keywords=["연구장려금", "자격", "절차", "서류"],
    ),
    # ===== Incorrect Terminology =====
    TestQuery(
        query="학기 말 시험 일정 알려줘",
        persona="신입생",
        query_style="잘못된 용어",
        expertise="초급",
        expected_intent="final_exam_schedule",
        expected_keywords=["시험", "일정"],
    ),
    TestQuery(
        query="학교 도서관 대출 연장 방법",
        persona="신입생",
        query_style="잘못된 용어",
        expertise="초급",
        expected_intent="library_renewal",
        expected_keywords=["도서관", "대출", "연장"],
    ),
    # ===== Typos/Grammar Errors =====
    TestQuery(
        query="성적 이의 신청하는법 알려줘",
        persona="신입생",
        query_style="오타/문법오류",
        expertise="초급",
        expected_intent="grade_appeal",
        expected_keywords=["성적", "이의신청"],
    ),
    TestQuery(
        query="졸업 논문 제출 마감이 언제인가요??",
        persona="대학원생",
        query_style="오타/문법오류",
        expertise="중급",
        expected_intent="thesis_deadline",
        expected_keywords=["졸업논문", "제출", "마감"],
    ),
    TestQuery(
        query="연구비 집행시 유의사항과 영수증 제출방법",
        persona="교수",
        query_style="오타/문법오류",
        expertise="전문가",
        expected_intent="research_expenditure_receipt",
        expected_keywords=["연구비", "집행", "영수증"],
    ),
    # ===== International Student Queries =====
    TestQuery(
        query="How do I apply for leave of absence?",
        persona="유학생",
        query_style="영문",
        expertise="중급",
        expected_intent="leave_of_absence",
        expected_keywords=["휴학", "신청", "절차"],
    ),
    TestQuery(
        query="What is the tuition fee for international students?",
        persona="유학생",
        query_style="영문",
        expertise="중급",
        expected_intent="international_tuition",
        expected_keywords=["등록금", "유학생", "비용"],
    ),
    TestQuery(
        query="비자 발급을 위한 학생 확인 절차가 궁금합니다.",
        persona="유학생",
        query_style="국문혼용",
        expertise="중급",
        expected_intent="visa_confirmation",
        expected_keywords=["비자", "학생 확인", "재학증명"],
    ),
    TestQuery(
        query="기숙사 신청하는 방법 알려주세요. Can international students apply?",
        persona="유학생",
        query_style="국문혼용",
        expertise="중급",
        expected_intent="dormitory_application",
        expected_keywords=["기숙사", "신청", "유학생"],
    ),
    TestQuery(
        query="Where can I get English support for academic writing?",
        persona="유학생",
        query_style="영문",
        expertise="중급",
        expected_intent="english_support",
        expected_keywords=["영어", "학술지도", "작성"],
    ),
]


class RAGQualityEvaluator:
    """Comprehensive RAG quality evaluator."""

    def __init__(self, db_path: str = "data/chroma_db"):
        """Initialize the evaluator."""
        from src.rag.config import get_config
        from src.rag.infrastructure.chroma_store import ChromaVectorStore
        from src.rag.infrastructure.llm_adapter import LLMClientAdapter
        from src.rag.interface.query_handler import QueryHandler, QueryOptions

        self.db_path = db_path
        self.config = get_config()
        self.results: List[EvaluationResult] = []

        # Initialize components
        self.store = ChromaVectorStore(persist_directory=db_path)
        self.llm_client = LLMClientAdapter(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
        )

        # Create QueryHandler
        self.query_handler = QueryHandler(
            store=self.store,
            llm_client=self.llm_client,
            use_reranker=True,
        )

        # Default query options
        self.default_options = QueryOptions(
            top_k=5,
            use_rerank=True,
        )

    def execute_query(self, query: str, answer_mode: bool = True) -> Dict[str, Any]:
        """
        Execute a query through the RAG system.

        Returns:
            Dict with answer_text, sources, confidence
        """
        try:
            from src.rag.interface.query_handler import QueryOptions

            # Try ask mode first, fall back to search mode if LLM fails
            options = QueryOptions(
                top_k=5,
                use_rerank=True,
                force_mode="ask" if answer_mode else "search",
            )

            result = self.query_handler.process_query(
                query=query,
                options=options,
            )

            # Extract answer and sources from QueryResult
            answer_text = result.content if result.success else ""
            sources = []
            confidence = 0.0

            # Try to extract structured data with multiple fallback paths
            if result.data:
                # Path 1: FunctionGemma tool results
                if "tool_results" in result.data:
                    for tool_result in result.data.get("tool_results", []):
                        if tool_result.get("tool_name") == "search_regulations":
                            # Get result dict with better error handling
                            result_data = tool_result.get("result")
                            if result_data and isinstance(result_data, dict):
                                search_results = result_data.get("results", [])
                                # Extract sources with flexible field mapping
                                sources = []
                                for r in search_results[:5]:
                                    if isinstance(r, dict):
                                        sources.append(
                                            {
                                                "title": r.get("title")
                                                or r.get("regulation_title", ""),
                                                "text": (
                                                    r.get("text", "")
                                                    or r.get("content", "")
                                                )[:200],
                                                "rule_code": r.get("rule_code", "")
                                                or r.get("rule_code", ""),
                                                "score": r.get("score", 0.0)
                                                or r.get("similarity", 0.0),
                                            }
                                        )
                                confidence = sources[0]["score"] if sources else 0.0
                                break
                            break

                # Path 2: Direct sources
                elif "sources" in result.data:
                    sources_data = result.data["sources"]
                    if sources_data:
                        sources = [
                            {
                                "title": s.get("title", ""),
                                "text": (s.get("text", "") or s.get("content", ""))[
                                    :200
                                ],
                                "rule_code": s.get("rule_code", ""),
                                "score": s.get("score", 0.0),
                            }
                            for s in sources_data[:5]
                            if isinstance(s, dict)
                        ]
                        confidence = sources[0]["score"] if sources else 0.0

                # Path 3: Search results in different format
                elif "search_results" in result.data:
                    search_results = result.data["search_results"]
                    if isinstance(search_results, list):
                        sources = [
                            {
                                "title": r.get("title", ""),
                                "text": (r.get("text", "") or r.get("content", ""))[
                                    :200
                                ],
                                "rule_code": r.get("rule_code", ""),
                                "score": r.get("score", 0.0),
                            }
                            for r in search_results[:5]
                            if isinstance(r, dict)
                        ]
                        confidence = sources[0]["score"] if sources else 0.0

            # If no sources extracted but answer contains regulation references,
            # try to extract from content
            if not sources and result.content:
                # Look for patterns like "교원인사규정" or "제X조" in content

                # Check if there's any regulation content
                if any(
                    term in result.content
                    for term in ["규정", "조", "항", "에 따라", "관련하여"]
                ):
                    # Content exists but sources weren't extracted
                    # Mark as having content even if sources extraction failed
                    answer_text = result.content

            return {
                "answer_text": answer_text,
                "sources": sources,
                "confidence": confidence,
                "has_content": bool(answer_text),
            }

        except Exception as e:
            import traceback

            return {
                "answer_text": f"Error: {str(e)}\n{traceback.format_exc()}",
                "sources": [],
                "confidence": 0.0,
                "has_content": False,
            }

    def evaluate_query(self, test_query: TestQuery) -> EvaluationResult:
        """Evaluate a single test query."""
        result = self.execute_query(test_query.query)

        # Analyze the result
        issues = []
        strengths = []
        recommendations = []

        answer_text = result["answer_text"]
        sources = result["sources"]
        confidence = result["confidence"]

        # Intent Recognition Assessment
        intent_score = self._assess_intent_recognition(test_query, answer_text, sources)

        # Answer Quality Assessment
        answer_score = self._assess_answer_quality(test_query, answer_text, sources)

        # User Experience Assessment
        ux_score = self._assess_user_experience(
            test_query, answer_text, sources, confidence
        )

        # Identify issues
        if not answer_text or answer_text.startswith("Error:"):
            issues.append("시스템 오류로 답변 생성 실패")
        elif len(answer_text) < 50:
            issues.append("답변이 너무 짧음")
        elif not sources:
            issues.append("관련 규정을 찾지 못함")

        # Identify strengths
        if confidence > 0.8:
            strengths.append("높은 검색 신뢰도")
        if len(sources) >= 3:
            strengths.append("다양한 참고 규정 제공")
        if "제" in answer_text or "조" in answer_text:
            strengths.append("구체적인 조문 인용")

        # Generate recommendations
        if intent_score < 3:
            recommendations.append("사용자 의도 파악 개선 필요")
        if answer_score < 3:
            recommendations.append("답변 정확도 및 완결성 개선 필요")
        if not sources:
            recommendations.append("검색 품질 개선 필요")

        return EvaluationResult(
            query=test_query.query,
            persona=test_query.persona,
            query_style=test_query.query_style,
            answer_text=answer_text,
            sources=sources,
            confidence=confidence,
            intent_score=intent_score,
            answer_score=answer_score,
            ux_score=ux_score,
            issues=issues,
            strengths=strengths,
            recommendations=recommendations,
        )

    def _assess_intent_recognition(
        self, test_query: TestQuery, answer_text: str, sources: List[Dict]
    ) -> int:
        """Assess intent recognition quality (1-5)."""
        # Check if expected keywords are in answer or sources
        keyword_matches = sum(
            1
            for kw in test_query.expected_keywords
            if kw in answer_text or any(kw in s.get("text", "") for s in sources)
        )

        # Check if sources are relevant
        relevant_sources = sum(1 for s in sources if s.get("score", 0) > 0.5)

        if (
            keyword_matches >= len(test_query.expected_keywords)
            and relevant_sources >= 2
        ):
            return 5  # Perfect
        elif (
            keyword_matches >= len(test_query.expected_keywords) // 2
            and relevant_sources >= 1
        ):
            return 4  # Good
        elif keyword_matches >= 1:
            return 3  # Acceptable
        elif relevant_sources >= 1:
            return 2  # Partial
        else:
            return 1  # Poor

    def _assess_answer_quality(
        self, test_query: TestQuery, answer_text: str, sources: List[Dict]
    ) -> int:
        """Assess answer quality (1-5)."""
        if not answer_text or answer_text.startswith("Error:"):
            return 1

        # Check for hallucination indicators
        hallucination_terms = ["한국외국어대", "서울대", "02-XXXX", "일반적으로"]
        has_hallucination = any(term in answer_text for term in hallucination_terms)

        if has_hallucination:
            return 1  # Hallucination detected

        # Check completeness
        if len(answer_text) < 100:
            return 2  # Too short

        # Check for specific regulation citations
        has_citations = any(
            term in answer_text for term in ["제", "조", "항", "규정", "에 따르"]
        )

        if has_citations and len(answer_text) >= 200:
            return 5  # Excellent
        elif has_citations:
            return 4  # Good
        elif len(answer_text) >= 200:
            return 3  # Acceptable
        else:
            return 2  # Poor

    def _assess_user_experience(
        self,
        test_query: TestQuery,
        answer_text: str,
        sources: List[Dict],
        confidence: float,
    ) -> int:
        """Assess user experience quality (1-5)."""
        if not answer_text or answer_text.startswith("Error:"):
            return 1

        # Check if answer matches expertise level
        is_too_formal = test_query.expertise == "초급" and all(
            term in answer_text for term in ["귀하", "귀하의", "하여야", "하여야한다"]
        )

        # Check if answer is clear and well-structured
        has_structure = any(marker in answer_text for marker in ["1.", "-", "•", "※"])

        if confidence > 0.8 and not is_too_formal and has_structure:
            return 5  # Excellent
        elif confidence > 0.6 and not is_too_formal:
            return 4  # Good
        elif confidence > 0.4:
            return 3  # Acceptable
        elif confidence > 0.2:
            return 2  # Poor
        else:
            return 1  # Frustrating

    def run_evaluation(self, limit: Optional[int] = None) -> List[EvaluationResult]:
        """Run the full evaluation."""
        queries_to_test = TEST_QUERIES[:limit] if limit else TEST_QUERIES

        print("🔍 RAG Quality Evaluation Started")
        print(f"   Testing {len(queries_to_test)} queries...")
        print()

        for i, test_query in enumerate(queries_to_test, 1):
            print(f"[{i}/{len(queries_to_test)}] Testing: {test_query.query}")
            result = self.evaluate_query(test_query)
            self.results.append(result)
            time.sleep(0.5)  # Brief pause to avoid overwhelming

        print()
        print(f"✅ Evaluation Complete: {len(self.results)} queries tested")
        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        if not self.results:
            return "No results to report."

        # Calculate statistics
        total = len(self.results)
        avg_intent = sum(r.intent_score for r in self.results) / total
        avg_answer = sum(r.answer_score for r in self.results) / total
        avg_ux = sum(r.ux_score for r in self.results) / total
        avg_confidence = sum(r.confidence for r in self.results) / total

        # Count by persona
        by_persona = {}
        for result in self.results:
            persona = result.persona
            if persona not in by_persona:
                by_persona[persona] = {"count": 0, "intent": 0, "answer": 0, "ux": 0}
            by_persona[persona]["count"] += 1
            by_persona[persona]["intent"] += result.intent_score
            by_persona[persona]["answer"] += result.answer_score
            by_persona[persona]["ux"] += result.ux_score

        # Count issues
        issue_counts = {}
        for result in self.results:
            for issue in result.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Generate report
        report_lines = [
            "=" * 80,
            "RAG Quality Evaluation Report",
            "=" * 80,
            "",
            "## Test Summary",
            f"- Total queries tested: {total}",
            f"- Pass rate (Answer Score >= 3): {sum(1 for r in self.results if r.answer_score >= 3) / total:.1%}",
            "",
            "## Overall Scores",
            f"- Intent Recognition: {avg_intent:.2f}/5.0",
            f"- Answer Quality: {avg_answer:.2f}/5.0",
            f"- User Experience: {avg_ux:.2f}/5.0",
            f"- Average Confidence: {avg_confidence:.2%}",
            "",
            "## Results by Persona",
            "",
        ]

        for persona, stats in sorted(by_persona.items()):
            count = stats["count"]
            report_lines.extend(
                [
                    f"### {persona}",
                    f"  Queries: {count}",
                    f"  Intent: {stats['intent'] / count:.2f}/5.0",
                    f"  Answer: {stats['answer'] / count:.2f}/5.0",
                    f"  UX: {stats['ux'] / count:.2f}/5.0",
                    "",
                ]
            )

        # Issues section
        report_lines.extend(
            [
                "## Issues Found",
                "",
            ]
        )

        for issue, count in sorted(
            issue_counts.items(), key=lambda x: x[1], reverse=True
        ):
            report_lines.append(f"{count}x: {issue}")

        if issue_counts:
            report_lines.append("")

        # Recommendations
        all_recommendations = {}
        for result in self.results:
            for rec in result.recommendations:
                all_recommendations[rec] = all_recommendations.get(rec, 0) + 1

        report_lines.extend(
            [
                "## Improvement Recommendations",
                "",
            ]
        )

        for rec, count in sorted(
            all_recommendations.items(), key=lambda x: x[1], reverse=True
        ):
            report_lines.append(f"Priority {count}: {rec}")

        # Detailed results
        report_lines.extend(
            [
                "",
                "## Detailed Results",
                "",
            ]
        )

        for i, result in enumerate(self.results, 1):
            report_lines.extend(
                [
                    f"### {i}. {result.query} ({result.persona}, {result.query_style})",
                    f"**Scores:** Intent={result.intent_score}/5, Answer={result.answer_score}/5, UX={result.ux_score}/5",
                    f"**Confidence:** {result.confidence:.2%}",
                ]
            )

            if result.issues:
                report_lines.append(f"**Issues:** {', '.join(result.issues)}")
            if result.strengths:
                report_lines.append(f"**Strengths:** {', '.join(result.strengths)}")
            if result.recommendations:
                report_lines.append(
                    f"**Recommendations:** {', '.join(result.recommendations)}"
                )

            report_lines.extend(
                [
                    f"**Answer Preview:** {result.answer_text[:200]}...",
                    f"**Sources:** {len(result.sources)} found",
                    "",
                ]
            )

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Quality Evaluator")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of queries to test"
    )
    parser.add_argument("--db-path", default="data/chroma_db", help="Path to ChromaDB")
    parser.add_argument("--output", default=None, help="Output report to file")

    args = parser.parse_args()

    evaluator = RAGQualityEvaluator(db_path=args.db_path)
    evaluator.run_evaluation(limit=args.limit)

    report = evaluator.generate_report()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
