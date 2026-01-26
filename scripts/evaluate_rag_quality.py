"""
RAG 시스템 전체 품질 평가 스크립트

평가 범위:
1. 애매한 질문 처리 (20-30개 쿼리)
2. 다양한 페르소나 시뮬레이션 (10개 페르소나 × 3개 쿼리)
3. 다중 턴 대화 평가 (10개 시나리오)

결과: 한국어 보고서로 출력
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.automation.domain.entities import PersonaType
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType
from src.rag.infrastructure.reranker import BGEReranker
from src.rag.interface.query_handler import QueryContext, QueryHandler, QueryOptions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """단일 테스트 쿼리"""

    query_id: str
    query: str
    category: str  # ambiguous, persona_based, multi_turn
    persona_type: Optional[PersonaType] = None
    difficulty: str = "medium"  # easy, medium, hard
    expected_intent: Optional[str] = None
    context_hints: List[str] = field(default_factory=list)


@dataclass
class QueryEvaluationResult:
    """쿼리 평가 결과"""

    query_id: str
    query: str
    category: str

    # Intent Recognition
    intent_recognition_score: float  # 1-5
    intent_recognition_details: str

    # Answer Quality
    answer_quality_score: float  # 1-5
    answer_quality_details: str

    # User Experience
    user_experience_score: float  # 1-5
    user_experience_details: str

    # System Metrics
    sources_count: int
    confidence: float
    execution_time_ms: int
    query_rewrite_used: bool
    query_type_detected: Optional[str] = None

    # Additional Info
    answer_text: str = ""
    sources: List[str] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        """총점 (15점 만점)"""
        return (
            self.intent_recognition_score
            + self.answer_quality_score
            + self.user_experience_score
        )

    @property
    def passed(self) -> bool:
        """합격 여부 (총점 10점 이상)"""
        return self.total_score >= 10.0


@dataclass
class MultiTurnResult:
    """다중 턴 대화 결과"""

    scenario_id: str
    persona_type: PersonaType
    initial_query: str
    turns: List[Dict[str, Any]]

    # 평가 메트릭
    context_preservation_rate: float
    avg_confidence: float
    total_turns: int

    # 각 턴의 평가
    turn_evaluations: List[QueryEvaluationResult] = field(default_factory=list)


class RAGQualityEvaluator:
    """RAG 시스템 품질 평가기"""

    def __init__(self, db_path: str = "data/chroma_db"):
        """초기화"""
        self.db_path = db_path
        self.query_handler: Optional[QueryHandler] = None
        self.query_analyzer: Optional[QueryAnalyzer] = None
        self._initialize_system()

    def _initialize_system(self):
        """RAG 시스템 초기화"""
        logger.info("RAG 시스템 초기화 중...")

        # ChromaDB 초기화
        store = ChromaVectorStore(persist_directory=self.db_path)
        logger.info(f"ChromaDB 문서 수: {store.count()}")

        # LLM 클라이언트 초기화
        llm_client = LLMClientAdapter(
            provider=os.getenv("LLM_PROVIDER", "openrouter"),
            model=os.getenv("LLM_MODEL", "z-ai/glm-4.7-flash"),
            base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        )

        # Reranker 초기화
        try:
            reranker = BGEReranker()
            use_reranker = True
            logger.info("BGE Reranker 활성화")
        except Exception as e:
            logger.warning(f"Reranker 초기화 실패: {e}")
            reranker = None
            use_reranker = False

        # Query Analyzer 초기화
        self.query_analyzer = QueryAnalyzer(llm_client=llm_client)

        # Query Handler 초기화
        self.query_handler = QueryHandler(
            store=store, llm_client=llm_client, use_reranker=use_reranker
        )

        logger.info("RAG 시스템 초기화 완료")

    def evaluate_query(self, test_query: TestQuery) -> QueryEvaluationResult:
        """단일 쿼리 평가"""
        start_time = time.time()

        # 쿼리 실행
        options = QueryOptions(top_k=5, use_rerank=True, show_debug=False)

        context = QueryContext()

        result = self.query_handler.process_query(
            query=test_query.query, context=context, options=options
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # 결과 분석
        eval_result = self._analyze_result(
            test_query=test_query,
            query_result=result,
            execution_time_ms=execution_time_ms,
        )

        return eval_result

    def _analyze_result(
        self, test_query: TestQuery, query_result, execution_time_ms: int
    ) -> QueryEvaluationResult:
        """결과 분석 및 점수 매기기"""

        # 1. 의도 인식 평가 (1-5점)
        intent_score, intent_details = self._evaluate_intent_recognition(
            test_query, query_result
        )

        # 2. 답변 품질 평가 (1-5점)
        quality_score, quality_details = self._evaluate_answer_quality(
            test_query, query_result
        )

        # 3. 사용자 경험 평가 (1-5점)
        ux_score, ux_details = self._evaluate_user_experience(test_query, query_result)

        # 시스템 메트릭 추출
        sources_count = len(query_result.data.get("results", []))
        confidence = query_result.data.get("confidence", 0.0)

        query_rewrite = hasattr(self.query_handler, "_last_query_rewrite")
        query_type = (
            self.query_analyzer.analyze(test_query.query)
            if self.query_analyzer
            else None
        )

        return QueryEvaluationResult(
            query_id=test_query.query_id,
            query=test_query.query,
            category=test_query.category,
            intent_recognition_score=intent_score,
            intent_recognition_details=intent_details,
            answer_quality_score=quality_score,
            answer_quality_details=quality_details,
            user_experience_score=ux_score,
            user_experience_details=ux_details,
            sources_count=sources_count,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            query_rewrite_used=query_rewrite,
            query_type_detected=query_type.value if query_type else None,
            answer_text=query_result.content,
            sources=query_result.data.get("results", []),
        )

    def _evaluate_intent_recognition(
        self, test_query: TestQuery, query_result
    ) -> Tuple[float, str]:
        """의도 인식 평가"""
        score = 3.0  # 기본점
        details = []

        # 쿼리 타입 분석
        if self.query_analyzer:
            detected_type = self.query_analyzer.analyze(test_query.query)
            details.append(f"감지된 쿼리 타입: {detected_type.value}")

            # 애매한 질문에 대한 타입 감지 보너스
            if test_query.category == "ambiguous":
                if detected_type in [QueryType.NATURAL_QUESTION, QueryType.INTENT]:
                    score += 1.0
                    details.append("애매한 질문을 올바르게 자연어 질문/의도로 감지")
                else:
                    score -= 0.5
                    details.append("애매한 질문 감지 미흡")

        # 결과 타입 확인
        if query_result.type.value in ["ask", "search"]:
            score += 0.5
            details.append(f"적절한 결과 타입: {query_result.type.value}")

        # 소스 찾음
        if query_result.data.get("results"):
            score += 0.5
            details.append(
                f"관련 규정 찾음: {len(query_result.data.get('results', []))}개"
            )
        else:
            score -= 1.0
            details.append("관련 규정을 찾지 못함")

        return min(5.0, max(1.0, score)), "; ".join(details)

    def _evaluate_answer_quality(
        self, test_query: TestQuery, query_result
    ) -> Tuple[float, str]:
        """답변 품질 평가"""
        score = 3.0  # 기본점
        details = []

        content = query_result.content

        # 길이 확인
        if len(content) > 100:
            score += 0.5
            details.append("충분한 답변 길이")
        elif len(content) < 50:
            score -= 1.0
            details.append("답변이 너무 짧음")

        # 출처 인용 확인
        has_citation = any(marker in content for marker in ["제", "조", "규정", "학칙"])
        if has_citation:
            score += 0.5
            details.append("출처 인용 포함")
        else:
            score -= 0.5
            details.append("출처 인용 부족")

        # 구체적 정보 포함 확인
        concrete_indicators = ["기간", "신청", "방법", "조건", "제출", "서류"]
        if any(indicator in content for indicator in concrete_indicators):
            score += 0.5
            details.append("구체적 정보 포함")

        # 일반화 답변 감지
        if "대학마다 다를 수 있습니다" in content or "알 수 없습니다" in content:
            score -= 2.0
            details.append("일반화 답변 감지 (자동 감기)")

        # 오류 메시지 확인
        if query_result.type.value == "error":
            score = 1.0
            details.append("시스템 오류 발생")

        return min(5.0, max(1.0, score)), "; ".join(details)

    def _evaluate_user_experience(
        self, test_query: TestQuery, query_result
    ) -> Tuple[float, str]:
        """사용자 경험 평가"""
        score = 3.0  # 기본점
        details = []

        # 응답 시간
        execution_time = getattr(query_result, "execution_time_ms", 0) or 0
        if execution_time < 2000:
            score += 0.5
            details.append(f"빠른 응답 ({execution_time}ms)")
        elif execution_time > 5000:
            score -= 0.5
            details.append(f"느린 응답 ({execution_time}ms)")

        # 명확성
        content = query_result.content
        if "⚠️" in content or "❌" in content:
            score -= 0.5
            details.append("오류/경고 메시지 포함")

        # 제안 표시
        if query_result.suggestions:
            score += 0.5
            details.append(f"후속 질문 제안 제공 ({len(query_result.suggestions)}개)")

        # 오류 응답
        if query_result.type.value == "error":
            score = 2.0
            details.append("오류 응답")

        return min(5.0, max(1.0, score)), "; ".join(details)

    def generate_test_queries(self) -> List[TestQuery]:
        """테스트 쿼리 세트 생성"""
        queries = []

        # 1. 애매한 질문 (20개)
        ambiguous_queries = [
            # 모호한 표현
            TestQuery(
                "amb-001",
                "규정 바뀌었어?",
                "ambiguous",
                difficulty="medium",
                expected_intent="inquiry",
            ),
            TestQuery(
                "amb-002",
                "학교 안 가고 싶어",
                "ambiguous",
                difficulty="hard",
                expected_intent="absence",
            ),
            TestQuery(
                "amb-003",
                "돈 좀 주세요",
                "ambiguous",
                difficulty="hard",
                expected_intent="financial_aid",
            ),
            TestQuery(
                "amb-004",
                "점수 너무 안 좋아",
                "ambiguous",
                difficulty="medium",
                expected_intent="grade_complaint",
            ),
            TestQuery(
                "amb-005",
                "교수님 싫어",
                "ambiguous",
                difficulty="hard",
                expected_intent="complaint",
            ),
            # 구어체/비문
            TestQuery(
                "amb-006",
                "장학금 뭐야?",
                "ambiguous",
                difficulty="medium",
                expected_intent="scholarship",
            ),
            TestQuery(
                "amb-007",
                "휴학 어떻게 함?",
                "ambiguous",
                difficulty="medium",
                expected_intent="leave_of_absence",
            ),
            TestQuery(
                "amb-008",
                "성적 이의 어떻게?",
                "ambiguous",
                difficulty="medium",
                expected_intent="grade_appeal",
            ),
            TestQuery(
                "amb-009",
                "졸업 가능?",
                "ambiguous",
                difficulty="medium",
                expected_intent="graduation",
            ),
            TestQuery(
                "amb-010",
                "F 받음",
                "ambiguous",
                difficulty="easy",
                expected_intent="retake",
            ),
            # 오타/비문
            TestQuery(
                "amb-011",
                "장학금 밫고 시퍼",
                "ambiguous",
                difficulty="hard",
                expected_intent="scholarship",
            ),
            TestQuery(
                "amb-012",
                "학교 그만두고 싶음",
                "ambiguous",
                difficulty="medium",
                expected_intent="withdrawal",
            ),
            TestQuery(
                "amb-013",
                "성적 공정 안됨",
                "ambiguous",
                difficulty="hard",
                expected_intent="grade_appeal",
            ),
            # 복합 질문
            TestQuery(
                "amb-014",
                "전공 바꾸고 싶은데 성적은 어카징",
                "ambiguous",
                difficulty="hard",
                expected_intent="complex",
            ),
            TestQuery(
                "amb-015",
                "장학금 받으면서 휴학 가능?",
                "ambiguous",
                difficulty="hard",
                expected_intent="complex",
            ),
            # 감정적 표현
            TestQuery(
                "amb-016",
                "학교 너무 힘들어",
                "ambiguous",
                difficulty="hard",
                expected_intent="counseling",
            ),
            TestQuery(
                "amb-017",
                "성적이 부족해서 그만두고 싶어요",
                "ambiguous",
                difficulty="hard",
                expected_intent="withdrawal",
            ),
            TestQuery(
                "amb-018",
                "교수님이 과제 너무 많이 주셔",
                "ambiguous",
                difficulty="medium",
                expected_intent="complaint",
            ),
            # 맥락 의존적
            TestQuery(
                "amb-019", "그거 신청 기간 언제야?", "ambiguous", difficulty="hard"
            ),
            TestQuery("amb-020", "어디서 해?", "ambiguous", difficulty="hard"),
        ]
        queries.extend(ambiguous_queries)

        # 2. 페르소나별 쿼리 (10개 페르소나 × 3개 = 30개)
        persona_queries = [
            # 신입생
            TestQuery(
                "fr-001",
                "학교 처음 왔는데 뭐부터 해야 되나요?",
                "persona_based",
                PersonaType.FRESHMAN,
                "easy",
                context_hints=["new_student"],
            ),
            TestQuery(
                "fr-002",
                "기숙사 어떻게 신청해요?",
                "persona_based",
                PersonaType.FRESHMAN,
                "medium",
                context_hints=["dormitory"],
            ),
            TestQuery(
                "fr-003",
                "장학금 뭐 있나요?",
                "persona_based",
                PersonaType.FRESHMAN,
                "medium",
                context_hints=["scholarship"],
            ),
            # 재학생 (3학년)
            TestQuery(
                "jr-001",
                "졸업 요건이 어떻게 되나요?",
                "persona_based",
                PersonaType.JUNIOR,
                "medium",
                context_hints=["graduation"],
            ),
            TestQuery(
                "jr-002",
                "전공 바꾸고 싶어요",
                "persona_based",
                PersonaType.JUNIOR,
                "medium",
                context_hints=["change_major"],
            ),
            TestQuery(
                "jr-003",
                "교환학생 어떻게 하나요?",
                "persona_based",
                PersonaType.JUNIOR,
                "hard",
                context_hints=["exchange_student"],
            ),
            # 대학원생
            TestQuery(
                "gr-001",
                "논문 심사 기준이 어떻게 됩니까?",
                "persona_based",
                PersonaType.GRADUATE,
                "hard",
                context_hints=["thesis"],
            ),
            TestQuery(
                "gr-002",
                "연구비 지원 받을 수 있나요?",
                "persona_based",
                PersonaType.GRADUATE,
                "medium",
                context_hints=["research_funding"],
            ),
            TestQuery(
                "gr-003",
                "박사 과정 지원 자격이 무엇입니까?",
                "persona_based",
                PersonaType.GRADUATE,
                "medium",
                context_hints=["phd_application"],
            ),
            # 신임 교수
            TestQuery(
                "np-001",
                "교원 연구년 신청 방법을 알려주세요",
                "persona_based",
                PersonaType.NEW_PROFESSOR,
                "medium",
                context_hints=["sabbatical"],
            ),
            TestQuery(
                "np-002",
                "책임 시수는 어떻게 되나요?",
                "persona_based",
                PersonaType.NEW_PROFESSOR,
                "medium",
                context_hints=["teaching_load"],
            ),
            TestQuery(
                "np-003",
                "업적 평가 기준이 궁금합니다",
                "persona_based",
                PersonaType.NEW_PROFESSOR,
                "hard",
                context_hints=["performance_eval"],
            ),
            # 정교수
            TestQuery(
                "pr-001",
                "정년 보장 규정을 확인하고 싶습니다",
                "persona_based",
                PersonaType.PROFESSOR,
                "hard",
                context_hints=["tenure"],
            ),
            TestQuery(
                "pr-002",
                "교원 휴직 절차가 어떻게 됩니까?",
                "persona_based",
                PersonaType.PROFESSOR,
                "medium",
                context_hints=["faculty_leave"],
            ),
            TestQuery(
                "pr-003",
                "학회 지원经费 지원 가능한가요?",
                "persona_based",
                PersonaType.PROFESSOR,
                "medium",
                context_hints=["conference_funding"],
            ),
            # 신입 직원
            TestQuery(
                "ns-001",
                "연차 휴가 사용 방법을 알려주세요",
                "persona_based",
                PersonaType.NEW_STAFF,
                "easy",
                context_hints=["annual_leave"],
            ),
            TestQuery(
                "ns-002",
                "복지 혜택이 어떤 게 있나요?",
                "persona_based",
                PersonaType.NEW_STAFF,
                "medium",
                context_hints=["benefits"],
            ),
            TestQuery(
                "ns-003",
                "퇴직금 계산 방법이 궁금합니다",
                "persona_based",
                PersonaType.NEW_STAFF,
                "medium",
                context_hints=["severance"],
            ),
            # 과장급 직원
            TestQuery(
                "sm-001",
                "부서 예산 집행 절차를 알려주세요",
                "persona_based",
                PersonaType.STAFF_MANAGER,
                "hard",
                context_hints=["budget"],
            ),
            TestQuery(
                "sm-002",
                "직원 승진 기준이 어떻게 됩니까?",
                "persona_based",
                PersonaType.STAFF_MANAGER,
                "medium",
                context_hints=["promotion"],
            ),
            TestQuery(
                "sm-003",
                "파견 규정을 확인하고 싶습니다",
                "persona_based",
                PersonaType.STAFF_MANAGER,
                "hard",
                context_hints=["secondment"],
            ),
            # 학부모
            TestQuery(
                "pa-001",
                "등록금 납부 기간이 언제인가요?",
                "persona_based",
                PersonaType.PARENT,
                "easy",
                context_hints=["tuition"],
            ),
            TestQuery(
                "pa-002",
                "자녀 성적 확인 방법을 알려주세요",
                "persona_based",
                PersonaType.PARENT,
                "medium",
                context_hints=["grade_check"],
            ),
            TestQuery(
                "pa-003",
                "학부모 상담 어떻게 하나요?",
                "persona_based",
                PersonaType.PARENT,
                "medium",
                context_hints=["parent_conference"],
            ),
            # 어려운 상황의 학생
            TestQuery(
                "ds-001",
                "학교 다니기 너무 힘들어요",
                "persona_based",
                PersonaType.DISTRESSED_STUDENT,
                "hard",
                context_hints=["counseling"],
            ),
            TestQuery(
                "ds-002",
                "도와주세요...",
                "persona_based",
                PersonaType.DISTRESSED_STUDENT,
                "hard",
                context_hints=["help_seeking"],
            ),
            TestQuery(
                "ds-003",
                "상담하고 싶은데 어디로 가야 하나요?",
                "persona_based",
                PersonaType.DISTRESSED_STUDENT,
                "medium",
                context_hints=["counseling_center"],
            ),
            # 불만있는 구성원
            TestQuery(
                "dm-001",
                "성적 처리가 부당했습니다",
                "persona_based",
                PersonaType.DISSATISFIED_MEMBER,
                "hard",
                context_hints=["grade_complaint"],
            ),
            TestQuery(
                "dm-002",
                "신고하고 싶습니다",
                "persona_based",
                PersonaType.DISSATISFIED_MEMBER,
                "hard",
                context_hints=["report"],
            ),
            TestQuery(
                "dm-003",
                "항의하고 싶은데 어떻게 하나요?",
                "persona_based",
                PersonaType.DISSATISFIED_MEMBER,
                "medium",
                context_hints=["protest"],
            ),
        ]
        queries.extend(persona_queries)

        return queries

    def run_evaluation(self, output_dir: str = "test_results") -> Dict[str, Any]:
        """전체 평가 실행"""
        logger.info("=" * 60)
        logger.info("RAG 시스템 전체 품질 평가 시작")
        logger.info("=" * 60)

        # 테스트 쿼리 생성
        test_queries = self.generate_test_queries()
        logger.info(f"생성된 테스트 쿼리: {len(test_queries)}개")

        # 평가 실행
        results = []
        passed = 0
        failed = 0

        for idx, test_query in enumerate(test_queries, 1):
            logger.info(
                f"[{idx}/{len(test_queries)}] 평가 중: {test_query.query[:50]}..."
            )

            try:
                result = self.evaluate_query(test_query)
                results.append(result)

                if result.passed:
                    passed += 1
                    logger.info(f"  ✓ 합격 (총점: {result.total_score:.1f}/15)")
                else:
                    failed += 1
                    logger.warning(f"  ✗ 불합격 (총점: {result.total_score:.1f}/15)")

            except Exception as e:
                logger.error(f"  ✗ 평가 실패: {e}")
                failed += 1

        # 결과 집계
        summary = self._generate_summary(results, passed, failed)

        # 결과 저장
        self._save_results(summary, results, output_dir)

        # 보고서 생성
        self._generate_report(summary, results, output_dir)

        return summary

    def _generate_summary(
        self, results: List[QueryEvaluationResult], passed: int, failed: int
    ) -> Dict[str, Any]:
        """결과 집계"""
        total = len(results)

        # 평균 점수
        avg_intent = (
            sum(r.intent_recognition_score for r in results) / total if total else 0
        )
        avg_quality = (
            sum(r.answer_quality_score for r in results) / total if total else 0
        )
        avg_ux = sum(r.user_experience_score for r in results) / total if total else 0
        avg_total = sum(r.total_score for r in results) / total if total else 0

        # 카테고리별 분석
        by_category: Dict[str, List[QueryEvaluationResult]] = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)

        category_stats = {}
        for category, cat_results in by_category.items():
            cat_total = len(cat_results)
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_avg_score = (
                sum(r.total_score for r in cat_results) / cat_total if cat_total else 0
            )

            category_stats[category] = {
                "total": cat_total,
                "passed": cat_passed,
                "failed": cat_total - cat_passed,
                "pass_rate": cat_passed / cat_total if cat_total else 0,
                "avg_score": cat_avg_score,
            }

        # 페르소나별 분석
        by_persona: Dict[str, List[QueryEvaluationResult]] = {}
        for r in results:
            if r.category == "persona_based" and hasattr(r, "query"):
                # 결과에서 페르소나 정보 추출 시도
                pass  # 추후 구현

        return {
            "evaluated_at": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total else 0,
            "average_scores": {
                "intent_recognition": avg_intent,
                "answer_quality": avg_quality,
                "user_experience": avg_ux,
                "total": avg_total,
            },
            "category_statistics": category_stats,
            "score_distribution": {
                "excellent": sum(1 for r in results if r.total_score >= 13),
                "good": sum(1 for r in results if 11 <= r.total_score < 13),
                "fair": sum(1 for r in results if 9 <= r.total_score < 11),
                "poor": sum(1 for r in results if r.total_score < 9),
            },
        }

    def _save_results(
        self,
        summary: Dict[str, Any],
        results: List[QueryEvaluationResult],
        output_dir: str,
    ):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # JSON 저장
        output_path = (
            Path(output_dir)
            / f"rag_quality_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        output_data = {
            "summary": summary,
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "category": r.category,
                    "scores": {
                        "intent_recognition": r.intent_recognition_score,
                        "answer_quality": r.answer_quality_score,
                        "user_experience": r.user_experience_score,
                        "total": r.total_score,
                    },
                    "details": {
                        "intent": r.intent_recognition_details,
                        "quality": r.answer_quality_details,
                        "ux": r.user_experience_details,
                    },
                    "metrics": {
                        "sources_count": r.sources_count,
                        "confidence": r.confidence,
                        "execution_time_ms": r.execution_time_ms,
                    },
                    "passed": r.passed,
                }
                for r in results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"결과 저장 완료: {output_path}")

    def _generate_report(
        self,
        summary: Dict[str, Any],
        results: List[QueryEvaluationResult],
        output_dir: str,
    ):
        """한국어 보고서 생성"""
        report_path = (
            Path(output_dir)
            / f"rag_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

        # 보고서 생성
        report_lines = []
        report_lines.append("# RAG 시스템 품질 평가 보고서")
        report_lines.append(f"\n**평가 시각**: {summary['evaluated_at']}")
        report_lines.append("\n## 1. 평가 개요")
        report_lines.append(f"\n- **전체 테스트**: {summary['total_tests']}개")
        report_lines.append(f"- **합격**: {summary['passed']}개")
        report_lines.append(f"- **불합격**: {summary['failed']}개")
        report_lines.append(f"- **합격률**: {summary['pass_rate']:.1%}")
        report_lines.append("\n## 2. 평균 점수")
        report_lines.append("\n| 평가 항목 | 평균 점수 | 만점 |")
        report_lines.append("|----------|----------|------|")
        report_lines.append(
            f"| 의도 인식 | {summary['average_scores']['intent_recognition']:.2f} | 5.0 |"
        )
        report_lines.append(
            f"| 답변 품질 | {summary['average_scores']['answer_quality']:.2f} | 5.0 |"
        )
        report_lines.append(
            f"| 사용자 경험 | {summary['average_scores']['user_experience']:.2f} | 5.0 |"
        )
        report_lines.append(
            f"| **총점** | **{summary['average_scores']['total']:.2f}** | **15.0** |"
        )

        # 카테고리별 통계
        report_lines.append("\n## 3. 카테고리별 분석")
        for category, stats in summary["category_statistics"].items():
            report_lines.append(f"\n### {category.upper()}")
            report_lines.append(f"- 전체: {stats['total']}개")
            report_lines.append(
                f"- 합격: {stats['passed']}개 ({stats['pass_rate']:.1%})"
            )
            report_lines.append(f"- 평균 점수: {stats['avg_score']:.2f}/15")

        # 점수 분포
        report_lines.append("\n## 4. 점수 분포")
        dist = summary["score_distribution"]
        report_lines.append(f"- 우수 (13-15점): {dist['excellent']}개")
        report_lines.append(f"- 양호 (11-12점): {dist['good']}개")
        report_lines.append(f"- 보통 (9-10점): {dist['fair']}개")
        report_lines.append(f"- 부족 (0-8점): {dist['poor']}개")

        # 실패 케이스 분석
        report_lines.append("\n## 5. 실패 케이스 분석")
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            report_lines.append(f"\n불합격 케이스 {len(failed_results)}개:")
            for r in failed_results[:10]:  # 최대 10개
                report_lines.append(f"\n#### {r.query_id}: {r.query[:50]}...")
                report_lines.append(f"- 총점: {r.total_score:.1f}/15")
                report_lines.append(
                    f"- 의도 인식: {r.intent_recognition_score:.1f} - {r.intent_recognition_details}"
                )
                report_lines.append(
                    f"- 답변 품질: {r.answer_quality_score:.1f} - {r.answer_quality_details}"
                )
                report_lines.append(
                    f"- 사용자 경험: {r.user_experience_score:.1f} - {r.user_experience_details}"
                )
        else:
            report_lines.append("\n불합격 케이스 없음!")

        # 우수 케이스
        report_lines.append("\n## 6. 우수 케이스")
        excellent_results = sorted(results, key=lambda r: r.total_score, reverse=True)[
            :5
        ]
        for r in excellent_results:
            report_lines.append(f"\n#### {r.query_id}: {r.query[:50]}...")
            report_lines.append(f"- 총점: {r.total_score:.1f}/15")

        # 개선 제안
        report_lines.append("\n## 7. 개선 제안")

        if summary["average_scores"]["intent_recognition"] < 4.0:
            report_lines.append("\n1. **의도 인식 개선**:")
            report_lines.append("   - 애매한 질문 처리를 위한 추가 패턴 학습")
            report_lines.append("   - LLM 기반 쿼리 재작성 기능 강화")

        if summary["average_scores"]["answer_quality"] < 4.0:
            report_lines.append("\n2. **답변 품질 개선**:")
            report_lines.append("   - 출처 인용 강화")
            report_lines.append("   - 구체적 정보 포함 개선")

        if summary["average_scores"]["user_experience"] < 4.0:
            report_lines.append("\n3. **사용자 경험 개선**:")
            report_lines.append("   - 응답 시간 최적화")
            report_lines.append("   - 후속 질문 제안 기능 강화")

        report_text = "\n".join(report_lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"보고서 저장 완료: {report_path}")


def main():
    """메인 함수"""
    # .env 파일 로드
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # 평가 실행
    evaluator = RAGQualityEvaluator()
    summary = evaluator.run_evaluation()

    # 요약 출력
    print("\n" + "=" * 60)
    print("평가 완료!")
    print("=" * 60)
    print(f"합격률: {summary['pass_rate']:.1%}")
    print(f"평균 점수: {summary['average_scores']['total']:.2f}/15")
    print("=" * 60)


if __name__ == "__main__":
    main()
