"""
RAG 시스템 빠른 품질 평가 스크립트

50개 테스트 쿼리에 대한 신속 평가
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.infrastructure.query_analyzer import QueryAnalyzer
from src.rag.infrastructure.reranker import BGEReranker
from src.rag.interface.query_handler import QueryContext, QueryHandler, QueryOptions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 테스트 쿼리 세트
TEST_QUERIES = [
    # 애매한 질문 (20개)
    ("amb-001", "규정 바뀌었어?", "ambiguous"),
    ("amb-002", "학교 안 가고 싶어", "ambiguous"),
    ("amb-003", "돈 좀 주세요", "ambiguous"),
    ("amb-004", "점수 너무 안 좋아", "ambiguous"),
    ("amb-005", "교수님 싫어", "ambiguous"),
    ("amb-006", "장학금 뭐야?", "ambiguous"),
    ("amb-007", "휴학 어떻게 함?", "ambiguous"),
    ("amb-008", "성적 이의 어떻게?", "ambiguous"),
    ("amb-009", "졸업 가능?", "ambiguous"),
    ("amb-010", "F 받음", "ambiguous"),
    ("amb-011", "장학금 밫고 시퍼", "ambiguous"),
    ("amb-012", "학교 그만두고 싶음", "ambiguous"),
    ("amb-013", "성적 공정 안됨", "ambiguous"),
    ("amb-014", "전공 바꾸고 싶은데 성적은 어카징", "ambiguous"),
    ("amb-015", "장학금 받으면서 휴학 가능?", "ambiguous"),
    ("amb-016", "학교 너무 힘들어", "ambiguous"),
    ("amb-017", "성적이 부족해서 그만두고 싶어요", "ambiguous"),
    ("amb-018", "교수님이 과제 너무 많이 주셔", "ambiguous"),
    ("amb-019", "그거 신청 기간 언제야?", "ambiguous"),
    ("amb-020", "어디서 해?", "ambiguous"),
    # 페르소나별 쿼리 (30개)
    ("fr-001", "학교 처음 왔는데 뭐부터 해야 되나요?", "persona_based"),
    ("fr-002", "기숙사 어떻게 신청해요?", "persona_based"),
    ("fr-003", "장학금 뭐 있나요?", "persona_based"),
    ("jr-001", "졸업 요건이 어떻게 되나요?", "persona_based"),
    ("jr-002", "전공 바꾸고 싶어요", "persona_based"),
    ("jr-003", "교환학생 어떻게 하나요?", "persona_based"),
    ("gr-001", "논문 심사 기준이 어떻게 됩니까?", "persona_based"),
    ("gr-002", "연구비 지원 받을 수 있나요?", "persona_based"),
    ("gr-003", "박사 과정 지원 자격이 무엇입니까?", "persona_based"),
    ("np-001", "교원 연구년 신청 방법을 알려주세요", "persona_based"),
    ("np-002", "책임 시수는 어떻게 되나요?", "persona_based"),
    ("np-003", "업적 평가 기준이 궁금합니다", "persona_based"),
    ("pr-001", "정년 보장 규정을 확인하고 싶습니다", "persona_based"),
    ("pr-002", "교원 휴직 절차가 어떻게 됩니까?", "persona_based"),
    ("pr-003", "학회 지원经费 지원 가능한가요?", "persona_based"),
    ("ns-001", "연차 휴가 사용 방법을 알려주세요", "persona_based"),
    ("ns-002", "복지 혜택이 어떤 게 있나요?", "persona_based"),
    ("ns-003", "퇴직금 계산 방법이 궁금합니다", "persona_based"),
    ("sm-001", "부서 예산 집행 절차를 알려주세요", "persona_based"),
    ("sm-002", "직원 승진 기준이 어떻게 됩니까?", "persona_based"),
    ("sm-003", "파견 규정을 확인하고 싶습니다", "persona_based"),
    ("pa-001", "등록금 납부 기간이 언제인가요?", "persona_based"),
    ("pa-002", "자녀 성적 확인 방법을 알려주세요", "persona_based"),
    ("pa-003", "학부모 상담 어떻게 하나요?", "persona_based"),
    ("ds-001", "학교 다니기 너무 힘들어요", "persona_based"),
    ("ds-002", "도와주세요...", "persona_based"),
    ("ds-003", "상담하고 싶은데 어디로 가야 하나요?", "persona_based"),
    ("dm-001", "성적 처리가 부당했습니다", "persona_based"),
    ("dm-002", "신고하고 싶습니다", "persona_based"),
    ("dm-003", "항의하고 싶은데 어떻게 하나요?", "persona_based"),
]


def main():
    """메인 함수"""
    # .env 파일 로드
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    logger.info("=" * 70)
    logger.info("RAG 시스템 빠른 품질 평가 시작")
    logger.info("=" * 70)

    # RAG 시스템 초기화
    logger.info("RAG 시스템 초기화 중...")
    store = ChromaVectorStore(persist_directory="data/chroma_db")
    logger.info(f"ChromaDB 문서 수: {store.count()}")

    llm_client = LLMClientAdapter(
        provider=os.getenv("LLM_PROVIDER", "openrouter"),
        model=os.getenv("LLM_MODEL", "z-ai/glm-4.7-flash"),
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    try:
        reranker = BGEReranker()
        use_reranker = True
        logger.info("BGE Reranker 활성화")
    except Exception as e:
        logger.warning(f"Reranker 초기화 실패: {e}")
        use_reranker = False

    query_analyzer = QueryAnalyzer(llm_client=llm_client)

    query_handler = QueryHandler(
        store=store, llm_client=llm_client, use_reranker=use_reranker
    )

    logger.info("RAG 시스템 초기화 완료")

    # 평가 실행
    results = []
    passed = 0
    failed = 0

    for idx, (query_id, query, category) in enumerate(TEST_QUERIES, 1):
        logger.info(f"[{idx}/{len(TEST_QUERIES)}] {query[:40]}...")

        start_time = time.time()
        try:
            options = QueryOptions(top_k=5, use_rerank=use_reranker, show_debug=False)
            context = QueryContext()

            result = query_handler.process_query(
                query=query, context=context, options=options
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # 결과 분석
            sources_count = len(result.data.get("results", []))
            confidence = result.data.get("confidence", 0.0)
            has_answer = len(result.content) > 50
            has_sources = sources_count > 0

            # 간단 합격/불합격 판정
            is_passed = has_answer and has_sources

            result_data = {
                "query_id": query_id,
                "query": query,
                "category": category,
                "sources_count": sources_count,
                "confidence": confidence,
                "execution_time_ms": execution_time_ms,
                "has_answer": has_answer,
                "has_sources": has_sources,
                "passed": is_passed,
                "content_preview": result.content[:100] if result.content else "",
                "result_type": result.type.value,
            }

            results.append(result_data)

            if is_passed:
                passed += 1
                logger.info(f"  ✓ 합격 (소스: {sources_count}, 신뢰: {confidence:.2f})")
            else:
                failed += 1
                logger.warning(
                    f"  ✗ 불합격 (답변: {has_answer}, 소스: {sources_count})"
                )

        except Exception as e:
            logger.error(f"  ✗ 오류: {e}")
            failed += 1
            results.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "category": category,
                    "error": str(e),
                    "passed": False,
                }
            )

    # 결과 저장
    total = len(TEST_QUERIES)
    pass_rate = passed / total if total else 0

    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "results": results,
    }

    output_path = (
        Path("test_results")
        / f"quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 결과 출력
    print("\n" + "=" * 70)
    print("평가 완료!")
    print("=" * 70)
    print(f"합격률: {pass_rate:.1%}")
    print(f"결과 저장: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
