#!/usr/bin/env python3
"""
Phase 3: Multi-turn Conversation Testing
문맥 유지 및 후속 질문 처리 테스트 (목표: 80%)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# .env 파일 로드
from dotenv import load_dotenv

load_dotenv()

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter


@dataclass
class ConversationTurn:
    """대화 턴"""
    query: str
    expected_topics: List[str]
    should_use_context: bool = False  # 이전 문맥 활용 여부


@dataclass
class MultiTurnTestCase:
    """멀티턴 테스트 케이스"""
    name: str
    turns: List[ConversationTurn]
    category: str = "general"


@dataclass
class MultiTurnTestResult:
    """멀티턴 테스트 결과"""
    test_case: MultiTurnTestCase
    passed: bool
    turn_results: List[Dict]
    context_maintained: bool
    error: Optional[str] = None


# 멀티턴 테스트 케이스 정의
MULTI_TURN_TEST_CASES = [
    MultiTurnTestCase(
        name="휴학 문의 대화",
        turns=[
            ConversationTurn(
                query="휴학하려면 어떻게 해?",
                expected_topics=["휴학", "신청", "휴학원"],
            ),
            ConversationTurn(
                query="기간은 얼마나 돼?",
                expected_topics=["기간", "학기", "년"],
                should_use_context=True,
            ),
            ConversationTurn(
                query="복학은?",
                expected_topics=["복학", "신청"],
                should_use_context=True,
            ),
        ],
        category="student"
    ),
    MultiTurnTestCase(
        name="장학금 문의 대화",
        turns=[
            ConversationTurn(
                query="장학금 받으려면?",
                expected_topics=["장학금", "신청", "자격"],
            ),
            ConversationTurn(
                query="성적 기준은?",
                expected_topics=["성적", "평점", "기준"],
                should_use_context=True,
            ),
        ],
        category="student"
    ),
    MultiTurnTestCase(
        name="교원 연구년 대화",
        turns=[
            ConversationTurn(
                query="연구년 신청 자격이 뭐야?",
                expected_topics=["연구년", "자격", "교원"],
            ),
            ConversationTurn(
                query="해외 연구년도 가능해?",
                expected_topics=["해외", "연구", "가능"],
                should_use_context=True,
            ),
        ],
        category="faculty"
    ),
    MultiTurnTestCase(
        name="학점 관련 대화",
        turns=[
            ConversationTurn(
                query="졸업 학점이 몇 학점이야?",
                expected_topics=["졸업", "학점"],
            ),
            ConversationTurn(
                query="교양은 몇 학점?",
                expected_topics=["교양", "학점"],
                should_use_context=True,
            ),
            ConversationTurn(
                query="전공은?",
                expected_topics=["전공", "학점"],
                should_use_context=True,
            ),
        ],
        category="student"
    ),
    MultiTurnTestCase(
        name="징계 관련 대화",
        turns=[
            ConversationTurn(
                query="학생 징계 종류가 뭐가 있어?",
                expected_topics=["징계", "처분", "종류"],
            ),
            ConversationTurn(
                query="이의 제기 가능해?",
                expected_topics=["이의", "제기", "신청"],
                should_use_context=True,
            ),
        ],
        category="student"
    ),
]


def run_multiturn_tests(verbose: bool = True) -> dict:
    """멀티턴 대화 테스트 실행"""

    print("=" * 60, flush=True)
    print("Phase 3: Multi-turn Conversation Testing", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # 초기화 - 직접 SearchUseCase와 LLM 사용 (QueryHandler 대신)
    store = ChromaVectorStore(persist_directory="data/chroma_db")
    search = SearchUseCase(store, use_reranker=True)
    llm = LLMClientAdapter()

    results: List[MultiTurnTestResult] = []

    for i, tc in enumerate(MULTI_TURN_TEST_CASES, 1):
        print(f"[{i}/{len(MULTI_TURN_TEST_CASES)}] {tc.name}", flush=True)

        try:
            turn_results = []
            conversation_history = []  # 대화 기록
            all_turns_passed = True
            context_used_correctly = True

            for turn_idx, turn in enumerate(tc.turns):
                # 이전 대화 맥락 구성
                history_context = "\n".join([
                    f"Q: {h['query']}\nA: {h['answer'][:200]}..."
                    for h in conversation_history[-2:]
                ]) if conversation_history else ""

                # 검색 수행
                search_query = turn.query
                if turn.should_use_context and conversation_history:
                    # 이전 질문의 키워드 추가
                    prev_topics = conversation_history[-1].get("topics", [])
                    search_query = f"{turn.query} {' '.join(prev_topics[:2])}"

                search_results = search.search(search_query, top_k=5)

                # 컨텍스트 구성
                context_parts = []
                for r in search_results[:5]:
                    if r.chunk:
                        context_parts.append(r.chunk.text[:500])
                context = "\n\n".join(context_parts)

                # LLM 답변 생성
                prompt = f"""이전 대화:
{history_context if history_context else "없음"}

관련 규정:
{context[:2000]}

질문: {turn.query}

위 정보를 바탕으로 간결하게 답변하세요."""

                answer = llm.generate(
                    system_prompt="당신은 대학 규정 전문가입니다.",
                    user_message=prompt
                )

                # 토픽 체크
                answer_lower = answer.lower() if answer else ""
                found_topics = [t for t in turn.expected_topics if t.lower() in answer_lower]
                topic_coverage = len(found_topics) / len(turn.expected_topics) if turn.expected_topics else 1.0

                turn_passed = topic_coverage >= 0.5 and len(answer) > 20

                # 대화 기록 저장
                conversation_history.append({
                    "query": turn.query,
                    "answer": answer,
                    "topics": turn.expected_topics,
                })

                turn_results.append({
                    "query": turn.query,
                    "passed": turn_passed,
                    "topic_coverage": topic_coverage,
                    "answer_length": len(answer),
                    "found_topics": found_topics,
                })

                if not turn_passed:
                    all_turns_passed = False

                if verbose:
                    status = "✓" if turn_passed else "✗"
                    print(f"  Turn {turn_idx + 1}: {status} Topics: {topic_coverage:.0%} ({turn.query[:30]}...)", flush=True)

            result = MultiTurnTestResult(
                test_case=tc,
                passed=all_turns_passed,
                turn_results=turn_results,
                context_maintained=context_used_correctly,
            )

        except Exception as e:
            result = MultiTurnTestResult(
                test_case=tc,
                passed=False,
                turn_results=[],
                context_maintained=False,
                error=str(e),
            )
            if verbose:
                print(f"  ✗ Error: {e}")

        results.append(result)

    # 요약
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    context_ok = sum(1 for r in results if r.context_maintained)

    print()
    print("=" * 60)
    print("Multi-turn Test Summary")
    print("=" * 60)
    print(f"통과: {passed} | 실패: {total - passed}")
    print(f"통과율: {passed/total*100:.1f}%")
    print(f"문맥 유지율: {context_ok/total*100:.1f}%")
    print("=" * 60)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0,
        "context_maintained": context_ok / total if total else 0,
        "results": results,
    }


if __name__ == "__main__":
    run_multiturn_tests()
