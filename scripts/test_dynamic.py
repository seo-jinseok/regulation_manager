#!/usr/bin/env python3
"""
Phase 2: Dynamic Query Testing
실시간 검색 및 답변 품질 테스트 (목표: 80%)
"""

import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.llm_adapter import LLMClientAdapter


@dataclass
class DynamicTestCase:
    """동적 테스트 케이스"""
    query: str
    expected_topics: List[str]  # 답변에 포함되어야 할 주제
    expected_regulations: List[str] = field(default_factory=list)  # 참조되어야 할 규정
    min_response_time: float = 30.0  # 최대 응답 시간 (초)
    category: str = "general"


@dataclass 
class DynamicTestResult:
    """동적 테스트 결과"""
    test_case: DynamicTestCase
    passed: bool
    response_time: float
    topic_coverage: float
    regulation_coverage: float
    answer_length: int
    has_answer: bool
    error: Optional[str] = None


# 동적 테스트 케이스 정의
DYNAMIC_TEST_CASES = [
    DynamicTestCase(
        query="휴학 신청 절차 알려줘",
        expected_topics=["휴학원", "제출", "신청", "기간", "학기"],
        expected_regulations=["학칙"],
        category="student_procedure"
    ),
    DynamicTestCase(
        query="교수 연구년 신청하려면?",
        expected_topics=["연구년", "신청", "자격", "기간"],
        expected_regulations=["연구년", "교원"],
        category="faculty_procedure"
    ),
    DynamicTestCase(
        query="장학금 종류가 뭐가 있어?",
        expected_topics=["장학금", "성적", "국가장학금", "교내장학금"],
        expected_regulations=["장학"],
        category="student_info"
    ),
    DynamicTestCase(
        query="학생이 징계 받으면 어떻게 돼?",
        expected_topics=["징계", "처분", "위원회", "절차"],
        expected_regulations=["학칙", "징계"],
        category="student_info"
    ),
    DynamicTestCase(
        query="교원 채용 절차",
        expected_topics=["채용", "심사", "공고", "임용"],
        expected_regulations=["인사", "교원"],
        category="faculty_procedure"
    ),
    DynamicTestCase(
        query="등록금 환불 규정",
        expected_topics=["환불", "등록금", "반환", "비율"],
        expected_regulations=["학칙", "등록금"],
        category="student_info"
    ),
    DynamicTestCase(
        query="학점 인정 범위",
        expected_topics=["학점", "인정", "이수", "학기"],
        expected_regulations=["학칙"],
        category="student_info"
    ),
    DynamicTestCase(
        query="연구비 부정 사용하면?",
        expected_topics=["연구비", "부정", "제재", "환수"],
        expected_regulations=["연구", "윤리"],
        category="research"
    ),
    DynamicTestCase(
        query="졸업 요건이 뭐야?",
        expected_topics=["졸업", "학점", "이수", "요건"],
        expected_regulations=["학칙"],
        category="student_info"
    ),
    DynamicTestCase(
        query="교수가 학생 성희롱하면?",
        expected_topics=["성희롱", "신고", "징계", "피해자"],
        expected_regulations=["성희롱", "인권"],
        category="complaint"
    ),
]


def run_dynamic_tests(verbose: bool = True) -> dict:
    """동적 테스트 실행"""
    
    print("=" * 60)
    print("Phase 2: Dynamic Query Testing")
    print("=" * 60)
    print()
    
    # 초기화
    store = ChromaVectorStore(persist_directory="data/chroma_db")
    search = SearchUseCase(store, use_reranker=True)
    llm = LLMClientAdapter()
    
    results: List[DynamicTestResult] = []
    
    for i, tc in enumerate(DYNAMIC_TEST_CASES, 1):
        print(f"[{i}/{len(DYNAMIC_TEST_CASES)}] Testing: {tc.query}")
        
        try:
            # 시간 측정 시작
            start_time = time.time()
            
            # 검색 + 답변 생성
            search_results = search.search(tc.query, top_k=5)
            
            # 컨텍스트 구성
            context_parts = []
            found_regulations = set()
            for r in search_results[:5]:
                if r.chunk:
                    context_parts.append(r.chunk.text[:500])
                    # 규정명은 parent_path[0]에 있음
                    if r.chunk.parent_path:
                        found_regulations.add(r.chunk.parent_path[0])
            
            context = "\n\n".join(context_parts)
            
            # LLM 답변 생성
            answer = ""
            if context:
                prompt = f"""다음 규정 내용을 바탕으로 질문에 답하세요.

질문: {tc.query}

규정 내용:
{context}

답변:"""
                answer = llm.generate(
                    system_prompt="당신은 대학 규정 전문가입니다. 질문에 간결하게 답변하세요.",
                    user_message=prompt
                )
            
            response_time = time.time() - start_time
            
            # 평가
            answer_lower = answer.lower() if answer else ""
            found_topics = [t for t in tc.expected_topics if t.lower() in answer_lower]
            topic_coverage = len(found_topics) / len(tc.expected_topics) if tc.expected_topics else 1.0
            
            found_regs = [r for r in tc.expected_regulations 
                         if any(r in reg for reg in found_regulations)]
            reg_coverage = len(found_regs) / len(tc.expected_regulations) if tc.expected_regulations else 1.0
            
            # 통과 조건: 토픽 50%+, 응답시간 30초 이내, 답변 존재
            passed = (
                topic_coverage >= 0.5 and
                response_time <= tc.min_response_time and
                len(answer) > 50
            )
            
            result = DynamicTestResult(
                test_case=tc,
                passed=passed,
                response_time=response_time,
                topic_coverage=topic_coverage,
                regulation_coverage=reg_coverage,
                answer_length=len(answer),
                has_answer=len(answer) > 50,
            )
            
            if verbose:
                status = "✓" if passed else "✗"
                print(f"  {status} Time: {response_time:.2f}s, Topics: {topic_coverage:.0%}, Answer: {len(answer)} chars")
            
        except Exception as e:
            result = DynamicTestResult(
                test_case=tc,
                passed=False,
                response_time=0,
                topic_coverage=0,
                regulation_coverage=0,
                answer_length=0,
                has_answer=False,
                error=str(e),
            )
            if verbose:
                print(f"  ✗ Error: {e}")
        
        results.append(result)
    
    # 요약
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_time = sum(r.response_time for r in results) / total if total else 0
    avg_topic = sum(r.topic_coverage for r in results) / total if total else 0
    
    print()
    print("=" * 60)
    print("Dynamic Test Summary")
    print("=" * 60)
    print(f"통과: {passed} | 실패: {total - passed}")
    print(f"통과율: {passed/total*100:.1f}%")
    print(f"평균 응답 시간: {avg_time:.2f}초")
    print(f"평균 토픽 커버리지: {avg_topic:.1%}")
    print("=" * 60)
    
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0,
        "avg_response_time": avg_time,
        "avg_topic_coverage": avg_topic,
        "results": results,
    }


if __name__ == "__main__":
    run_dynamic_tests()
