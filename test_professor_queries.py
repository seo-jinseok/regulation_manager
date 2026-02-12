#!/usr/bin/env python3
"""
Professor Persona RAG System Test
Tests realistic Korean professor queries and evaluates responses.
"""
import os
import sys
import json
import time
from datetime import datetime

# Prevent MLX from loading (it crashes on this system)
os.environ['MLX_SKIP'] = '1'

# Add project root to path
sys.path.insert(0, '/Users/truestone/Dropbox/repo/University/regulation_manager')

from dotenv import load_dotenv
load_dotenv()

# Force OpenRouter provider to avoid MLX crash
os.environ['LLM_PROVIDER'] = 'openrouter'
os.environ['LLM_MODEL'] = 'z-ai/glm-4.7-flash'
os.environ['LLM_BASE_URL'] = 'https://openrouter.ai/api/v1'

from src.rag.infrastructure.hybrid_search import HybridSearcher
from src.rag.infrastructure.llm_adapter import LLMClientAdapter


def test_professor_query(query: str, category: str, difficulty: str):
    """Test a single professor query and evaluate the response."""
    start_time = time.time()

    try:
        # Initialize retriever and LLM client
        retriever = HybridSearcher()
        llm_client = LLMClientAdapter(
            provider='openrouter',
            model='z-ai/glm-4.7-flash',
            base_url='https://openrouter.ai/api/v1'
        )

        # Retrieve relevant documents
        docs = retriever.retrieve(query, top_k=5)

        # Generate answer
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get('file_name', 'Unknown') for doc in docs]

        prompt = f"""당신은 대학 규정 전문가입니다. 다음 규정 내용을 바탕으로 교수님의 질문에 답변해주세요.

질문: {query}

관련 규정:
{context}

답변 시 다음 사항을 준수해주세요:
1. 정확한 조문 번호를 인용하세요 (예: "교원인사규정 제15조 제2항")
2. 절차와 요건을 명확하게 설명하세요
3. 예외 사항이 있다면 함께 안내하세요
4. 법적 효력이 있는 규정 내용만 기반으로 답변하세요"""

        answer = llm_client.complete(prompt, max_tokens=1000)

        execution_time = (time.time() - start_time) * 1000

        # Evaluate response quality
        evaluation = evaluate_response(query, answer, context, docs)

        return {
            "persona": "professor",
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "result": {
                "answer": answer,
                "sources": sources,
                "context_length": len(context),
                "execution_time_ms": round(execution_time, 2)
            },
            "evaluation": evaluation
        }

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return {
            "persona": "professor",
            "query": query,
            "category": category,
            "difficulty": difficulty,
            "error": str(e),
            "execution_time_ms": round(execution_time, 2)
        }


def evaluate_response(query: str, answer: str, context: str, docs):
    """Evaluate response quality based on professor persona expectations."""
    scores = {
        "legal_accuracy": 0.0,
        "citation_completeness": 0.0,
        "governance_compliance": 0.0,
        "completeness": 0.0
    }

    # Check for article citations (조문 인용)
    import re
    citation_patterns = [
        r'제\d+조',
        r'\d+조',
        r'규정',
        r'학칙'
    ]

    citation_count = 0
    for pattern in citation_patterns:
        citation_count += len(re.findall(pattern, answer))

    if citation_count >= 3:
        scores["citation_completeness"] = 1.0
    elif citation_count >= 1:
        scores["citation_completeness"] = 0.6
    else:
        scores["citation_completeness"] = 0.2

    # Check for procedure mentions
    procedure_keywords = ['절차', '승인', '의결', '심사', '신청', '제출', '기준', '요건']
    procedure_mention = sum(1 for kw in procedure_keywords if kw in answer)
    scores["governance_compliance"] = min(1.0, procedure_mention / 4)

    # Check answer completeness
    if len(answer) > 300:
        scores["completeness"] = 1.0
    elif len(answer) > 100:
        scores["completeness"] = 0.7
    else:
        scores["completeness"] = 0.3

    # Legal accuracy (based on context relevance)
    if len(context) > 500 and docs:
        scores["legal_accuracy"] = 0.8
    elif len(context) > 100:
        scores["legal_accuracy"] = 0.5
    else:
        scores["legal_accuracy"] = 0.2

    # Overall satisfaction
    avg_score = sum(scores.values()) / len(scores)
    if avg_score >= 0.8:
        satisfaction = "very_satisfied"
    elif avg_score >= 0.6:
        satisfaction = "satisfied"
    elif avg_score >= 0.4:
        satisfaction = "neutral"
    else:
        satisfaction = "dissatisfied"

    scores["satisfaction"] = satisfaction
    scores["confidence"] = avg_score

    return scores


def main():
    """Run all professor queries."""
    queries = [
        {
            "query": "교원 연구년 승인 절차와 교원회 의결 사항은 무엇입니까?",
            "category": "faculty_affairs",
            "difficulty": "medium"
        },
        {
            "query": "정년 보장 심사 기준과 제출 서류를 알려주세요",
            "category": "faculty_affairs",
            "difficulty": "medium"
        },
        {
            "query": "교과과정 개편 절차와 교원회 의결 정족수는 어떻게 됩니까?",
            "category": "academic_governance",
            "difficulty": "medium"
        },
        {
            "query": "겸임 교원의 업적 평가 시 본교와 겸임교 기여도 배분 기준을 어느 규정에서 따릅니까?",
            "category": "faculty_affairs",
            "difficulty": "hard"
        },
        {
            "query": "연구윤리 위반 학생 지도 교원의 연구년 자격 제한 규정과 적용 범위를 확인하고 싶습니다",
            "category": "research_administration",
            "difficulty": "hard"
        }
    ]

    results = []

    print("=" * 80)
    print("Professor Persona RAG System Test")
    print("=" * 80)
    print()

    for i, test_case in enumerate(queries, 1):
        print(f"Test {i}/{len(queries)}: {test_case['query'][:50]}...")
        result = test_professor_query(
            test_case['query'],
            test_case['category'],
            test_case['difficulty']
        )
        results.append(result)
        print(f"  ✓ Completed in {result.get('execution_time_ms', 0)}ms")
        print()

    # Generate report
    print("=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    print()

    # Calculate statistics
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"Total Queries: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        avg_scores = {
            "legal_accuracy": sum(r['evaluation']['legal_accuracy'] for r in successful) / len(successful),
            "citation_completeness": sum(r['evaluation']['citation_completeness'] for r in successful) / len(successful),
            "governance_compliance": sum(r['evaluation']['governance_compliance'] for r in successful) / len(successful),
            "completeness": sum(r['evaluation']['completeness'] for r in successful) / len(successful),
        }

        print("Average Scores:")
        for metric, score in avg_scores.items():
            print(f"  {metric}: {score:.2f}")
        print()

        satisfaction_dist = {}
        for r in successful:
            sat = r['evaluation']['satisfaction']
            satisfaction_dist[sat] = satisfaction_dist.get(sat, 0) + 1

        print("Satisfaction Distribution:")
        for sat, count in satisfaction_dist.items():
            print(f"  {sat}: {count}")
        print()

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/Users/truestone/Dropbox/repo/University/regulation_manager/professor_test_results_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {output_file}")

    # Show first result as example
    if successful:
        print()
        print("=" * 80)
        print("Example Response (Test 1)")
        print("=" * 80)
        print()
        print("Query:", successful[0]['query'])
        print()
        print("Answer:", successful[0]['result']['answer'][:500] + "...")
        print()
        print("Sources:", successful[0]['result']['sources'])
        print()
        print("Evaluation:", successful[0]['evaluation'])


if __name__ == "__main__":
    main()
