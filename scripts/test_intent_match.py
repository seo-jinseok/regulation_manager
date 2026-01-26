#!/usr/bin/env python
"""평가 결과 상세 확인 스크립트"""

from src.rag.application.evaluate import EvaluationUseCase, TestCase
from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore


def test_evaluation():
    store = ChromaVectorStore()
    search = SearchUseCase(store, use_reranker=False)
    evaluator = EvaluationUseCase(search)

    # 테스트할 쿼리 목록
    test_queries = [
        ("휴학하고 싶어", ["휴학 관심"], ["휴학", "휴학원", "휴학 신청"], [], 0.1),
        ("연구년 신청하고 싶어", ["연구년 관심"], ["연구년", "안식년", "연구년제"], [], 0.05),
        ("교수에게 부당한 대우를 받았어", ["학생 권리 침해"], ["인권센터", "학생권리"], ["5-1-38"], 0.05),
    ]

    for query, intents, keywords, codes, min_score in test_queries:
        tc = TestCase(
            id="test",
            query=query,
            expected_intents=intents,
            expected_keywords=keywords,
            expected_rule_codes=codes,
            min_relevance_score=min_score
        )

        result = evaluator.evaluate_single(tc, top_k=5)
        print(f"\n=== {query} ===")
        print(f"  Passed: {result.passed}")
        print(f"  Intent matched: {result.intent_matched} (expected: {intents})")
        print(f"  Matched intents: {result.matched_intents}")
        print(f"  Keyword coverage: {result.keyword_coverage:.2f} (found: {result.found_keywords})")
        print(f"  Top score: {result.top_score:.3f} (min: {min_score})")
        print(f"  Rule code matched: {result.rule_code_matched} (found: {result.found_rule_codes})")

if __name__ == "__main__":
    test_evaluation()
