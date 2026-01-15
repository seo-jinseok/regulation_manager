#!/usr/bin/env python
"""실패 케이스 상세 분석 스크립트 - Reranker 비교."""

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore


def analyze_query(usecase, query, expected_intents, expected_rule_codes, min_score):
    """단일 쿼리 분석."""
    results = usecase.search(query, top_k=5)
    rewrite = usecase.get_last_query_rewrite()

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print("=" * 60)

    # Intent check
    print(f"\n[Intent Check]")
    print(f"  Expected: {expected_intents}")
    matched = rewrite.matched_intents if rewrite else []
    print(f"  Matched: {matched}")
    expected_lower = {i.lower() for i in expected_intents}
    matched_lower = {m.lower() for m in matched} if matched else set()
    intent_ok = bool(expected_lower & matched_lower)
    print(f"  Result: {'✅ PASS' if intent_ok else '❌ FAIL'}")

    # Rule code check
    print(f"\n[Rule Code Check]")
    print(f"  Expected: {expected_rule_codes}")
    found_codes = [r.chunk.rule_code for r in results if r.chunk]
    print(f"  Found: {found_codes}")
    rule_code_ok = bool(set(expected_rule_codes) & set(found_codes)) if expected_rule_codes else True
    print(f"  Result: {'✅ PASS' if rule_code_ok else '❌ FAIL'}")

    # Score check
    print(f"\n[Score Check]")
    top_score = results[0].score if results else 0.0
    print(f"  Top score: {top_score:.4f}")
    print(f"  Min threshold: {min_score}")
    score_ok = top_score >= min_score
    print(f"  Result: {'✅ PASS' if score_ok else '❌ FAIL'}")

    # Overall
    overall = intent_ok and rule_code_ok and score_ok
    print(f"\n[Overall]: {'✅ PASS' if overall else '❌ FAIL'}")
    return overall


def main():
    # Load store
    store = ChromaVectorStore(persist_directory="data/chroma_db")
    
    # Test cases - matching evaluation_dataset.json
    test_queries = [
        (
            "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래",
            ["교수 행동 불만 (학생)"],
            ["5-1-38"],
            0.05,
        ),
        (
            "강의실 예약하고 싶어",
            ["시설 이용 관련"],
            ["3-1-50", "2-1-1", "3-3-1"],
            0.05,
        ),
    ]

    print("\n" + "="*70)
    print("WITHOUT RERANKER")
    print("="*70)
    usecase_no_rerank = SearchUseCase(store, use_reranker=False)
    for query, expected_intents, expected_rule_codes, min_score in test_queries:
        analyze_query(usecase_no_rerank, query, expected_intents, expected_rule_codes, min_score)

    print("\n\n" + "="*70)
    print("WITH RERANKER")
    print("="*70)
    usecase_rerank = SearchUseCase(store, use_reranker=True)
    for query, expected_intents, expected_rule_codes, min_score in test_queries:
        analyze_query(usecase_rerank, query, expected_intents, expected_rule_codes, min_score)


if __name__ == "__main__":
    main()
