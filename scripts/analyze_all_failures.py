#!/usr/bin/env python3
"""실패 케이스 상세 분석"""

from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.application.search_usecase import SearchUseCase
from src.rag.application.evaluate import EvaluationUseCase

store = ChromaVectorStore(persist_directory="data/chroma_db")
search = SearchUseCase(store, use_reranker=False)
evaluator = EvaluationUseCase(search_usecase=search)

# 전체 평가 실행
summary = evaluator.run_evaluation(top_k=5)

print("=== 실패 케이스 상세 분석 ===")
print()
for r in summary.results:
    if not r.passed:
        print(f"Query: {r.test_case.query}")
        print(f"  - Intent: {r.intent_matched}")
        print(f"    Expected: {r.test_case.expected_intents}")
        print(f"    Got: {r.matched_intents}")
        print(f"  - Keywords: {r.keyword_coverage:.0%}")
        print(f"    Expected: {r.test_case.expected_keywords}")
        print(f"    Found: {r.found_keywords}")
        print(f"  - Rule codes: {r.rule_code_matched}")
        print(f"    Expected: {r.test_case.expected_rule_codes}")
        print(f"    Found: {r.found_rule_codes}")
        print(f"  - Score: {r.top_score:.3f} (min: {r.test_case.min_relevance_score})")
        print()

print(f"\n총 실패: {summary.total_cases - summary.passed_cases} / {summary.total_cases}")
