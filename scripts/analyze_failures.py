#!/usr/bin/env python3
"""Test intent matching for evaluation failures."""

import json
from pathlib import Path

from src.rag.application.evaluate import EvaluationUseCase, TestCase
from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore

# Load store
store = ChromaVectorStore(persist_directory="data/chroma_db")
search = SearchUseCase(store, use_reranker=False)
evaluator = EvaluationUseCase(search_usecase=search)

# Test specific failing queries
failures = [
    "연구 부정행위 신고하고 싶어",
    "강의 면제 받으려면?",
    "취업 준비로 졸업 미루고 싶어",
    "징계 절차가 어떻게 돼?",
    "원격수업 규정이 어떻게 돼?",
    "학생 창업 지원받을 수 있어?",
    "취업 지원 프로그램이 있어?",
    "교수가 과제 기한 너무 짧게 줬어",
    "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래",
]

# Load dataset for expected values
data_path = Path("data/config/evaluation_dataset.json")
data = json.loads(data_path.read_text())
tc_map = {tc["query"]: tc for tc in data["test_cases"]}

print("=== Detailed Failure Analysis ===\n")

for query in failures:
    tc_data = tc_map.get(query, {})
    tc = TestCase(
        id=tc_data.get("id", query[:20]),
        query=query,
        expected_intents=tc_data.get("expected_intents", []),
        expected_keywords=tc_data.get("expected_keywords", []),
        expected_rule_codes=tc_data.get("expected_rule_codes", []),
        min_relevance_score=tc_data.get("min_relevance_score", 0.05),
        category=tc_data.get("category", "general"),
    )

    result = evaluator.evaluate_single(tc, top_k=5)

    print(f"=== {query} ===")
    print(f"  Passed: {result.passed}")
    print(f"  Intent matched: {result.intent_matched} (expected: {tc.expected_intents})")
    print(f"  Matched intents: {result.matched_intents}")
    print(f"  Keyword coverage: {result.keyword_coverage:.2f} (found: {result.found_keywords})")
    print(f"  Rule code matched: {result.rule_code_matched} (found: {result.found_rule_codes})")
    print(f"  Top score: {result.top_score:.3f} (min: {tc.min_relevance_score})")
    if result.error:
        print(f"  Error: {result.error}")
    print()
