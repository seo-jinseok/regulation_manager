#!/usr/bin/env python3
"""상세한 실패 원인 분석"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.application.evaluate import EvaluationUseCase
from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

# Load dataset
dataset_path = PROJECT_ROOT / "data" / "config" / "evaluation_dataset.json"
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

# Setup
store = ChromaVectorStore(persist_directory="data/chroma_db")
llm = LLMClientAdapter()
search = SearchUseCase(store, llm_client=llm, use_reranker=False)
eval_usecase = EvaluationUseCase(search_usecase=search)

# Failed queries from auto_evaluate
failed_queries = [
    "연구 부정행위 신고하고 싶어",
    "강의 면제 받으려면?",
    "취업 준비로 졸업 미루고 싶어",
    "징계 절차가 어떻게 돼?",
    "학생 창업 지원받을 수 있어?",
    "취업 지원 프로그램이 있어?",
]

# Find test cases
test_cases = {item["query"]: item for item in dataset.get("test_cases", [])}

print("=== 실패 케이스 상세 분석 ===\n")

for query in failed_queries:
    tc = test_cases.get(query)
    if not tc:
        print(f"[NOT FOUND] {query}")
        continue

    # Run search
    results = search.search(query, top_k=5)
    rewrite_info = search.get_last_query_rewrite()

    # Get metrics
    matched_intents = rewrite_info.matched_intents if rewrite_info else []
    rewritten = rewrite_info.rewritten if rewrite_info else query
    found_codes = [r.chunk.rule_code for r in results[:5]]
    top_score = results[0].score if results else 0.0

    # Check conditions
    expected_intents = tc.get("expected_intents", [])
    expected_keywords = tc.get("expected_keywords", [])
    expected_codes = tc.get("expected_rule_codes", [])
    min_score = tc.get("min_relevance_score", 0.1)

    # Intent match
    intent_matched = bool(set([i.lower() for i in expected_intents]) & set([m.lower() for m in matched_intents])) if expected_intents else True

    # Keyword coverage
    found_keywords = [kw for kw in expected_keywords if kw.lower() in rewritten.lower()]
    keyword_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0

    # Code match
    code_matched = bool(set(expected_codes) & set(found_codes)) if expected_codes else True

    # Score pass
    score_pass = top_score >= min_score

    # Overall
    passed = (
        (not expected_intents or intent_matched)
        and keyword_coverage >= 0.5
        and (not expected_codes or code_matched)
        and score_pass
    )

    print(f"Query: {query}")
    print(f"  ID: {tc['id']}")
    print()
    print(f"  1. Intent Match: {'✅' if intent_matched else '❌'}")
    print(f"     Expected: {expected_intents}")
    print(f"     Matched:  {matched_intents}")
    print()
    print(f"  2. Keyword Coverage: {'✅' if keyword_coverage >= 0.5 else '❌'} ({keyword_coverage:.0%})")
    print(f"     Expected: {expected_keywords}")
    print(f"     Found:    {found_keywords}")
    print()
    print(f"  3. Code Match: {'✅' if code_matched else '❌'}")
    print(f"     Expected: {expected_codes}")
    print(f"     Found:    {found_codes}")
    print()
    print(f"  4. Score Pass: {'✅' if score_pass else '❌'}")
    print(f"     Min: {min_score}, Actual: {top_score:.4f}")
    print()
    print(f"  => OVERALL: {'PASS ✅' if passed else 'FAIL ❌'}")
    print()
    print("-" * 60)
    print()
