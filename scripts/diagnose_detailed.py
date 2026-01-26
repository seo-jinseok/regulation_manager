#!/usr/bin/env python3
"""Detailed diagnosis of failed cases - MATCHING auto_evaluate setup"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

load_dotenv()

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

store = ChromaVectorStore(persist_directory='data/chroma_db')

# Match auto_evaluate setup: with LLM client
llm = None
try:
    llm = LLMClientAdapter()
except Exception as e:
    print(f"Warning: LLM init failed ({e})")

search = SearchUseCase(store, llm_client=llm, use_reranker=False)

# The 5 remaining failed cases
test_cases = [
    {
        "query": "연구 부정행위 신고하고 싶어",
        "expected_intents": ["연구윤리 관련"],
        "expected_keywords": ["연구윤리", "부정행위", "신고"],
        "expected_rule_codes": ["3-1-13", "3-1-2"],
        "min_relevance_score": 0.05
    },
    {
        "query": "취업 준비로 졸업 미루고 싶어",
        "expected_intents": ["졸업유예 관심"],
        "expected_keywords": ["졸업유예", "졸업연기", "학사학위취득유예", "학위유예"],
        "expected_rule_codes": [],
        "min_relevance_score": 0.1
    },
    {
        "query": "징계 절차가 어떻게 돼?",
        "expected_intents": ["징계 관련"],
        "expected_keywords": ["징계", "징계위원회", "징계처분"],
        "expected_rule_codes": ["3-1-5", "2-1-1", "3-3-2"],
        "min_relevance_score": 0.05
    },
    {
        "query": "학생 창업 지원받을 수 있어?",
        "expected_intents": ["창업 관심"],
        "expected_keywords": ["창업", "창업지원"],
        "expected_rule_codes": ["5-1-31", "6-0-2"],
        "min_relevance_score": 0.05
    },
    {
        "query": "취업 지원 프로그램이 있어?",
        "expected_intents": ["취업 관심"],
        "expected_keywords": ["취업", "취업지원"],
        "expected_rule_codes": [],
        "min_relevance_score": 0.1
    },
]

print("=== Detailed Evaluation Diagnosis ===\n")

for tc in test_cases:
    query = tc["query"]
    print(f"Query: {query}")

    results = search.search(query, top_k=5)
    rewrite = search.get_last_query_rewrite()

    # Check intent
    matched_intents = rewrite.matched_intents if rewrite else []
    intent_matched = any(exp.lower() in [m.lower() for m in matched_intents] for exp in tc["expected_intents"])
    print(f"  Matched Intents: {matched_intents}")
    print(f"  Intent Matched: {intent_matched}")

    # Check keywords in rewritten query
    rewritten = rewrite.rewritten if rewrite else query
    found_kws = [kw for kw in tc["expected_keywords"] if kw.lower() in rewritten.lower()]
    kw_coverage = len(found_kws) / len(tc["expected_keywords"]) if tc["expected_keywords"] else 1.0
    print(f"  Expected KWs: {tc['expected_keywords']}")
    print(f"  Found KWs: {found_kws}")
    print(f"  KW Coverage: {kw_coverage:.1%}")

    # Check rule codes
    found_codes = [r.chunk.rule_code for r in results]
    exp_codes = tc["expected_rule_codes"]
    if exp_codes:
        code_matched = any(c in found_codes for c in exp_codes)
    else:
        code_matched = True
    print(f"  Expected Codes: {exp_codes}")
    print(f"  Found Codes: {found_codes}")
    print(f"  Code Matched: {code_matched}")

    # Check score
    top_score = results[0].score if results else 0.0
    score_ok = top_score >= tc["min_relevance_score"]
    print(f"  Top Score: {top_score:.3f} (min: {tc['min_relevance_score']})")
    print(f"  Score OK: {score_ok}")

    # Final
    passed = (
        (not tc["expected_intents"] or intent_matched) and
        kw_coverage >= 0.5 and
        code_matched and
        score_ok
    )
    print(f"  => PASSED: {passed}")
    print()
