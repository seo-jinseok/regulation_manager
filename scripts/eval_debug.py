#!/usr/bin/env python3
"""EvaluationUseCase를 직접 사용하여 상세 분석"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.application.search_usecase import SearchUseCase
from src.rag.application.evaluate import EvaluationUseCase

# Setup - exactly like auto_evaluate.py
store = ChromaVectorStore(persist_directory="data/chroma_db")
llm = LLMClientAdapter()
search = SearchUseCase(store, llm_client=llm, use_reranker=False)
eval_usecase = EvaluationUseCase(search_usecase=search)

# Run evaluation
summary = eval_usecase.run_evaluation(top_k=5)

print(f"Total: {summary.total_cases}, Pass: {summary.passed_cases}, Fail: {summary.failed_cases}")
print(f"Pass Rate: {summary.pass_rate:.1%}")
print()

# Show failed cases with details
print("=== FAILED CASES ===\n")
for result in summary.results:
    if not result.passed:
        print(f"Query: {result.test_case.query}")
        print(f"  ID: {result.test_case.id}")
        print(f"  Intent Matched: {result.intent_matched}")
        print(f"    Expected: {result.test_case.expected_intents}")
        print(f"    Matched:  {result.matched_intents}")
        print(f"  Keyword Coverage: {result.keyword_coverage:.0%}")
        print(f"    Expected: {result.test_case.expected_keywords}")
        print(f"    Found:    {result.found_keywords}")
        print(f"  Code Matched: {result.rule_code_matched}")
        print(f"    Expected: {result.test_case.expected_rule_codes}")
        print(f"    Found:    {result.found_rule_codes}")
        print(f"  Top Score: {result.top_score:.4f} (min: {result.test_case.min_relevance_score})")
        if result.error:
            print(f"  ERROR: {result.error}")
        print()
