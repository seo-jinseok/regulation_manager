#!/usr/bin/env python3
"""실제 반환되는 인텐트 확인"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

# Setup
store = ChromaVectorStore(persist_directory="data/chroma_db")
llm = LLMClientAdapter()
search = SearchUseCase(store, llm_client=llm, use_reranker=False)

# Test queries (failed cases)
test_queries = [
    "연구 부정행위 신고하고 싶어",
    "취업 준비로 졸업 미루고 싶어",
    "징계 절차가 어떻게 돼?",
    "학생 창업 지원받을 수 있어?",
    "취업 지원 프로그램이 있어?",
]

print("=== 실제 반환 인텐트 확인 ===\n")

for q in test_queries:
    results = search.search(q, top_k=5)
    rewrite_info = search.get_last_query_rewrite()

    print(f"Query: {q}")
    print(f"  Rewritten: {rewrite_info.rewritten[:80] if rewrite_info else 'N/A'}...")
    print(f"  Matched Intents: {rewrite_info.matched_intents if rewrite_info else []}")
    print(f"  Found Codes: {[r.chunk.rule_code for r in results[:3]]}")
    print()
