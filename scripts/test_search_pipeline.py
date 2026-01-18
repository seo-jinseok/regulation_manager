#!/usr/bin/env python3
"""Test full search pipeline with updated weights"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore

store = ChromaVectorStore(persist_directory='data/chroma_db')
search = SearchUseCase(store, use_reranker=False)

failed_queries = [
    ("연구 부정행위 신고하고 싶어", ['3-1-13', '3-1-2'], ['연구윤리', '부정행위', '신고']),
    ("편입학 자격이 어떻게 돼?", ['2-1-1', '3-2-113'], ['편입', '편입학']),
    ("취업 준비로 졸업 미루고 싶어", [], ['졸업유예', '졸업연기']),
    ("징계 절차가 어떻게 돼?", ['3-1-5', '2-1-1', '3-3-2'], ['징계', '징계위원회', '징계처분']),
    ("학생 창업 지원받을 수 있어?", ['5-1-31', '6-0-2'], ['창업', '창업지원']),
    ("취업 지원 프로그램이 있어?", [], ['취업', '취업지원']),
]

print("=== Full Search Pipeline Test ===\n")

for query, expected_codes, expected_keywords in failed_queries:
    print(f"Query: {query}")
    print(f"  Expected codes: {expected_codes}")
    print(f"  Expected keywords: {expected_keywords}")
    
    results = search.search(query, top_k=5)
    
    # Get query rewrite info
    rewrite = search.get_last_query_rewrite()
    if rewrite:
        print(f"  Rewritten: {rewrite.rewritten[:80]}...")
    
    print(f"  Results:")
    found_codes = []
    found_keywords_in_results = set()
    
    for r in results:
        code = r.chunk.rule_code
        found_codes.append(code)
        content = r.chunk.text.lower()
        
        # Check keywords in result
        for kw in expected_keywords:
            if kw.lower() in content:
                found_keywords_in_results.add(kw)
        
        parent = ' > '.join(r.chunk.parent_path) if r.chunk.parent_path else code
        print(f"    [{r.score:.3f}] {code} | {parent[:50]}...")
    
    # Check pass/fail
    code_match = not expected_codes or any(c in found_codes for c in expected_codes)
    kw_coverage = len(found_keywords_in_results) / len(expected_keywords) if expected_keywords else 1.0
    
    status = "PASS" if code_match and kw_coverage >= 0.5 else "FAIL"
    print(f"  Code Match: {code_match} | KW Coverage: {kw_coverage:.1%} | Status: {status}")
    print()
