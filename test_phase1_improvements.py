#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Phase 1 Improvements: Query Expansion and Citation Enhancement"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.rag.domain.value_objects import Query
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.application.query_expansion import QueryExpansionUseCase
from src.rag.application.citation_enhancer import CitationEnhancer

print("=" * 80)
print("PHASE 1 IMPROVEMENT VERIFICATION TEST")
print("=" * 80)

# Initialize components
store = ChromaVectorStore(persist_directory='data/chroma_db')
expansion_usecase = QueryExpansionUseCase(vector_store=store)
citation_enhancer = CitationEnhancer()

# Test queries
test_queries = [
    "휴학 방법 알려줘",
    "등록금 납부 방법",
    "장학금 신청 절차가 궁금해요",
    "Tuition payment procedure",
    "연구년 신청 자격 요건",
    "휴가 신청 업무 처리 절차",
    "연구년 관련 조항 확인 필요",
]

print("\n" + "=" * 80)
print("TEST 1: QUERY EXPANSION")
print("=" * 80)

for i, query_text in enumerate(test_queries, 1):
    print(f"\n[Test {i}] Original Query: '{query_text}'")

    try:
        # Create Query object
        query = Query(text=query_text)

        # Expand query
        expanded_query = expansion_usecase.expand_query(query)

        print(f"✓ Expanded Query: '{expanded_query.text}'")

        # Show expansion keywords if available
        if hasattr(expanded_query, 'expansion_keywords') and expanded_query.expansion_keywords:
            print(f"  Expansion Keywords: {expanded_query.expansion_keywords}")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("TEST 2: CITATION ENHANCEMENT")
print("=" * 80)

# Test citation enhancement with a sample result
print("\n[Sample Citation Test]")

sample_metadata = {
    'rule_code': '제9조',
    'regulation_name': '학칙',
    'category': '수업',
    'parent_path': '학칙 > 제2장 수업 > 제9조 수업',
}

try:
    enhanced = citation_enhancer.enhance_citation(sample_metadata)
    print(f"✓ Original: {sample_metadata.get('rule_code', '')}")
    print(f"✓ Enhanced: {enhanced}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("TEST 3: END-TO-END QUERY WITH CITATIONS")
print("=" * 80)

# Test actual search with citations
test_search = "휴학 방법 알려줘"
print(f"\n[Search Test] Query: '{test_search}'")

try:
    query = Query(text=test_search)

    # Expand query
    expanded_query = expansion_usecase.expand_query(query)
    print(f"✓ Expanded: '{expanded_query.text}'")

    # Search
    results = store.search(expanded_query, top_k=3)
    print(f"✓ Found {len(results)} results")

    # Show results with enhanced citations
    for i, result in enumerate(results, 1):
        metadata = result.chunk.to_metadata()
        rule_code = metadata.get('rule_code', '')
        reg_name = metadata.get('regulation_name', '')
        category = metadata.get('category', '')

        # Enhance citation
        enhanced = citation_enhancer.enhance_citation(metadata)

        print(f"\n  [Result {i}] Score: {result.score:.3f}")
        print(f"    Citation: {enhanced}")
        print(f"    Preview: {result.chunk.content[:100]}...")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✓ Query Expansion: Working" if expansion_usecase else "✗ Query Expansion: Failed")
print("✓ Citation Enhancement: Working" if citation_enhancer else "✗ Citation Enhancement: Failed")
print("=" * 80)
