#!/usr/bin/env python3
"""실패 케이스 정밀 분석"""
import sys

sys.path.insert(0, 'src')

from src.rag.domain.value_objects import Query
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.query_analyzer import QueryAnalyzer

store = ChromaVectorStore(persist_directory='data/chroma_db')
analyzer = QueryAnalyzer()

failed_queries = [
    ("연구 부정행위 신고하고 싶어", ["연구진실성", "부정행위", "연구윤리"]),
    ("강의 면제 받으려면?", ["수강면제", "학점인정", "면제"]),
    ("취업 준비로 졸업 미루고 싶어", ["졸업연기", "졸업유예", "휴학"]),
    ("징계 절차가 어떻게 돼?", ["징계", "징계위원회"]),
    ("학생 창업 지원받을 수 있어?", ["창업", "창업지원"]),
    ("취업 지원 프로그램이 있어?", ["취업지원", "취업", "진로", "취창업"])
]

for query_text, expected in failed_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query_text}")
    print(f"Expected: {expected}")
    print("-"*60)

    # Intent analysis
    intents = analyzer._match_intents(query_text)
    if intents:
        for i in intents[:2]:
            print(f"Intent: {i.intent_id} / {i.label}")
            print(f"  Keywords: {i.keywords}")
    else:
        print("Intent: No match")

    # Expanded query
    expanded = analyzer.expand_query(query_text)
    print(f"Expanded: {expanded}")

    # Actual search
    query = Query(text=expanded)
    results = store.search(query, top_k=5)
    print("\nSearch Results:")
    for r in results:
        meta = r.chunk.to_metadata()
        parent = meta.get('parent_path', meta.get('rule_code', ''))
        content_preview = r.chunk.text[:80].replace('\n', ' ')
        print(f"  [{r.score:.3f}] {parent}")
        print(f"           -> {content_preview}...")

    # Check expected keywords
    found_any = False
    for r in results[:3]:
        chunk_text = r.chunk.text.lower()
        meta = r.chunk.to_metadata()
        all_text = f"{chunk_text} {str(meta).lower()}"
        for exp in expected:
            if exp.lower() in all_text or exp in all_text:
                found_any = True
                print(f"  [OK] '{exp}' found in result!")
                break

    if not found_any:
        print("  [FAIL] Expected keywords not found in top-3")
