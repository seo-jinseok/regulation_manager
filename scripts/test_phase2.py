#!/usr/bin/env python
"""Phase 2 테스트: 4개 실패 쿼리 검색 품질 확인"""

from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.application.search_usecase import SearchUseCase

store = ChromaVectorStore(persist_directory="data/chroma_db")
usecase = SearchUseCase(store, use_reranker=False, use_hybrid=True, enable_warmup=False)

queries = [
    "장학금 받으려면 성적이 몇 점이어야 해?",
    "졸업학점이 몇 학점이야?",
    "졸업하려면 영어 점수도 필요해?",
    "교수 승진 기준이 어떻게 됩니까?",
]

print("=" * 60)
print("Phase 2 테스트: 4개 실패 쿼리 검색 품질 확인")
print("=" * 60)

for q in queries:
    results = usecase.search(q, top_k=3)
    print(f"\n=== Query: {q} ===")
    if results:
        for r in results[:3]:
            c = r.chunk
            title = getattr(c, "parent_path", None) or c.title
            display = getattr(c, "display_no", None) or c.title
            print(f"  - [{r.score:.3f}] {title} / {display}")
            print(f"    Text: {c.text[:100]}...")
    else:
        print("  No results")
