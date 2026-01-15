#!/usr/bin/env python
"""원격수업 쿼리 분석 스크립트."""

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore


def main():
    store = ChromaVectorStore(persist_directory="data/chroma_db")
    usecase = SearchUseCase(store, use_reranker=False)

    query = "원격수업 규정이 어떻게 돼?"
    results = usecase.search(query, top_k=10)
    rewrite = usecase.get_last_query_rewrite()

    print(f"Query: {query}")
    print(f"\nMatched intents: {rewrite.matched_intents if rewrite else []}")
    print(f"Rewritten query: {rewrite.rewritten if rewrite else query}")

    print(f"\n=== Top 10 Results ===")
    for i, r in enumerate(results[:10]):
        c = r.chunk
        print(f"{i+1}. [{c.rule_code}] {c.title} (score: {r.score:.4f})")


if __name__ == "__main__":
    main()
