#!/usr/bin/env python3
"""재테스트 스크립트"""
import os

from dotenv import load_dotenv

load_dotenv()

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.query_analyzer import QueryAnalyzer

store = ChromaVectorStore()
analyzer = QueryAnalyzer(intents_path=os.getenv("RAG_INTENTS_PATH"))
usecase = SearchUseCase(store, use_reranker=False)

# 테스트 쿼리들
test_queries = [
    "연차 휴가가 며칠이에요?",
    "돈이 없어서 학교 다니기 힘든데 어떡해야 할지 모르겠어요",
    "지도교수를 바꾸고 싶은데 가능한가요?",
    "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래",
    "강의실 예약하고 싶어"
]

for query in test_queries:
    print("=" * 70)
    print(f"[원본 쿼리] {query}")

    # 인텐트 매칭
    matches = analyzer._match_intents(query)
    if matches:
        print(f"[인텐트 매칭] {matches[0].intent_id} ({matches[0].label}), score={matches[0].score}")
    else:
        print("[인텐트 매칭] 없음")

    # 쿼리 확장
    expanded = analyzer.expand_query(query)
    print(f"[확장 쿼리] {expanded[:100]}...")

    # 검색
    results = usecase.search(expanded, top_k=3)
    print("\n[검색 결과]")
    for i, r in enumerate(results, 1):
        chunk = r.chunk
        meta = chunk.to_metadata()
        parent_path = meta.get("parent_path", "")[:60]
        text_preview = chunk.text[:150].replace("\n", " ")
        print(f"  {i}. [{r.score:.3f}] {parent_path}")
        print(f"     {text_preview}...")
    print()
