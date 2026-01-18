#!/usr/bin/env python3
"""Check graduation delay results"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.application.search_usecase import SearchUseCase
from src.rag.infrastructure.chroma_store import ChromaVectorStore

store = ChromaVectorStore(persist_directory='data/chroma_db')
search = SearchUseCase(store, use_reranker=False)

results = search.search('취업 준비로 졸업 미루고 싶어', top_k=3)
for r in results[:2]:
    print(f'Rule: {r.chunk.rule_code}')
    print(f'Title: {r.chunk.title}')
    print(f'Text: {r.chunk.text[:400]}')
    print('---')
