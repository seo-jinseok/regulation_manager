#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct keyword search to verify DB content"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.domain.value_objects import Query
from src.rag.infrastructure.chroma_store import ChromaVectorStore

store = ChromaVectorStore(persist_directory='data/chroma_db')

searches = [
    ('research_ethics', 'research ethics violation report'),
    ('disciplinary', 'disciplinary committee'),
    ('employment', 'employment support career'),
    ('graduation_delay', 'graduation delay postpone'),
    ('lecture_exemption', 'lecture exemption research year'),
]

# Korean searches with exact keywords
korean_searches = [
    ('research_ethics_kr', '연구윤리 부정행위 신고'),
    ('disciplinary_kr', '징계 징계위원회 징계처분'),
    ('employment_kr', '취업 취업지원 진로'),
    ('graduation_kr', '졸업유예 졸업연기'),
    ('lecture_kr', '강의면제 연구년'),
]

for name, query_text in korean_searches:
    print(f'\n=== {name}: "{query_text}" ===')
    q = Query(text=query_text)
    results = store.search(q, top_k=5)
    for r in results:
        m = r.chunk.to_metadata()
        parent = m.get('parent_path', m.get('rule_code', ''))
        print(f'  {r.score:.3f} | {parent}')
