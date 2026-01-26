#!/usr/bin/env python3
"""Check if expected rule codes exist in DB"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.infrastructure.chroma_store import ChromaVectorStore

store = ChromaVectorStore(persist_directory='data/chroma_db')

# Expected rule codes from failed cases
expected_codes = [
    '3-1-13',  # research ethics
    '3-1-2',   # research ethics
    '2-1-1',   # 학칙
    '3-2-113', # 편입
    '3-1-5',   # 교원인사규정
    '3-3-2',   # 학생상벌규정
    '5-1-31',  # 창업보육센터
    '6-0-2',   # 산학협력단
]

print("=== Checking Rule Codes in DB ===\n")

# Get all chunks and their rule codes
collection = store._collection
all_data = collection.get(include=['metadatas'])
all_rule_codes = set()
for meta in all_data['metadatas']:
    if meta and 'rule_code' in meta:
        all_rule_codes.add(meta['rule_code'])

print(f"Total unique rule codes in DB: {len(all_rule_codes)}")
print()

for code in expected_codes:
    if code in all_rule_codes:
        print(f"[OK] {code} - EXISTS")
    else:
        # Check partial match
        partial = [c for c in all_rule_codes if code in c or c in code]
        if partial:
            print(f"[WARN] {code} - NOT FOUND (similar: {partial[:3]})")
        else:
            print(f"[FAIL] {code} - NOT FOUND")
