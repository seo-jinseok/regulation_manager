#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test query analyzer and weights after code change"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Search using query analyzer expansion
from src.rag.infrastructure.query_analyzer import QueryAnalyzer
analyzer = QueryAnalyzer()

test_queries = [
    "징계 절차가 어떻게 돼?",
    "연구 부정행위 신고하고 싶어",
    "취업 지원 프로그램이 있어?",
    "취업 준비로 졸업 미루고 싶어",
    "강의 면제 받으려면?",
    "학생 창업 지원받을 수 있어?"
]

print("=== Query Analyzer Test (after WEIGHT_PRESETS change) ===\n")

for q in test_queries:
    print(f"Query: {q}")
    
    # Check query type detection
    qtype = analyzer.analyze(q)
    print(f"  Query Type: {qtype}")
    
    # Get weights
    weights = analyzer.get_weights(q)
    print(f"  Weights (BM25, Dense): {weights}")
    
    # Expanded query
    expanded = analyzer.expand_query(q)
    if len(expanded) > 80:
        expanded = expanded[:80] + "..."
    print(f"  Expanded: {expanded}")
    
    # Intent matches
    intents = analyzer._match_intents(q)
    if intents:
        top_intent = intents[0]
        print(f"  Intent: {top_intent.intent_id} / {top_intent.label}")
        print(f"  Intent Keywords: {top_intent.keywords[:5]}...")
    else:
        print(f"  Intent: None")
    
    print()
