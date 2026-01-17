#!/usr/bin/env python3
"""Test script for RAG quality improvements."""

from src.rag.infrastructure.query_analyzer import QueryAnalyzer
from src.rag.interface.query_handler import detect_deletion_warning

def test_decompose_query():
    """Test query decomposition."""
    qa = QueryAnalyzer()
    
    test_queries = [
        "장학금 받으면서 휴학",
        "교원 휴직 그리고 복직",
        "단순 쿼리",
        "학교 그만두고 싶어",
        "휴학하고 장학금 어떻게 되나요",
    ]
    
    print("=== decompose_query 테스트 ===")
    for q in test_queries:
        result = qa.decompose_query(q)
        print(f"  {q!r} → {result}")
    print()


def test_deletion_warning():
    """Test deletion warning detection."""
    test_texts = [
        "제5조(삭제 2023.3.1)",
        "본 조는 삭제되었습니다.",
        "(삭제)",
        "[삭제]",
        "제1항은 폐지(2024.6.15)되었습니다.",
        "일반 조항 내용입니다.",
        "삭제(2020)",
        "본 항은 삭제함",
    ]
    
    print("=== 삭제 조항 감지 테스트 ===")
    for t in test_texts:
        result = detect_deletion_warning(t)
        status = "⚠️ 감지" if result else "✓ 정상"
        print(f"  [{status}] {t!r}")
        if result:
            print(f"      → {result}")
    print()


def test_intent_patterns():
    """Test intent pattern matching for new patterns."""
    qa = QueryAnalyzer()
    
    test_queries = [
        "부당대우 당했어요",
        "교수님이 갑질해요", 
        "학자금 대출 받고 싶어요",
        "장학금 받으면서 휴학하면 어떻게 되나요",
    ]
    
    print("=== 인텐트 매칭 테스트 ===")
    for q in test_queries:
        matches = qa._match_intents(q)
        expanded = qa.expand_query(q)
        print(f"  {q!r}")
        print(f"    매칭: {[m.intent_id for m in matches]}")
        print(f"    확장: {expanded}")
    print()


if __name__ == "__main__":
    test_decompose_query()
    test_deletion_warning()
    test_intent_patterns()
