#!/usr/bin/env python3
"""Test decompose_query fix for '하고 싶어' pattern."""

from src.rag.infrastructure.query_analyzer import QueryAnalyzer

qa = QueryAnalyzer()

test_queries = [
    ("휴학하고 싶어", "NOT decomposed"),
    ("연구년 신청하고 싶어", "NOT decomposed"),
    ("장학금 받고 싶어", "NOT decomposed"),
    ("교원 휴직 그리고 복직", "SHOULD decompose"),
    ("교수에게 부당한 대우를 받았어", "NOT decomposed"),
    ("졸업하고 취업 준비", "SHOULD decompose"),
    ("연구 부정행위 신고하고 싶어", "NOT decomposed"),
    ("강의 면제 받으려면?", "NOT decomposed"),
    ("교수가 과제 기한 너무 짧게 줬어", "NOT decomposed"),
    ("교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래", "NOT decomposed"),
    ("장학금과 휴학", "SHOULD decompose"),  # should still work
]

print("=== decompose_query Test ===\n")

for query, expectation in test_queries:
    result = qa.decompose_query(query)
    decomposed = len(result) > 1
    status = "✓" if (
        ("NOT" in expectation and not decomposed) or
        ("SHOULD" in expectation and decomposed)
    ) else "✗"
    print(f"{status} {query!r}")
    print(f"   Expected: {expectation}")
    print(f"   Result: {result}")
    print()
