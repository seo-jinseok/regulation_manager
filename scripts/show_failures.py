#!/usr/bin/env python3
"""Show expected values for failing test cases."""

import json
from pathlib import Path

data_path = Path("data/config/evaluation_dataset.json")
data = json.loads(data_path.read_text())

failures = [
    "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내고 그래",
    "연구 부정행위 신고하고 싶어",
    "강의 면제 받으려면?",
    "취업 준비로 졸업 미루고 싶어",
    "징계 절차가 어떻게 돼?",
    "원격수업 규정이 어떻게 돼?",
    "학생 창업 지원받을 수 있어?",
    "취업 지원 프로그램이 있어?",
    "교수가 과제 기한 너무 짧게 줬어",
]

print("=== Failed Test Cases Expected Values ===\n")

for tc in data["test_cases"]:
    if tc["query"] in failures:
        print(f"Query: {tc['query']!r}")
        print(f"  Expected intents: {tc.get('expected_intents', [])}")
        print(f"  Expected keywords: {tc.get('expected_keywords', [])}")
        print(f"  Expected rule codes: {tc.get('expected_rule_codes', [])}")
        print(f"  Min score: {tc.get('min_relevance_score', 0.05)}")
        print()
