#!/usr/bin/env python3
"""Test decomposition pattern matching."""

import re

patterns = [
    r"하고\s*싶",
    r"하면\s*싶",
    r"하면서\s*싶",
    r"받고\s*싶",
    r"알고\s*싶",
    r"가고\s*싶",
    r"싶고\s*",
]

queries = [
    "연구 부정행위 신고하고 싶어",
    "휴학하고 싶어",
    "신고하고 싶어",
    "강의 면제 받으려면?",  # 패턴 매칭 안 됨
]

print("=== Pattern Matching Test ===\n")

for q in queries:
    matched = None
    for p in patterns:
        if re.search(p, q):
            matched = p
            break
    print(f"{q!r}")
    print(f"  Matched: {matched}\n")
