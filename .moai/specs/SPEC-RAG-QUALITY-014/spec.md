---
id: SPEC-RAG-QUALITY-014
version: 1.0.0
status: implemented
created: 2026-03-01
implemented: 2026-03-01
author: MoAI
priority: high
title: "RAG Query Classification Bottleneck Fix"
tags: [self-rag, query-analyzer, keyword-expansion, audience-detection]
affected_files:
  - src/rag/infrastructure/self_rag.py
  - src/rag/infrastructure/query_analyzer.py
  - tests/rag/unit/infrastructure/test_self_rag.py
  - tests/rag/unit/infrastructure/test_query_analyzer.py
---

# SPEC-RAG-QUALITY-014: RAG Query Classification Bottleneck Fix

## Background

RAG quality evaluation shows 93.3% pass rate (28/30). Two queries consistently FAIL due to
query classification errors in the pre-search routing layer. Both failures are keyword-list
gaps — deterministic bugs with precise fixes.

### Failing Queries

| Query | Score | Root Cause |
|-------|-------|------------|
| "교원 승진 관련 편/장/조 구체적 근거" | 0.175 | Audience ambiguity false positive |
| "급여 관련 서식 양식 알려주세요" | 0.515 | Self-RAG retrieval rejection |

### Target

93.3% → ≥96.7% pass rate (both FAIL → PASS)

## Requirements (EARS Format)

### EARS-U-001: Self-RAG Keyword Coverage

**When** a user query contains staff/admin regulation terms ("급여", "서식", "양식", "수당",
"보수", "복무", "근무", "출장", "휴가", "연수", "퇴직", "인사"),
**the system shall** classify the query as requiring retrieval via `_has_regulation_keywords()`.

**Rationale:** Current `REGULATION_KEYWORDS` in `self_rag.py:36-70` is missing 12+ staff/admin
terms. This causes `_has_regulation_keywords()` to return False, triggering LLM fallback that
incorrectly rejects legitimate regulation queries.

### EARS-U-002: Audience Ambiguity Resolution for "승진"

**When** a user query contains both an explicit audience keyword (e.g., "교원") and "승진",
**the system shall** resolve to the explicit audience without triggering ambiguity clarification.

**Rationale:** "승진" (promotion) in `STAFF_KEYWORDS` causes false matches. Query "교원 승진"
matches FACULTY via "교원" AND STAFF via "승진" → ambiguity triggered → clarification returned
instead of search. "승진" should be in `AMBIGUOUS_AUDIENCE_KEYWORDS` since it applies to both
faculty and staff.

### EARS-E-003: Regression Prevention

**Where** the query classification subsystem is modified,
**the system shall** maintain all existing passing queries at their current scores (±0.05 tolerance).

## Technical Approach

### Task 1 (P1): Expand REGULATION_KEYWORDS

**File:** `src/rag/infrastructure/self_rag.py` (lines 36-70)

Add missing staff/admin terms to `REGULATION_KEYWORDS`:
```python
# Staff/Admin topics (Korean) - SPEC-RAG-QUALITY-014 EARS-U-001
"급여", "보수", "수당", "서식", "양식",
"복무", "근무", "출장", "휴가", "연수",
"퇴직", "인사", "복리후생", "보험", "겸직",
```

**Risk:** Low. Adding keywords only increases retrieval rate (false positives are preferable to
false negatives in this system). No existing passing query will be affected.

### Task 2 (P2): Fix Audience Keyword Classification

**File:** `src/rag/infrastructure/query_analyzer.py`

1. Remove "승진" from `STAFF_KEYWORDS` (line ~222)
2. Add "승진" to `AMBIGUOUS_AUDIENCE_KEYWORDS` (line ~244)

**Verification Matrix:**
| Query | Before | After | Correct? |
|-------|--------|-------|----------|
| "교원 승진" | [FACULTY, STAFF] → ambiguous | [FACULTY] → resolved | Yes |
| "직원 승진" | [STAFF] → resolved | [STAFF] (via "직원") → resolved | Yes |
| "승진 요건" | [STAFF] → resolved (wrong!) | clarification → ambiguous | Yes |
| "교원 징계" | [FACULTY] + ambiguous | unchanged | Yes |

**Risk:** Medium. Must verify existing test `test_query_analyzer_coverage.py:657` which tests
`detect_audience_candidates("직원 승진")` → expects STAFF. After fix, "직원" alone still
matches STAFF_KEYWORDS, so test should still pass.

### Task 3 (P3, Optional): Low-Relevance Retrieval Fallback

Deferred to future SPEC. Issue 3 ("복무 처리 기한") is a data coverage gap (0 documents with
both "복무" and "기한" in ChromaDB), not a code bug. Score 0.868 already passes.

## Dependencies

- No external dependencies
- No breaking API changes
- Both fixes are additive keyword changes

## Methodology

TDD (test-first) per `quality.yaml` `development_mode: hybrid` — new keyword tests are
new features, so TDD applies.
