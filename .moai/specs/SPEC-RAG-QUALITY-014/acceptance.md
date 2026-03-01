---
spec: SPEC-RAG-QUALITY-014
type: acceptance-criteria
---

# Acceptance Criteria: SPEC-RAG-QUALITY-014

## AC-001: Self-RAG Keyword Coverage (EARS-U-001)

### Scenario 1: Staff salary/form query triggers retrieval

**Given** REGULATION_KEYWORDS includes "급여", "서식", "양식"
**When** user submits "급여 관련 서식 양식 알려주세요"
**Then** `_has_regulation_keywords()` returns True
**And** Self-RAG does NOT reject the query
**And** the system performs hybrid search and returns regulation content
**And** evaluation score ≥ 0.7

### Scenario 2: All new staff/admin keywords are recognized

**Given** REGULATION_KEYWORDS includes all 15 added terms
**When** user submits queries containing each new keyword individually
**Then** `_has_regulation_keywords()` returns True for each keyword:
  - "급여", "보수", "수당", "서식", "양식"
  - "복무", "근무", "출장", "휴가", "연수"
  - "퇴직", "인사", "복리후생", "보험", "겸직"

### Scenario 3: Existing keywords still work

**Given** REGULATION_KEYWORDS still includes all original terms
**When** user submits "등록금 납부 기간" (existing test query)
**Then** `_has_regulation_keywords()` returns True (no regression)

## AC-002: Audience Ambiguity Fix (EARS-U-002)

### Scenario 4: Faculty promotion resolves to faculty

**Given** "승진" is in AMBIGUOUS_AUDIENCE_KEYWORDS (not STAFF_KEYWORDS)
**When** user submits "교원 승진 관련 편/장/조 구체적 근거"
**Then** `detect_audience_candidates()` returns [FACULTY] only
**And** `is_audience_ambiguous()` returns False
**And** the system performs search instead of asking clarification
**And** evaluation score ≥ 0.7

### Scenario 5: Staff promotion resolves to staff

**Given** "승진" is in AMBIGUOUS_AUDIENCE_KEYWORDS (not STAFF_KEYWORDS)
**When** user submits "직원 승진 요건"
**Then** `detect_audience_candidates()` returns [STAFF] (via "직원" keyword)
**And** `is_audience_ambiguous()` returns False

### Scenario 6: Bare promotion triggers clarification

**Given** "승진" is in AMBIGUOUS_AUDIENCE_KEYWORDS
**When** user submits "승진 요건이 어떻게 되나요" (no explicit audience)
**Then** `detect_audience_candidates()` returns [] or triggers ambiguous path
**And** `is_audience_ambiguous()` returns True

### Scenario 7: Existing ambiguous keywords unchanged

**Given** AMBIGUOUS_AUDIENCE_KEYWORDS still includes "징계", "처분", "위반"
**When** user submits "징계 관련 규정"
**Then** `is_audience_ambiguous()` returns True (no regression)

## AC-003: Regression Prevention (EARS-E-003)

### Scenario 8: Full test suite passes

**Given** Task 1 and Task 2 changes are applied
**When** running `uv run pytest tests/ -v`
**Then** all existing tests pass (0 failures, 0 errors)

### Scenario 9: Existing passing queries maintain scores

**Given** Task 1 and Task 2 changes are applied
**When** running RAG quality evaluation
**Then** all 28 previously passing queries still pass (score ≥ 0.7)
**And** overall pass rate ≥ 96.7% (was 93.3%)

## Exit Criteria

- [ ] All unit tests pass (existing + new)
- [ ] RAG evaluation pass rate ≥ 96.7%
- [ ] No regression in any previously passing query
- [ ] Code changes limited to keyword lists only (minimal blast radius)
