---
spec: SPEC-RAG-QUALITY-014
phase: run
methodology: TDD
---

# Implementation Plan: SPEC-RAG-QUALITY-014

## Execution Order

Task 1 → Task 2 → Regression Test → Quality Evaluation

Tasks 1 and 2 are independent (different files) but executed sequentially for controlled
validation. Each task follows RED → GREEN → VERIFY cycle.

## Task 1: Expand REGULATION_KEYWORDS (P1)

### RED Phase — Write Failing Test

**File:** `tests/rag/unit/infrastructure/test_self_rag.py`

Add parametrized test:
```python
@pytest.mark.parametrize("query", [
    "급여 관련 서식 양식 알려주세요",
    "출장 규정이 어떻게 되나요",
    "복무 관련 규정 알려주세요",
    "퇴직 절차가 어떻게 되나요",
    "연수 승인 요건",
    "수당 지급 기준",
    "인사 관련 규정",
    "겸직 허가 절차",
])
def test_staff_admin_queries_need_retrieval(self, query):
    """SPEC-RAG-QUALITY-014 EARS-U-001: Staff/admin queries must trigger retrieval."""
    evaluator = SelfRAGEvaluator(llm_client=None)
    assert evaluator._has_regulation_keywords(query) is True
```

**Expected:** All 8 tests FAIL (RED)

### GREEN Phase — Add Keywords

**File:** `src/rag/infrastructure/self_rag.py` (lines 47-50, after "정년" line)

Insert:
```python
    # Staff/Admin topics (Korean) - SPEC-RAG-QUALITY-014 EARS-U-001
    "급여", "보수", "수당", "서식", "양식",
    "복무", "근무", "출장", "휴가", "연수",
    "퇴직", "인사", "복리후생", "보험", "겸직",
```

**Expected:** All 8 tests PASS (GREEN)

### VERIFY Phase

```bash
uv run pytest tests/rag/unit/infrastructure/test_self_rag.py -v
```

All existing + new tests must pass.

## Task 2: Fix Audience Keyword Classification (P2)

### RED Phase — Write Failing Test

**File:** `tests/rag/unit/infrastructure/test_query_analyzer.py`

Add test:
```python
def test_faculty_promotion_not_ambiguous(self):
    """SPEC-RAG-QUALITY-014 EARS-U-002: '교원 승진' must resolve to FACULTY, not ambiguous."""
    result = self.analyzer.is_audience_ambiguous("교원 승진 관련 근거")
    assert result is False

def test_bare_promotion_is_ambiguous(self):
    """SPEC-RAG-QUALITY-014 EARS-U-002: '승진' alone should be ambiguous."""
    result = self.analyzer.is_audience_ambiguous("승진 요건이 어떻게 되나요")
    assert result is True
```

**Expected:** First test FAILS (currently returns True), second test FAILS (currently returns False → STAFF only)

### GREEN Phase — Move "승진" keyword

**File:** `src/rag/infrastructure/query_analyzer.py`

1. Remove "승진" from `STAFF_KEYWORDS` (line ~222):
   ```python
   # Before
   STAFF_KEYWORDS = ["직원", "행정", "사무", "참사", "주사", "승진", "전보", "직원인데"]
   # After
   STAFF_KEYWORDS = ["직원", "행정", "사무", "참사", "주사", "전보", "직원인데"]
   ```

2. Add "승진" to `AMBIGUOUS_AUDIENCE_KEYWORDS` (line ~244):
   ```python
   # Before
   AMBIGUOUS_AUDIENCE_KEYWORDS = ["징계", "처분", "위반", "제재", "윤리", "고충"]
   # After
   AMBIGUOUS_AUDIENCE_KEYWORDS = ["징계", "처분", "위반", "제재", "윤리", "고충", "승진"]
   ```

**Expected:** Both new tests PASS (GREEN)

### VERIFY Phase — Regression Check

```bash
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer.py -v
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer_coverage.py -v
```

**Critical regression test:** `test_query_analyzer_coverage.py:657` tests
`detect_audience_candidates("직원 승진")` → expects STAFF.
After fix: "직원" still matches STAFF_KEYWORDS → [STAFF] → test passes.

## Regression Verification

After both tasks, run full test suite:
```bash
uv run pytest tests/ -v --tb=short
```

## Quality Evaluation

Re-run RAG quality evaluation to confirm:
```bash
uv run python run_rag_quality_eval.py
```

**Expected Results:**
- "교원 승진 관련 편/장/조 구체적 근거": 0.175 → ≥0.7 (PASS)
- "급여 관련 서식 양식 알려주세요": 0.515 → ≥0.7 (PASS)
- Overall: 93.3% → ≥96.7%

## Rollback Plan

Both changes are keyword-list modifications. Revert via `git checkout` on the two files:
```bash
git checkout src/rag/infrastructure/self_rag.py
git checkout src/rag/infrastructure/query_analyzer.py
```
