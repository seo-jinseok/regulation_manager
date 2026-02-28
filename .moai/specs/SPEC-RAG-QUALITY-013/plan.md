# SPEC-RAG-QUALITY-013 Implementation Plan

## Phase 1: Quick Wins (Expected: +30% pass rate)

### Step 1.1: Search Relevance Hardening (REQ-001, REQ-002, REQ-003)

**Target**: Eliminate irrelevant documents from LLM context

**Tasks**:
1. Raise `min_relevance_score` from 0.25 to 0.40 in `search_usecase.py:ask()` (~line 2730)
2. Modify "all below threshold" branch: return no-info response for score < 0.40
3. Add fallback: if 0 results at 0.40, retry with 0.30 threshold and max 3 results
4. Implement `_filter_topic_relevance()`:
   - Extract topic keywords from query
   - Extract topic from document title/parent_path
   - Exclude documents with zero topical overlap
5. Write characterization tests for current behavior
6. Write unit tests for new filtering logic
7. Run evaluation: verify improved context_relevance

**Files**:
- `src/rag/application/search_usecase.py` (modify `ask()`, add `_filter_topic_relevance()`)
- `tests/rag/unit/application/test_search_relevance_filtering.py` (new)

### Step 1.2: CoT Stripping Enhancement (REQ-007)

**Target**: Strip glm-4.7-flash analysis format from output

**Tasks**:
1. Add regex patterns to `_COT_PATTERNS` for:
   - `r"^\*\s+\*\*(Role|Input Question|Provided Context|Question|Context|Constraint|User Persona)[:\s]?\*\*.*$"`
   - `r"^\*\s+\[\d+\]\s+reg-\d+:.*$"` (document listing)
   - `r"^\*\s+\*\*User:\*\*.*$"` (user description)
2. Write tests with actual model output samples
3. Verify no false positives on good Korean answers

**Files**:
- `src/rag/application/search_usecase.py` (modify `_COT_PATTERNS`)
- `tests/rag/unit/application/test_cot_stripping.py` (add test cases)

### Step 1.3: Citation Auto-Enrichment (REQ-009)

**Target**: Convert reg-XXXX codes to 「규정명」 format

**Tasks**:
1. Build reg-code → regulation name mapping from chunk metadata
2. Post-process answer to replace `reg-XXXX` with `「규정명」`
3. Enhance `_enhance_answer_citations()` with this mapping
4. Write tests

**Files**:
- `src/rag/application/search_usecase.py` (enhance `_enhance_answer_citations()`)
- `tests/rag/unit/application/test_citation_enhancement.py` (add test cases)

### Phase 1 Validation
- Run `uv run pytest` (all tests pass)
- Run `uv run python run_rag_quality_eval.py --quick --summary`
- Expected: Pass rate ≥ 45% (14/30)

---

## Phase 2: Core Improvements (Expected: +20% pass rate)

### Step 2.1: LLM Output Normalization (REQ-004, REQ-008)

**Target**: Extract Korean answer from English analysis output

**Tasks**:
1. Create `OutputNormalizer` class in `src/rag/domain/llm/output_normalizer.py`
2. Detection logic: identify analysis-format responses
3. Extraction logic: find Korean content, conclusion sections
4. Integration: call after `_strip_cot_from_answer()` in pipeline
5. Write comprehensive tests

**Files**:
- `src/rag/domain/llm/output_normalizer.py` (new)
- `src/rag/application/search_usecase.py` (integrate normalization)
- `tests/rag/unit/domain/llm/test_output_normalizer.py` (new)

### Step 2.2: Compact Prompt (REQ-005)

**Target**: Create simplified prompt for small models

**Tasks**:
1. Create `regulation_qa_compact` in `data/config/prompts.json`
2. Core instructions only: context boundary, citation format, Korean output
3. Add model-size detection in answer generation
4. Select compact prompt for small models

**Files**:
- `data/config/prompts.json` (add compact prompt)
- `src/rag/application/search_usecase.py` (prompt selection logic)

### Step 2.3: Analysis-Only Retry (REQ-006)

**Target**: Retry when response is analysis-only

**Tasks**:
1. Detect analysis-only response (no Korean paragraph)
2. Retry with focused re-prompt: "위 분석을 바탕으로 한국어로 답변해주세요"
3. Maximum 1 retry per query
4. Write tests

**Files**:
- `src/rag/application/search_usecase.py` (modify answer generation)

### Step 2.4: Citation Article Numbers (REQ-010, REQ-011)

**Target**: Auto-append article numbers and ensure minimum citation

**Tasks**:
1. Extract article numbers from chunk metadata
2. Append to citations missing article references
3. If answer has zero citations, inject from source metadata
4. Write tests

**Files**:
- `src/rag/application/search_usecase.py` (enhance `_verify_citations()`)

### Phase 2 Validation
- Run full test suite
- Run evaluation
- Expected cumulative: Pass rate ≥ 65% (20/30)

---

## Phase 3: Targeted Optimization (Expected: +10% pass rate)

### Step 3.1: Multi-Document Synthesis (REQ-012, REQ-013)

**Target**: Ensure answers use all relevant retrieved documents

**Tasks**:
1. Add instruction to prompt: "검색된 모든 관련 규정을 종합하여 답변하세요"
2. Add procedural completeness instruction for step-type queries
3. Verify with evaluation

**Files**:
- `data/config/prompts.json` (modify prompt)

### Step 3.2: Per-Persona Prompts (REQ-014, REQ-015)

**Target**: Persona-specific prompt additions

**Tasks**:
1. Add persona-specific addendum in prompt config
2. staff-admin: "행정 절차, 승인 과정, 서식 참조를 강조하세요"
3. international: "핵심 용어를 한국어와 영어로 병기하세요"
4. Integrate persona detection → prompt selection
5. Test with persona-specific evaluation queries

**Files**:
- `data/config/prompts.json` (add persona addendums)
- `src/rag/application/search_usecase.py` (persona-aware prompt selection)

### Phase 3 Validation
- Run full test suite
- Run evaluation
- Expected cumulative: Pass rate ≥ 75-80% (23-24/30)

---

## Final Validation

1. `uv run pytest` — all existing tests pass + new tests
2. `uv run python run_rag_quality_eval.py --quick --summary` — pass rate ≥ 80%
3. Per-persona check: all groups ≥ 60% pass rate
4. Review failure cases for remaining missed queries

---

## DDD Cycle for Each Step

```
ANALYZE: Read existing code, identify dependencies
PRESERVE: Write characterization tests capturing current behavior
IMPROVE: Implement changes, verify tests pass after each change
```
