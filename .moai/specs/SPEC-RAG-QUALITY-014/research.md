# Research: RAG Query Classification Bottleneck Analysis

## Problem Context

RAG quality evaluation shows 93.3% pass rate (28/30 queries). Two queries consistently FAIL, one query has degraded relevance. All three issues trace to query classification/routing logic, not LLM generation quality.

## Architecture Analysis

### Query Processing Pipeline
```
User Query → NFC Normalize → Structural Pattern Check → Mode Decision
  → FunctionGemma Path (if enabled) OR Traditional Path
  → Traditional Path: Audience Ambiguity Check → Self-RAG Check → HybridSearch → LLM Answer
```

### Key Files
- `src/rag/interface/query_handler.py` (lines 440-500): Main process_query routing
- `src/rag/infrastructure/query_analyzer.py` (lines 897-944): Audience detection
- `src/rag/infrastructure/self_rag.py` (lines 35-90, 150-270): Retrieval necessity evaluation
- `src/rag/application/search_usecase.py` (lines 2750-2770): Self-RAG rejection point

## Issue 1: Audience Ambiguity False Positive

### Failing Query
"교원 승진 관련 편/장/조 구체적 근거" → Score: 0.175 (FAIL)

### Root Cause Trace
```
query_handler.py:481 → query_analyzer.is_audience_ambiguous(query)
  → detect_audience_candidates(query):
    - "교원" ∈ FACULTY_KEYWORDS → matches FACULTY
    - "승진" ∈ STAFF_KEYWORDS → matches STAFF
    - Returns [FACULTY, STAFF] (len=2 → ambiguous)
  → Returns CLARIFICATION: "질문 대상을 선택해주세요"
```

### Design Flaw
`STAFF_KEYWORDS` contains "승진" (promotion), but promotion applies to both staff AND faculty. When "교원" (faculty) explicitly appears, the audience is resolved — "승진" should not trigger a second audience match.

### Current Keyword Lists (relevant)
- `FACULTY_KEYWORDS`: ["교수", "교원", "강사", "전임", "안식년", "연구년", ...]
- `STAFF_KEYWORDS`: ["직원", "행정", "사무", "참사", "주사", **"승진"**, "전보", "직원인데"]
- `AMBIGUOUS_AUDIENCE_KEYWORDS`: ["징계", "처분", "위반", "제재", "윤리", "고충"]
- `FACULTY_CONTEXT_KEYWORDS`: ["강의", "연구", "논문", **"승진"**, "교수", ...]

Note: "승진" appears in BOTH `STAFF_KEYWORDS` (primary) and `FACULTY_CONTEXT_KEYWORDS` (secondary). The primary match takes precedence in the algorithm.

### Fix: Move "승진" from STAFF_KEYWORDS to AMBIGUOUS_AUDIENCE_KEYWORDS

Verification:
- "교원 승진" → "교원" matches FACULTY only (승진 no longer in STAFF) → [FACULTY] ✓
- "직원 승진" → "직원" matches STAFF only → [STAFF] ✓
- "승진 요건" → no primary match, "승진" hits AMBIGUOUS → clarification ✓ (correct)

## Issue 2: Self-RAG False Negative (Missing Keywords)

### Failing Query
"급여 관련 서식 양식 알려주세요" → Score: 0.515 (FAIL)

### Root Cause Trace
```
search_usecase.py:2753 → self_rag_pipeline.should_retrieve(question)
  → needs_retrieval(query):
    → _has_regulation_keywords("급여 관련 서식 양식 알려주세요"):
      - Checks REGULATION_KEYWORDS list
      - "급여" NOT found, "서식" NOT found, "양식" NOT found
      - Returns False
    → LLM evaluates: returns [RETRIEVE_NO]
    → Override check: _has_regulation_keywords → False
    → _has_university_topic_words → False (English-only)
    → Returns False → REJECTION MESSAGE
```

### Missing Keywords in REGULATION_KEYWORDS (self_rag.py:35-70)
The keyword list was designed with student-centric focus. Missing staff/admin terms:
- "급여" (salary) - present in no keyword list
- "서식" (form/template)
- "양식" (form)
- "수당" (allowance)
- "보수" (compensation)
- "복무" (service/attendance)
- "근무" (work) - partially covered via synonyms but NOT in REGULATION_KEYWORDS
- "출장" (business trip)
- "휴가" (leave/vacation)
- "연수" (training)
- "퇴직" (retirement)
- "인사" (HR)

### Fix: Add missing terms to REGULATION_KEYWORDS

This is safe because the system design principle states: "false positives (unnecessary search) are better than false negatives (missed search)"

## Issue 3: Retrieval Relevance Gap (Lower Priority)

### Query
"복무 처리 기한이 언제까지인가요?" → Score: 0.868 (PASS but low quality)

### Root Cause
ChromaDB contains 44 documents mentioning "복무" but 0 documents with both "복무" and "기한". The query has no exact answer in the database. The retriever returns "재수강규정" as best match (likely due to "기한" and "처리" term overlap).

### Key Finding
This is primarily a **data coverage gap**, not a code bug. The LLM correctly states "관련 내용을 찾을 수 없습니다" — this is proper behavior (hallucination prevention).

### Potential Improvements
1. Verify synonym expansion ("복무" → "근무", "근태") is actually used in BM25 pipeline
2. When all retrieved docs have low relevance scores, omit the misleading citation section
3. Lower priority since query already passes

## Test Coverage Impact

Existing tests:
- `tests/rag/unit/infrastructure/test_query_analyzer.py:284`: Tests `is_audience_ambiguous("징계") == True`
- `tests/rag/unit/infrastructure/test_query_analyzer.py:288`: Tests `is_audience_ambiguous("학생 휴학 절차") == False`
- `tests/rag/unit/infrastructure/test_query_analyzer_coverage.py:657`: Tests `detect_audience_candidates("직원 승진")` returns STAFF

**Critical**: The test at line 657 expects "직원 승진" → STAFF. After moving "승진" from STAFF_KEYWORDS, "직원" still matches STAFF_KEYWORDS, so this test should still pass. Need to verify.
