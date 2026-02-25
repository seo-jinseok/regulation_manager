# Research: RAG System Quality Improvement

**SPEC ID:** SPEC-RAG-QUALITY-011
**Research Date:** 2026-02-24
**Status:** Complete

## Executive Summary

This research document analyzes the root causes of the RAG quality evaluation failure where all 30 test queries received the same rejection response: "이 질문은 규정 검색이 필요하지 않습니다" (This question does not require regulation search).

**Key Findings:**
1. Self-RAG evaluator incorrectly classifies regulation queries as not needing retrieval
2. Context scores are extremely low (0.0-0.2) indicating search pipeline issues
3. LLM judge parsing errors suggest response format inconsistencies

---

## 1. Codebase Architecture Analysis

### 1.1 RAG System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Query Entry Point                        │
│  QueryHandler.process_query() → interface/query_handler.py   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Search UseCase Layer                        │
│  SearchUseCase.ask_sync() → application/search_usecase.py   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Self-RAG    │ │  ChromaDB    │ │    LLM       │
│  Evaluator   │ │  VectorStore │ │   Adapter    │
│ self_rag.py  │ │chroma_store.py│ │llm_adapter.py│
└──────────────┘ └──────────────┘ └──────────────┘
```

### 1.2 Critical Code Flow

**File:** `src/rag/application/search_usecase.py` (lines 2521-2534)

```python
# Self-RAG: Check if retrieval is even needed
retrieval_query = search_query or question
if self._enable_self_rag:
    self._ensure_self_rag()
    if self._self_rag_pipeline and not self._self_rag_pipeline.should_retrieve(
        question
    ):
        # Simple query that doesn't need retrieval (rare for regulation Q&A)
        logger.debug("Self-RAG: Skipping retrieval for simple query")
        return Answer(
            text="이 질문은 규정 검색이 필요하지 않습니다. 구체적인 규정 관련 질문을 해주세요.",
            sources=[],
            confidence=0.5,
        )
```

**Analysis:** When `should_retrieve()` returns `False`, the system immediately returns the rejection message without attempting any search.

---

## 2. Root Cause Analysis

### 2.1 Primary Issue: Self-RAG Classification

**File:** `src/rag/infrastructure/self_rag.py` (lines 36-48)

```python
RETRIEVAL_NEEDED_PROMPT = """다음 질문에 답하기 위해 외부 문서 검색이 필요한지 판단하세요.

질문: {query}

검색이 필요한 경우:
- 특정 규정, 절차, 자격 요건 등 사실적 정보가 필요한 경우
- 최신 정보나 특정 조항을 확인해야 하는 경우

검색이 불필요한 경우:
- 일반적인 인사말이나 간단한 설명 요청
- 이전 대화 내용에 대한 후속 질문 (컨텍스트가 이미 있는 경우)

답변 형식: [RETRIEVE_YES] 또는 [RETRIEVE_NO] 중 하나만 출력하세요."""
```

**Problem Analysis:**
1. The prompt is too ambiguous for regulation Q&A systems
2. LLM (GLM-4.7-flash via OpenRouter) tends to classify questions as "not needing retrieval"
3. For a regulation Q&A system, almost ALL queries should trigger retrieval
4. The prompt doesn't emphasize that regulations are domain-specific knowledge

**File:** `src/rag/infrastructure/self_rag.py` (lines 92-128)

```python
def needs_retrieval(self, query: str) -> bool:
    """
    Evaluate if retrieval is needed for this query.

    For regulation Q&A systems, retrieval is almost always needed.
    Only skip retrieval for very simple greetings/chat.
    """
    if not self._llm_client:
        return True  # Default to retrieval if no LLM

    # Quick heuristic: very short queries without question words
    # are probably not factual questions
    if len(query) < 5 and not any(
        k in query for k in ["?", "뭐", "어떻", "왜", "언제"]
    ):
        return True  # Still default to retrieval to be safe

    prompt = self.RETRIEVAL_NEEDED_PROMPT.format(query=query)

    try:
        response = self._llm_client.generate(
            system_prompt="You are a retrieval necessity evaluator.",
            user_message=prompt,
            temperature=0.0,
        )
        # Default to retrieval unless explicitly told not to
        if "[RETRIEVE_NO]" in response.upper():
            return False
        return True  # Default to retrieval
    except Exception:
        return True  # Default to retrieval on error
```

**Problem Analysis:**
1. The heuristic check (line 110-113) defaults to `True` which is correct
2. However, the LLM response parsing (line 124) allows `[RETRIEVE_NO]` to skip retrieval
3. The default LLM (GLM-4.7-flash) is returning `[RETRIEVE_NO]` for regulation queries
4. For a regulation Q&A system, this classification should default to `True` (always retrieve)

### 2.2 Secondary Issue: Empty Search Results

**File:** `src/rag/infrastructure/chroma_store.py` (lines 193-199)

```python
def search(
    self,
    query: Query,
    filter: Optional[SearchFilter] = None,
    top_k: int = 10,
) -> List[SearchResult]:
    # Check if embedding function is available
    if not hasattr(self, "_embedding_function") or self._embedding_function is None:
        logger.error(
            "Cannot search: no embedding function available. "
            "Provide an embedding_function during initialization."
        )
        return []
```

**Problem Analysis:**
1. If embedding function fails to initialize, all searches return empty
2. No data verification during initialization
3. No health check for ChromaDB collection status

**File:** `src/rag/infrastructure/chroma_store.py` (lines 66-76)

```python
# Handle embedding function
if embedding_function is None and auto_create_embedding:
    try:
        from .embedding_function import get_default_embedding_function

        self._embedding_function = get_default_embedding_function()
        logger.info("Using default embedding function (ko-sbert-sts)")
    except Exception as e:
        logger.warning(f"Failed to create default embedding function: {e}")
        self._embedding_function = None
```

**Problem Analysis:**
1. If embedding function creation fails, it's set to `None` with only a warning
2. System continues to operate but returns empty results for all queries

### 2.3 Configuration Analysis

**File:** `src/rag/config.py` (lines 241-243)

```python
enable_self_rag: bool = field(
    default_factory=lambda: os.getenv("ENABLE_SELF_RAG", "true").lower() == "true"
)
```

**Analysis:** Self-RAG is enabled by default, which means the classification issue affects all queries by default.

---

## 3. Evaluation Results Analysis

### 3.1 Test Results Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Total Queries | 30 | - | - |
| Pass Rate | 0% | - | FAIL |
| Average Score | 0.230 | 0.800 | FAIL |
| Evaluation Time | ~44 min | - | - |

### 3.2 Per-Persona Results

| Persona | Avg Score | Status |
|---------|-----------|--------|
| student-undergraduate | 0.277 | FAIL |
| student-graduate | 0.275 | FAIL |
| professor | 0.224 | FAIL |
| staff-admin | 0.203 | FAIL |
| parent | 0.201 | FAIL |
| student-international | 0.200 | FAIL |

### 3.3 Response Pattern Analysis

All 30 queries received identical response:
```
이 질문은 규정 검색이 필요하지 않습니다. 구체적인 규정 관련 질문을 해주세요.

💡 연관 질문:
  [1] ...
  [2] ...
```

This confirms the Self-RAG classification is rejecting all queries.

### 3.4 LLM Judge Parsing Errors

**Error Pattern:** "Extra data" JSON parsing errors (9 occurrences)

**Analysis:**
- LLM responses contain malformed JSON
- Fallback to rule-based evaluation occurred
- Indicates LLM response format inconsistency

---

## 4. Reference Implementation Analysis

### 4.1 Similar Pattern in Existing SPECs

**SPEC-RAG-QUALITY-001 through 010:** Various quality improvements implemented but none specifically address Self-RAG classification tuning.

### 4.2 Industry Best Practices

1. **Conservative Retrieval:** For domain-specific Q&A, always retrieve unless explicitly a greeting
2. **Graceful Degradation:** If classification fails, default to retrieval
3. **Data Verification:** Check ChromaDB collection health before accepting queries
4. **Prompt Engineering:** Use explicit prompts that emphasize domain knowledge requirements

---

## 5. Proposed Solution Architecture

### 5.1 Self-RAG Prompt Improvement

**Current Prompt Issue:**
- Too ambiguous for regulation domain
- Doesn't emphasize domain-specific knowledge requirement

**Proposed Solution:**
- Add explicit regulation domain context
- Default to retrieval for uncertain cases
- Add keyword-based pre-filtering for obvious non-retrieval cases

### 5.2 Data Verification Layer

**Proposed Additions:**
- Health check endpoint for ChromaDB
- Collection count verification
- Embedding function status check
- Automatic alerts for empty collections

### 5.3 Fallback Mechanism

**Proposed Additions:**
- If Self-RAG returns NO, verify with keyword matching
- If keywords match regulation terms, override to retrieval
- Add retry logic with different classification strategy

---

## 6. Technical Dependencies

### 6.1 Files Requiring Modification

| File | Changes |
|------|---------|
| `src/rag/infrastructure/self_rag.py` | Prompt improvement, fallback logic |
| `src/rag/infrastructure/chroma_store.py` | Health check, data verification |
| `src/rag/config.py` | Configuration options for thresholds |
| `src/rag/application/search_usecase.py` | Fallback handling |
| `scripts/evaluate_rag_quality.py` | Better error handling |

### 6.2 Test Coverage Requirements

| Component | Test Type | Coverage Target |
|-----------|-----------|-----------------|
| Self-RAG Classification | Unit | 90% |
| ChromaDB Health Check | Integration | 85% |
| End-to-End Query Flow | E2E | 80% |

---

## 7. Risk Assessment

### 7.1 High Risk Areas

1. **Self-RAG Prompt Changes:** May affect existing query classification
2. **ChromaDB Health Check:** May add latency to initialization
3. **Fallback Logic:** May increase complexity

### 7.2 Mitigation Strategies

1. **A/B Testing:** Test new prompts against current behavior
2. **Feature Flags:** Allow disabling new features if issues arise
3. **Rollback Plan:** Revert to previous behavior if metrics degrade

---

## 8. Conclusion

The root cause of the 0% pass rate is the Self-RAG evaluator incorrectly classifying all regulation queries as not needing retrieval. The LLM (GLM-4.7-flash) returns `[RETRIEVE_NO]` for queries that clearly require regulation lookup.

**Immediate Fix:** Disable Self-RAG or modify the prompt to be more conservative for regulation Q&A systems.

**Long-term Solution:** Implement a robust fallback mechanism with keyword-based verification and data health checks.

---

**Research Completed By:** MoAI System Architect Agent
**Next Step:** Create SPEC document with EARS format requirements
