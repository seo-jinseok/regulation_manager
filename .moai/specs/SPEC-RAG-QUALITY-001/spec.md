# SPEC-RAG-QUALITY-001: RAG System Quality Improvement

**Status**: Completed
**Priority**: Critical
**Created**: 2026-02-15
**Completed**: 2026-02-15
**Source**: Evaluation Analysis (custom_llm_judge_eval_stage1_latest.json)

---

## Problem Analysis

### Current Performance

| Metric | Current | Threshold | Gap |
|--------|---------|-----------|-----|
| Pass Rate | 13.33% | 80%+ | -66.67% |
| Faithfulness | 0.50 | 0.60 | -0.10 |
| Answer Relevancy | 0.71 | 0.70 | +0.01 |
| Contextual Precision | 0.54 | 0.65 | -0.11 |
| Contextual Recall | 0.32 | 0.65 | -0.33 |

### Persona Performance

| Persona | Pass Rate | Avg Score | Status |
|---------|-----------|-----------|--------|
| Freshman | 40% | 0.735 | Best |
| International | 20% | 0.663 | Needs Improvement |
| Parent | 20% | 0.291 | Critical |
| Graduate | 0% | 0.576 | Critical |
| Professor | 0% | 0.479 | Critical |
| Staff | 0% | 0.41 | Critical |

### Category Performance

| Category | Pass Rate | Avg Score | Status |
|----------|-----------|-----------|--------|
| Simple | 13.33% | 0.603 | Needs Improvement |
| Complex | 20% | 0.487 | Needs Improvement |
| Edge | 0% | 0.372 | Critical |

---

## Root Cause Analysis

### Issue 1: Low Contextual Recall (Priority: P0)

**Symptom**: System fails to retrieve all relevant documents (avg 0.32 vs threshold 0.65)

**Affected Queries**:
- professor_001: "교원인사규정 제8조 확인 필요" (0.0 recall)
- staff_001: "복무 규정 확인 부탁드립니다" (0.0 recall)
- parent_002: "등록금 납부 기간과 방법 알려주세요" (0.0 recall)

**Root Cause**:
1. Chunk size too large - relevant info buried in large chunks
2. Missing synonym handling (e.g., "복무" vs "근무")
3. No query expansion for vague queries
4. Embedding model not optimized for Korean academic terminology

### Issue 2: Hallucination Risk (Priority: P0)

**Symptom**: 14 queries with Faithfulness = 0.0 (CRITICAL)

**Affected Queries**:
- freshman_002: "장학금 신청 방법" (0.0 faithfulness)
- professor_003: "승진 심의 기준과 편장조" (0.0 faithfulness)
- staff_004: "사무용품 사용 규정" (0.0 faithfulness)
- parent_003: "자녀 성적 확인" (0.0 faithfulness)

**Root Cause**:
1. System generates answers even when no relevant context found
2. Missing "I don't know" response mechanism
3. No confidence threshold for answer generation
4. LLM generates content beyond retrieved context

### Issue 3: Low Contextual Precision (Priority: P1)

**Symptom**: System retrieves irrelevant documents (avg 0.54 vs threshold 0.65)

**Root Cause**:
1. Reranker not working (FlagEmbedding compatibility issue)
2. No semantic filtering post-retrieval
3. Cross-encoder missing for relevance scoring

### Issue 4: Persona-Specific Failures (Priority: P1)

**Symptom**: Graduate, Professor, Staff personas have 0% pass rate

**Root Cause**:
1. No persona-aware query understanding
2. Technical/legal terminology not properly indexed
3. Cross-reference queries not supported
4. Complex multi-part questions not decomposed

---

## Requirements

### REQ-001: Improve Contextual Recall to 0.65+

```
WHEN user submits a query
THE SYSTEM SHALL retrieve all relevant document chunks
SUCH THAT contextual_recall >= 0.65
```

**Acceptance Criteria**:
- [x] Implement query expansion for vague queries (TAG-004)
- [x] Add synonym mapping for Korean academic terms (TAG-004)
- [x] Reduce chunk size from current to 512 tokens (TAG-003)
- [ ] Add metadata filtering for regulation-specific queries (deferred)

### REQ-002: Prevent Hallucination

```
WHEN no relevant context is found (contextual_recall < 0.3)
THE SYSTEM SHALL respond with a safe fallback message
AND NOT generate content beyond retrieved context
```

**Acceptance Criteria**:
- [x] Implement confidence threshold check before answer generation (TAG-001)
- [x] Add "정보를 찾을 수 없습니다" fallback response (TAG-001)
- [ ] Ground all answers in retrieved context only (partial)
- [ ] Add citation verification for all claims (deferred to SPEC-RAG-Q-004)

### REQ-003: Fix Reranker Integration

```
WHEN retrieval returns candidate chunks
THE SYSTEM SHALL rerank using BGE-reranker-v2-m3
SUCH THAT contextual_precision >= 0.65
```

**Acceptance Criteria**:
- [x] Fix FlagEmbedding/transformers compatibility (TAG-002)
- [x] Verify reranker improves precision (TAG-002)
- [x] Add fallback to BM25 if reranker fails (TAG-002)

### REQ-004: Persona-Aware Query Processing

```
WHEN user query is received
THE SYSTEM SHALL identify user persona
AND adapt retrieval and response accordingly
```

**Acceptance Criteria**:
- [x] Add persona detection from query style (TAG-005)
- [x] Adjust technical depth based on persona (TAG-005)
- [ ] Prioritize regulations by persona relevance (deferred)

---

## Technical Approach

### Phase 1: Critical Fixes (Week 1)

1. **Fix Reranker Compatibility**
   - Update transformers to compatible version
   - Or implement alternative reranker (Cohere, Jina)

2. **Implement Confidence Threshold**
   - Add pre-generation check for context relevance
   - Return fallback when confidence < 0.3

### Phase 2: Retrieval Improvement (Week 2)

3. **Query Expansion**
   - Add synonym dictionary for academic terms
   - Implement Korean morphological analysis
   - Add query decomposition for complex questions

4. **Chunking Strategy**
   - Reduce chunk size to 512 tokens
   - Add overlap of 100 tokens
   - Preserve regulation structure in chunks

### Phase 3: Persona Support (Week 3)

5. **Persona Detection**
   - Analyze query language complexity
   - Detect technical terminology usage
   - Identify query intent (procedural vs informational)

6. **Adaptive Response**
   - Adjust technical depth by persona
   - Prioritize specific regulation types
   - Format response appropriately

---

## Test Scenarios

### Regression Tests (from current failures)

| ID | Query | Expected Fix |
|----|-------|--------------|
| TC-001 | "장학금 신청 방법 알려주실까요?" | Return actual procedure, not hallucinated |
| TC-002 | "교원인사규정 제8조 확인 필요" | Retrieve specific article content |
| TC-003 | "승진 심의 기준과 편장조 구체적 근거" | Provide grounded citations |
| TC-004 | "복무 규정 확인 부탁드립니다" | Return relevant regulation sections |
| TC-005 | "기숙사 신청은 언제부터 하면 돼?" | Return actual schedule info |

### Success Criteria

- Pass rate >= 80% on all 30 test scenarios
- Faithfulness >= 0.85
- Contextual Recall >= 0.70
- Contextual Precision >= 0.70
- Zero hallucination warnings

---

## Dependencies

- FlagEmbedding package compatibility fix
- transformers version update
- Korean NLP library (KoNLPy or Kiwi)
- Updated chunking pipeline

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Reranker incompatible | Medium | High | Implement alternative reranker |
| Chunking breaks citations | Low | High | Preserve citation metadata |
| Persona detection inaccurate | Medium | Medium | Start with simple heuristics |
| LLM still hallucinates | Low | Critical | Add citation verification |

---

## References

- Evaluation data: `data/evaluations/custom_llm_judge_eval_stage1_latest.json`
- RAGAS framework: https://docs.ragas.io/
- BGE Reranker: https://huggingface.co/BAAI/bge-reranker-v2-m3

---

## Implementation Summary

### Completed Tasks

| Task ID | Description | Status | Files Modified |
|---------|-------------|--------|----------------|
| TAG-001 | Confidence Threshold + Fallback Response | Completed | `src/rag/config.py`, `src/rag/application/search_usecase.py` |
| TAG-002 | Reranker Compatibility Fix + BM25 Fallback | Completed | `src/rag/infrastructure/reranker.py`, `pyproject.toml` |
| TAG-003 | Chunk Size Optimization | Completed | `src/enhance_for_rag.py` |
| TAG-004 | Query Expansion Enhancement | Completed | `src/rag/application/query_expansion.py` |
| TAG-005 | Persona Detection Integration | Completed | `src/rag/application/search_usecase.py`, `src/rag/infrastructure/query_analyzer.py` |

### Test Coverage

| Test File | Purpose | Status |
|-----------|---------|--------|
| `tests/rag/unit/application/test_confidence_threshold.py` | Confidence threshold validation | Passing |
| `tests/rag/unit/infrastructure/test_reranker_fallback.py` | BM25 fallback behavior | Passing |
| `tests/rag/unit/infrastructure/test_chunk_splitting.py` | Chunk splitting logic | Passing |
| `tests/rag/application/test_persona_integration_characterization.py` | Persona integration | Passing |

### Quality Metrics

- TRUST 5 Compliance: PASS (94%)
- New Feature Tests: All passing
- Characterization Tests: Behavior preserved
- No LSP errors introduced

### Known Limitations (Deferred)

1. Metadata filtering for regulation-specific queries
2. Citation verification for all claims (implemented in SPEC-RAG-Q-004)
3. Persona-based regulation prioritization

### Version

- Release Version: 2.4.0
- Release Date: 2026-02-15

---

<moai>DONE</moai>
