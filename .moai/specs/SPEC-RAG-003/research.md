# SPEC-RAG-003 Research Document

## Deep Codebase Analysis for RAG Quality Improvement

### 1. Architecture Analysis

#### Query Processing Pipeline

```
User Query → NFC Normalize → QueryHandler.ask()
                                    ↓
                           force_mode="ask" → SearchUseCase.ask()
                                    ↓
                           Self-RAG Check (self_rag.py)
                              ↓ YES           ↓ NO
                         HybridSearch     REJECTION MESSAGE
                              ↓
                         Corrective RAG
                              ↓
                         BGE Reranker
                              ↓
                         LLM Answer Gen
```

#### Key File Locations

| Component | File | Lines |
|-----------|------|-------|
| Query Router | `src/rag/interface/query_handler.py` | - |
| Self-RAG Evaluator | `src/rag/infrastructure/self_rag.py` | 35-170 |
| Self-RAG Keywords | `src/rag/infrastructure/self_rag.py` | 35-51 |
| Self-RAG LLM Prompt | `src/rag/infrastructure/self_rag.py` | 75-94 |
| Rejection Message | `src/rag/application/search_usecase.py` | 2531 |
| Hybrid Search | `src/rag/infrastructure/hybrid_search.py` | 677+ |
| Answer Gen Prompt | `src/rag/application/search_usecase.py` | 364-520 |
| Corrective RAG | `src/rag/application/search_usecase.py` | 2754-2821 |
| Retrieval Evaluator | `src/rag/infrastructure/retrieval_evaluator.py` | - |

### 2. Root Cause Analysis

#### 2.1 Self-RAG Bilingual Gap (CRITICAL)

**REGULATION_KEYWORDS** (self_rag.py:35-51):
- All 50+ keywords are Korean only
- No English equivalents: "regulation", "tuition", "scholarship", "leave", etc.
- English queries NEVER match keyword pre-filter

**RETRIEVAL_NEEDED_PROMPT** (self_rag.py:75-94):
- Entirely in Korean with Korean-only examples
- "어떻게", "언제", "누가" → Korean question words only
- No English equivalents: "How", "When", "What"
- GLM-4.7-flash LLM returns [RETRIEVE_NO] for English input

**needs_retrieval() Flow** (self_rag.py:119-170):
1. `_has_regulation_keywords()` → FALSE for English
2. LLM evaluation with Korean prompt → [RETRIEVE_NO]
3. Override check (keywords again) → FALSE for English
4. Return False → Rejection

#### 2.2 Answer Generation CoT Leakage

The answer generation prompt at `search_usecase.py:364-520` includes instructions but doesn't explicitly suppress CoT:
- LLM outputs reasoning steps: "1. Analyze the User's Request:", "User Persona:", "Constraint:"
- These are visible in the final answer
- GLM model tends to output extended reasoning when not constrained

#### 2.3 Retrieval Quality Issues

**BGE-M3 Embedding** (multilingual):
- Already supports English → Dense search should work for English queries
- Problem: Self-RAG blocks BEFORE search is attempted

**KoNLPy BM25** (Korean-only):
- Morphological analysis optimized for Korean
- English queries get poor BM25 scores
- Need: Dense-only search for English queries, or translate to Korean for BM25

**Corrective RAG Thresholds**:
- Dynamic: simple=0.3, medium=0.4, complex=0.5
- May be too permissive (allows low-relevance docs through)

#### 2.4 Hallucination Patterns

Three distinct patterns observed:
1. **Wrong document usage**: 휴직 query → 비품 지급기준 response (Q11)
2. **Context ignored**: 연구년 적용 기준 → "정보 없음" despite relevant context (Q12)
3. **Irrelevant doc marked relevant**: 등록금 납부 유예 + 비품 지급기준 = 100% (Q10)

Root cause: No post-retrieval relevance filtering between search and generation.

### 3. Existing Patterns & Conventions

#### Configuration Pattern
- Config class at `src/rag/domain/config.py`
- Self-RAG enabled by default (config.py:241-242)
- Environment variables: `ENABLE_SELF_RAG=true`, `ENABLE_HYDE=true`

#### Test Pattern
- Tests in `tests/` directory
- pytest with fixtures
- Mock LLM clients for testing

#### Error Handling
- Circuit breaker pattern for LLM connections
- Fallback providers for LLM
- Retry logic in LLM adapter

### 4. Risks & Constraints

1. **GLM-4.7-flash limitations**: Local model with limited multilingual ability
2. **Token budget**: max_tokens=2048 globally, answer generation needs sufficient tokens
3. **Latency**: Adding translation step increases response time
4. **BM25 Korean-only**: KoNLPy tokenizer cannot be easily made bilingual
5. **Clean Architecture**: Changes in infrastructure layer must not affect domain

### 5. Recommendations

1. **Highest impact**: Fix Self-RAG keyword list + prompt (addresses 8 failing queries)
2. **Quick win**: Add CoT stripping to answer generation (addresses 3+ queries)
3. **Medium effort**: Post-retrieval relevance filtering (addresses 3 queries)
4. **Consider**: Dense-only search path for English queries
5. **BGE-M3 advantage**: Already multilingual, leverage for English queries
