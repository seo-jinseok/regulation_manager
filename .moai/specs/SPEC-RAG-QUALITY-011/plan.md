# Implementation Plan: RAG System Quality Improvement

**SPEC ID:** SPEC-RAG-QUALITY-011
**Created:** 2026-02-24
**Status:** Planned

---

## Milestone Overview

| Milestone | Priority | Dependencies | Est. Complexity |
|-----------|----------|--------------|-----------------|
| M1: Self-RAG Prompt Fix | Primary | None | Low |
| M2: Keyword Pre-Filtering | Primary | None | Low |
| M3: Fallback Mechanism | Primary | M1, M2 | Medium |
| M4: Health Verification | Secondary | None | Low |
| M5: Metrics & Logging | Secondary | M3 | Low |
| M6: Configuration | Optional | M1-M3 | Low |
| M7: Testing & Validation | Final | M1-M6 | Medium |

---

## M1: Self-RAG Prompt Fix (Primary)

### Objective
Replace the ambiguous Self-RAG classification prompt with a domain-specific prompt that defaults to retrieval for uncertain cases.

### Tasks

1. **Update Prompt Constant**
   - File: `src/rag/infrastructure/self_rag.py`
   - Replace `RETRIEVAL_NEEDED_PROMPT` with improved version
   - Include explicit regulation domain context
   - Add examples of queries requiring retrieval
   - Emphasize default to retrieval for uncertain cases

2. **Update System Prompt**
   - Add domain context to LLM call
   - Specify Korean language response expected

3. **Unit Tests**
   - File: `tests/rag/unit/infrastructure/test_self_rag.py`
   - Test regulation queries return True
   - Test greeting queries return False
   - Test ambiguous queries default to True

### Files to Modify
- `src/rag/infrastructure/self_rag.py`
- `tests/rag/unit/infrastructure/test_self_rag.py`

### Validation Criteria
- [ ] All unit tests pass
- [ ] Manual testing with sample queries shows improved classification
- [ ] No regression in existing functionality

---

## M2: Keyword Pre-Filtering (Primary)

### Objective
Implement fast keyword-based classification that bypasses LLM for obvious regulation queries.

### Tasks

1. **Create Keyword List**
   - File: `data/config/self_rag_keywords.json`
   - Include regulation types, structural references, common topics
   - Make configurable via external file

2. **Implement Pre-Filter Method**
   - File: `src/rag/infrastructure/self_rag.py`
   - Add `_has_regulation_keywords(query: str) -> bool`
   - Load keywords from config file
   - Implement fast string matching

3. **Integrate with needs_retrieval()**
   - Check keywords before LLM call
   - Skip LLM if keywords found
   - Log keyword matches for debugging

4. **Unit Tests**
   - Test keyword matching accuracy
   - Test performance (target: <1ms)
   - Test with various query formats

### Files to Create
- `data/config/self_rag_keywords.json`

### Files to Modify
- `src/rag/infrastructure/self_rag.py`
- `src/rag/config.py`
- `tests/rag/unit/infrastructure/test_self_rag.py`

### Validation Criteria
- [ ] Keyword matching works correctly
- [ ] Performance meets <1ms target
- [ ] Keywords are configurable

---

## M3: Fallback Mechanism (Primary)

### Objective
Implement override logic that ensures retrieval when keywords are detected, even if LLM returns `[RETRIEVE_NO]`.

### Tasks

1. **Modify needs_retrieval() Logic**
   - Implement two-stage classification
   - Stage 1: Keyword pre-filtering
   - Stage 2: LLM classification (if Stage 1 passes)
   - Override: If LLM says NO but keywords exist, override to YES

2. **Add Override Logging**
   - Log when override activates
   - Track override reasons
   - Include in metrics

3. **Add Override Counter**
   - Track total overrides
   - Calculate override rate
   - Expose via method/property

4. **Integration Tests**
   - Test end-to-end query flow
   - Test override scenarios
   - Test logging output

### Files to Modify
- `src/rag/infrastructure/self_rag.py`
- `tests/rag/unit/infrastructure/test_self_rag.py`
- `tests/rag/integration/test_rag_pipeline.py`

### Validation Criteria
- [ ] Override logic activates correctly
- [ ] Logging captures override events
- [ ] No queries incorrectly rejected

---

## M4: Health Verification (Secondary)

### Objective
Add ChromaDB health checks to detect empty collections or missing embedding functions.

### Tasks

1. **Implement health_check() Method**
   - File: `src/rag/infrastructure/chroma_store.py`
   - Check collection count
   - Verify embedding function availability
   - Return structured health status

2. **Add Initialization Verification**
   - File: `src/rag/application/search_usecase.py`
   - Call health check during initialization
   - Log warnings for degraded/unhealthy status

3. **Add Empty Collection Handling**
   - Return informative error message
   - Suggest data ingestion
   - Distinguish from "no relevant results"

4. **Unit Tests**
   - Test health check with empty collection
   - Test with missing embedding function
   - Test with healthy status

### Files to Modify
- `src/rag/infrastructure/chroma_store.py`
- `src/rag/application/search_usecase.py`
- `tests/rag/unit/infrastructure/test_chroma_store.py`

### Validation Criteria
- [ ] Health check returns correct status
- [ ] Warnings logged for issues
- [ ] Empty collection returns informative error

---

## M5: Metrics & Logging (Secondary)

### Objective
Track and expose Self-RAG classification metrics for monitoring and debugging.

### Tasks

1. **Add Classification Counters**
   - Counter for `[RETRIEVE_YES]`
   - Counter for `[RETRIEVE_NO]`
   - Counter for keyword bypasses
   - Counter for overrides

2. **Implement Metrics Method**
   - Return current counter values
   - Calculate rates (override rate, etc.)
   - Format for logging/monitoring

3. **Add Structured Logging**
   - Log classification decisions
   - Include query hash for tracing
   - Include latency metrics

4. **Add Metrics Endpoint (Optional)**
   - Expose via API if applicable
   - Format for Prometheus/Grafana

### Files to Modify
- `src/rag/infrastructure/self_rag.py`
- `src/rag/application/search_usecase.py`

### Validation Criteria
- [ ] Counters track correctly
- [ ] Metrics accessible via method
- [ ] Logs contain required information

---

## M6: Configuration (Optional)

### Objective
Add configuration options for Self-RAG behavior tuning.

### Tasks

1. **Add Configuration Fields**
   - File: `src/rag/config.py`
   - `self_rag_keywords_path`: Path to keyword list
   - `self_rag_override_on_keywords`: Enable/disable override
   - `self_rag_log_overrides`: Enable/disable override logging

2. **Update Environment Variable Support**
   - `SELF_RAG_KEYWORDS_PATH`
   - `SELF_RAG_OVERRIDE_ON_KEYWORDS`
   - `SELF_RAG_LOG_OVERRIDES`

3. **Load Keywords from Config**
   - Read keyword file on initialization
   - Fallback to default list if file missing
   - Log keyword source

4. **Unit Tests**
   - Test configuration loading
   - Test environment variable override
   - Test default behavior

### Files to Modify
- `src/rag/config.py`
- `src/rag/infrastructure/self_rag.py`
- `tests/rag/unit/test_config.py`

### Validation Criteria
- [ ] Configuration loads correctly
- [ ] Environment variables override defaults
- [ ] Keywords load from file

---

## M7: Testing & Validation (Final)

### Objective
Comprehensive testing and validation of all changes.

### Tasks

1. **Unit Tests**
   - All modified components
   - Edge case handling
   - Error scenarios

2. **Integration Tests**
   - End-to-end query flow
   - Self-RAG + ChromaDB integration
   - LLM adapter integration

3. **Regression Tests**
   - Verify no existing functionality broken
   - Run existing test suite
   - Performance benchmarks

4. **Quality Evaluation**
   - Run `scripts/evaluate_rag_quality.py`
   - Compare before/after metrics
   - Target: >= 80% pass rate

5. **Manual Testing**
   - Test with various query types
   - Test edge cases
   - Verify logging output

### Validation Criteria
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] No regression in existing tests
- [ ] Quality evaluation shows improvement
- [ ] Manual testing confirms fixes

---

## Technical Approach

### Development Methodology
- **Method:** Hybrid TDD/DDD
- **New Features:** TDD (RED-GREEN-REFACTOR)
- **Existing Code:** DDD (ANALYZE-PRESERVE-IMPROVE)

### Code Quality Standards
- Test coverage: >= 85%
- Linting: ruff with project rules
- Type hints: Required for all public methods
- Documentation: Docstrings for all modified methods

### Branch Strategy
- Create feature branch: `fix/SPEC-RAG-QUALITY-011-self-rag-classification`
- Commit messages: Reference SPEC ID
- PR description: Include test results

---

## Risk Mitigation

### Rollback Plan
1. If issues detected, disable Self-RAG via `ENABLE_SELF_RAG=false`
2. System falls back to always-retrieve behavior
3. Investigate and fix issues in isolation

### Monitoring
1. Monitor override rate - should be < 10%
2. Monitor query pass rate - should be >= 80%
3. Monitor latency - should not increase significantly

---

## Dependencies

### Internal Dependencies
- `src/rag/infrastructure/self_rag.py`
- `src/rag/infrastructure/chroma_store.py`
- `src/rag/application/search_usecase.py`
- `src/rag/config.py`

### External Dependencies
- LLM Provider (OpenRouter)
- ChromaDB with ko-sbert-sts embeddings

### Test Dependencies
- pytest
- pytest-asyncio
- pytest-mock

---

## Deliverables

1. **Code Changes**
   - Modified source files per milestones
   - New configuration files
   - Updated test files

2. **Documentation**
   - Updated docstrings
   - Configuration documentation
   - Troubleshooting guide

3. **Test Results**
   - Unit test report
   - Integration test report
   - Quality evaluation report

4. **Metrics**
   - Before/after comparison
   - Performance benchmarks
   - Classification accuracy metrics

---

**Plan Version:** 1.0
**Last Updated:** 2026-02-24
**Estimated Complexity:** Medium
