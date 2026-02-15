# Implementation Plan: RAG Retrieval Quality Improvement

**SPEC ID:** SPEC-RAG-QUALITY-003
**Title:** Implementation Plan - 어휘 불일치 문제 해결
**Created:** 2026-02-15
**Status:** Planned

---

## Implementation Strategy

### Approach Overview

This implementation addresses the vocabulary mismatch problem through a multi-layered approach:

1. **Query Transformation Layer**: Convert colloquial Korean to formal Korean
2. **Morphological Expansion Layer**: Expand queries with Korean morphological variants
3. **Evaluation Enhancement Layer**: Use semantic similarity instead of keyword matching
4. **LLM Integration Layer**: Add LLM-as-Judge for nuanced evaluation
5. **Search Optimization Layer**: Dynamic weighting for hybrid search

### Development Methodology

**Methodology:** Hybrid (TDD for new components, DDD for modifications)

**Rationale:**
- New components (ColloquialTransformer, SemanticEvaluator) benefit from TDD
- Existing components (BM25Indexer, QueryProcessor) require DDD for behavior preservation

---

## Milestones

### Milestone 1: Foundation (Priority: P0 - Blocker)

**Goal:** Setup infrastructure and configuration

**Deliverables:**
- [ ] Create `colloquial_mappings.json` configuration file
- [ ] Add configuration schema to `evaluation/config.yaml`
- [ ] Setup test fixtures for colloquial query testing
- [ ] Create baseline evaluation metrics documentation

**Verification:**
- Configuration files exist and validate against schema
- Test fixtures cover at least 20 colloquial patterns

---

### Milestone 2: Colloquial Query Transformer (Priority: P1 - Critical)

**Goal:** Implement colloquial-to-formal query transformation

**Files:**
- `src/rag/application/colloquial_transformer.py` (new)
- `src/rag/application/query_processor.py` (modify)
- `tests/rag/application/test_colloquial_transformer.py` (new)

**Implementation Steps:**

1. **Create ColloquialTransformer class**
   - Define TransformResult dataclass
   - Implement pattern matching logic
   - Add regex pattern support

2. **Implement transformation logic**
   - Load mappings from configuration
   - Apply pattern-based transformation
   - Handle edge cases (unrecognized patterns)

3. **Integrate with QueryProcessor**
   - Add transformation step to query pipeline
   - Implement logging for debugging
   - Add configuration toggle

4. **Write comprehensive tests**
   - Unit tests for pattern matching
   - Integration tests with QueryProcessor
   - Edge case tests

**Verification:**
- All unit tests pass
- Transformation accuracy >= 95%
- Query processor overhead < 50ms

---

### Milestone 3: Morphological Expansion Enhancement (Priority: P2 - High)

**Goal:** Enhance BM25 indexer with morphological expansion

**Files:**
- `src/rag/infrastructure/bm25_indexer.py` (modify)
- `tests/rag/infrastructure/test_bm25_indexer.py` (extend)

**Implementation Steps:**

1. **Analyze existing BM25Indexer**
   - Read current implementation
   - Identify integration points
   - Document existing behavior

2. **Create characterization tests**
   - Capture current indexing behavior
   - Capture current query behavior
   - Ensure behavior preservation

3. **Add morphological expansion**
   - Integrate KiwiPiePy expansion
   - Implement caching for expansions
   - Add configurable expansion depth

4. **Verify behavior preservation**
   - Run characterization tests
   - Compare retrieval results
   - Document changes

**Verification:**
- All characterization tests pass
- Expansion cache hit rate >= 70%
- No regression in existing queries

---

### Milestone 4: Semantic Evaluator (Priority: P3 - High)

**Goal:** Implement embedding-based semantic evaluation

**Files:**
- `src/rag/evaluation/semantic_evaluator.py` (new)
- `src/rag/evaluation/metrics.py` (modify)
- `tests/rag/evaluation/test_semantic_evaluator.py` (new)

**Implementation Steps:**

1. **Create SemanticEvaluator class**
   - Define evaluation interface
   - Implement embedding generation
   - Add similarity calculation

2. **Integrate with BGE-M3**
   - Use existing embedding model
   - Implement batch processing
   - Add result caching

3. **Update evaluation metrics**
   - Add semantic similarity metric
   - Update evaluation pipeline
   - Configure thresholds

4. **Write comprehensive tests**
   - Unit tests for similarity calculation
   - Integration tests with evaluation framework
   - Threshold validation tests

**Verification:**
- Semantic evaluation accuracy >= 90%
- Evaluation time < 200ms per query
- Configurable threshold working

---

### Milestone 5: LLM-as-Judge Integration (Priority: P4 - Medium)

**Goal:** Integrate LLM-based evaluation for nuanced assessment

**Files:**
- `src/rag/evaluation/llm_judge.py` (modify)
- `src/rag/evaluation/config.yaml` (modify)
- `tests/rag/evaluation/test_llm_judge.py` (extend)

**Implementation Steps:**

1. **Analyze existing LLM integration**
   - Review current LLM provider support
   - Identify extension points
   - Document existing behavior

2. **Implement LLM-as-Judge logic**
   - Add judgment prompt templates
   - Implement response parsing
   - Add judgment caching

3. **Add graceful degradation**
   - Detect LLM unavailability
   - Fallback to semantic evaluation
   - Log degradation events

4. **Write integration tests**
   - Test with mock LLM
   - Test degradation paths
   - Test caching behavior

**Verification:**
- LLM judgment works when available
- Graceful degradation functions correctly
- Cache reduces API calls by >= 50%

---

### Milestone 6: Hybrid Weight Optimizer (Priority: P5 - Medium)

**Goal:** Implement dynamic weighting for hybrid search

**Files:**
- `src/rag/application/hybrid_weight_optimizer.py` (new)
- `src/rag/application/search_service.py` (modify)
- `tests/rag/application/test_hybrid_weight_optimizer.py` (new)

**Implementation Steps:**

1. **Create HybridWeightOptimizer class**
   - Define formality detection logic
   - Implement weight calculation
   - Add logging

2. **Integrate with search pipeline**
   - Modify search service
   - Add weight configuration
   - Test integration

3. **Write comprehensive tests**
   - Unit tests for formality detection
   - Integration tests with search
   - Performance tests

**Verification:**
- Formality detection accuracy >= 85%
- Weight adjustment improves retrieval
- No performance regression

---

### Milestone 7: Integration and Validation (Priority: P0 - Final)

**Goal:** Full integration testing and validation

**Implementation Steps:**

1. **Run full evaluation suite**
   - Execute all 30 evaluation queries
   - Collect metrics
   - Compare with targets

2. **Performance testing**
   - Measure query latency
   - Check memory usage
   - Identify bottlenecks

3. **Documentation update**
   - Update API documentation
   - Update configuration guide
   - Update README

4. **Final verification**
   - All acceptance criteria met
   - No regression detected
   - Documentation complete

**Verification:**
- Overall pass rate >= 80%
- All metrics >= 0.75
- Documentation reviewed

---

## Technical Approach

### Architecture Decisions

1. **Layered Transformation Pipeline**
   - Input: Raw user query
   - Layer 1: Colloquial transformation
   - Layer 2: Morphological expansion
   - Layer 3: Vector/BM25 search
   - Output: Ranked results

2. **Dual Evaluation Mode**
   - Primary: Semantic similarity (always available)
   - Secondary: LLM-as-Judge (when LLM available)

3. **Configuration-Driven Behavior**
   - All thresholds configurable
   - Transformation patterns in JSON
   - Feature flags for gradual rollout

### Performance Optimization

1. **Caching Strategy**
   - Morphological expansions: LRU cache (1000 entries)
   - Semantic evaluations: Redis cache (1 hour TTL)
   - LLM judgments: Redis cache (24 hour TTL)

2. **Lazy Loading**
   - Colloquial mappings loaded on first use
   - Embedding model loaded lazily

3. **Batch Processing**
   - Batch embedding generation for efficiency
   - Parallel evaluation where possible

---

## Dependencies

### Internal Dependencies

| Component | Dependency | Status |
|-----------|------------|--------|
| ColloquialTransformer | Configuration system | Available |
| SemanticEvaluator | BGE-M3 embedding model | Available |
| LLMJudge | LLM provider interface | Available |
| HybridWeightOptimizer | Search service | Available |

### External Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| KiwiPiePy | >= 0.20.0 | Korean morphological analysis |
| sentence-transformers | >= 2.2.0 | Embedding generation |
| numpy | >= 1.24.0 | Similarity calculation |

---

## Risk Management

### Technical Risks

| Risk | Mitigation | Contingency |
|------|------------|-------------|
| Transformation accuracy low | Iterative pattern expansion | Manual review process |
| Semantic evaluation slow | Caching and optimization | Reduce evaluation frequency |
| LLM unavailable | Graceful degradation | Semantic-only mode |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Pattern collection incomplete | Medium | High | Start with top 50 patterns |
| Integration issues | Low | Medium | Comprehensive testing |
| Performance regression | Low | High | Performance benchmarks |

---

## Testing Strategy

### Unit Tests

- ColloquialTransformer: 30+ test cases
- SemanticEvaluator: 20+ test cases
- HybridWeightOptimizer: 15+ test cases

### Integration Tests

- Query pipeline transformation
- Evaluation pipeline integration
- Search pipeline integration

### Performance Tests

- Query latency benchmarks
- Memory usage profiling
- Cache effectiveness

### Acceptance Tests

- Full evaluation suite (30 queries)
- Regression testing (existing queries)
- Edge case handling

---

## Rollout Plan

### Phase 1: Development (Current)
- Implement all components
- Write comprehensive tests
- Internal validation

### Phase 2: Staging
- Deploy to staging environment
- Run extended evaluation
- Collect feedback

### Phase 3: Production
- Feature flag rollout (10% -> 50% -> 100%)
- Monitor metrics
- Iterative improvement

---

## Definition of Done

### Code Quality
- [ ] All code passes ruff linting
- [ ] All code passes mypy type checking
- [ ] Test coverage >= 85%
- [ ] No security vulnerabilities

### Functionality
- [ ] All acceptance criteria met
- [ ] All tests pass
- [ ] No regression
- [ ] Performance requirements met

### Documentation
- [ ] API documentation updated
- [ ] Configuration guide updated
- [ ] README updated
- [ ] CHANGELOG entry created

### Deployment
- [ ] Configuration files deployed
- [ ] Feature flags configured
- [ ] Rollback procedure documented

---

**Plan Status:** Complete
**Ready for Implementation:** Pending approval
**Estimated Complexity:** Medium-High
