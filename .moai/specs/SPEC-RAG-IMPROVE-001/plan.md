# Implementation Plan: SPEC-RAG-IMPROVE-001

**TAG BLOCK**
```
SPEC: RAG-IMPROVE-001
Document: plan.md
Version: 1.0
Last Updated: 2026-02-09
```

## Milestones

### Primary Goals (Phase 1 + Phase 2)

**Milestone 1: Quick Wins (Week 1)**
Target: Achieve minimal quality targets through low-effort improvements

**Success Criteria:**
- Accuracy: 0.812 → 0.835+ (Gap: +0.023)
- Completeness: 0.736 → 0.750+ (Gap: +0.014)
- Citations: 0.743 → 0.780+ (Gap: +0.037)
- No regression in Context Relevance (maintain 0.833+)

**Deliverables:**
- Enhanced citation extraction and formatting
- Improved LLM prompts for context adherence
- Basic query expansion using existing synonym service
- Updated evaluation results showing improvement

### Secondary Goals (Phase 3)

**Milestone 2: Advanced Features (Week 3-4)**
Target: Push performance beyond initial targets

**Success Criteria:**
- Accuracy: 0.850 → 0.870+
- Completeness: 0.750 → 0.780+
- Professor/Parent/International pass rates: 80% → 90%+

**Deliverables:**
- Multi-hop retrieval implementation
- Self-consistency checking
- Adaptive retrieval strategy
- Comprehensive performance report

### Optional Goals (Future Enhancement)

**Milestone 3: Continuous Improvement**
Target: Establish ongoing quality monitoring

**Success Criteria:**
- Automated quality monitoring dashboard
- User feedback integration pipeline
- A/B testing framework for retrieval strategies

## Technical Approach

### Phase 1: Quick Wins (Week 1)

#### Task 1.1: Enhanced Citation Extraction
**Priority:** High
**Estimated Effort:** 2 days
**Dependencies:** None

**Approach:**
1. Extend existing `src/rag/domain/citation/citation_validator.py`
2. Add structured citation extraction from document metadata
3. Implement citation formatting according to university standards
4. Enforce citation inclusion in response template

**Implementation Steps:**
- Add `extract_citations()` method to CitationExtractor
- Parse `rule_code`, `article_number` from document metadata
- Format citations as `[규정 제X조 제Y항]`
- Update response generation to include citations

**Testing:**
- Unit test citation extraction accuracy (>95%)
- Integration test with sample queries requiring citations

#### Task 1.2: Factual Consistency Prompts
**Priority:** High
**Estimated Effort:** 1 day
**Dependencies:** None

**Approach:**
1. Modify LLM generation prompts in `src/rag/infrastructure/llm_adapter.py`
2. Add strict "answer from context" instructions
3. Implement negative examples (what NOT to do)
4. Add validation prompts for consistency checking

**Implementation Steps:**
- Update `generate_response()` prompt template
- Add context adherence instruction: "Answer ONLY using provided context"
- Add instruction: "If information is not in context, state that clearly"
- Add validation prompt: "Verify all claims are supported by context"

**Testing:**
- A/B test with/without new prompts on evaluation dataset
- Measure accuracy improvement target: +0.015

#### Task 1.3: Basic Query Expansion
**Priority:** High
**Estimated Effort:** 2 days
**Dependencies:** None

**Approach:**
1. Utilize existing `src/rag/application/synonym_generator_service.py`
2. Implement query expansion using synonym database
3. Generate query variants for parallel retrieval
4. Merge and deduplicate retrieval results

**Implementation Steps:**
- Create `QueryExpansionService` class
- Integrate with existing `SynonymGeneratorService`
- Generate 2-3 query variants per original query
- Execute parallel searches and merge results using reciprocal rank fusion

**Testing:**
- Evaluate retrieval improvement on test queries
- Measure completeness improvement target: +0.010

### Phase 2: Core Improvements (Week 2-3)

#### Task 2.1: Factual Consistency Validator
**Priority:** High
**Estimated Effort:** 3 days
**Dependencies:** Task 1.2 completion

**Approach:**
1. Create new component: `src/rag/domain/validation/factual_validator.py`
2. Implement claim extraction from generated response
3. Verify each claim against retrieved documents
4. Calculate consistency score and flag violations

**Implementation Steps:**
- Design `FactualConsistencyValidator` class with validation interface
- Implement claim extraction using NLP or LLM
- Implement claim verification against context
- Add consistency score calculation (0-1 scale)
- Integrate validator into query processing pipeline

**Testing:**
- Unit test validator accuracy (>90% hallucination detection)
- Integration test with full query pipeline
- Measure accuracy improvement target: +0.015

#### Task 2.2: Hybrid Retrieval System
**Priority:** High
**Estimated Effort:** 4 days
**Dependencies:** Task 1.3 completion

**Approach:**
1. Create new component: `src/rag/infrastructure/hybrid_retriever.py`
2. Implement dense vector search (existing ChromaDB)
3. Implement sparse keyword search (BM25 or similar)
4. Merge results using Reciprocal Rank Fusion (RRF)

**Implementation Steps:**
- Design `HybridRetriever` class with retrieval interface
- Integrate existing `ChromaVectorStore` for dense search
- Implement sparse search using BM25 (e.g., `rank_bm25` library)
- Implement RRF for result merging: `score = 1/(k+rank_dense) + 1/(k+rank_sparse)`
- Add re-ranking using existing reranker

**Testing:**
- Evaluate retrieval quality on test dataset
- Measure context relevance and completeness improvements
- Target: Context Relevance maintain 0.833+, Completeness +0.020

#### Task 2.3: Persona-Aware Response Generation
**Priority:** Medium
**Estimated Effort:** 3 days
**Dependencies:** None

**Approach:**
1. Create new component: `src/rag/application/persona_generator.py`
2. Implement persona detection from query patterns
3. Create persona-specific response templates
4. Adapt response complexity based on detected persona

**Implementation Steps:**
- Design `PersonaAwareGenerator` class
- Implement persona detection using heuristics or classifier
  - Language detection (Korean vs English)
  - Query complexity analysis
  - Topic patterns (research vs administrative)
- Create response templates for each persona:
  - Professor: Technical, detailed, comprehensive
  - Parent: Simple, practical, action-oriented
  - International: Bilingual, culturally adapted
- Adjust terminology and detail level

**Testing:**
- Persona-specific test suite with sample queries
- Measure persona pass rate improvement target: 60% → 80%+

### Phase 3: Advanced Features (Week 4-5) [Optional]

#### Task 3.1: Multi-Hop Retrieval
**Priority:** Low
**Estimated Effort:** 5 days
**Dependencies:** Task 2.2 completion

**Approach:**
1. Implement query decomposition for complex questions
2. Execute sequential retrieval for decomposed sub-queries
3. Aggregate and synthesize information from multiple hops

#### Task 3.2: Self-Consistency Checking
**Priority:** Low
**Estimated Effort:** 3 days
**Dependencies:** Task 2.1 completion

**Approach:**
1. Generate multiple response variants
2. Compare and reconcile differences
3. Select most consistent response

#### Task 3.3: Adaptive Retrieval Strategy
**Priority:** Low
**Estimated Effort:** 4 days
**Dependencies:** Task 2.2, Task 2.3 completion

**Approach:**
1. Classify query complexity
2. Select appropriate retrieval strategy (simple/hybrid/multi-hop)
3. Dynamically adjust retrieval parameters

## Risk Management

### Technical Risks

**Risk 1: LLM API Rate Limits**
- Probability: Medium
- Impact: High
- Mitigation: Implement caching, batch requests, use fallback models

**Risk 2: Retrieval Performance Degradation**
- Probability: Medium
- Impact: Medium
- Mitigation: A/B testing, gradual rollout, performance monitoring

**Risk 3: Citation Extraction Accuracy**
- Probability: Low
- Impact: Medium
- Mitigation: Comprehensive testing, manual validation of edge cases

### Integration Risks

**Risk 4: Breaking Existing Functionality**
- Probability: Medium
- Impact: High
- Mitigation: Comprehensive regression testing, feature flags

**Risk 5: Increased Latency**
- Probability: High
- Impact: Medium
- Mitigation: Parallel processing, caching, performance optimization

### Schedule Risks

**Risk 6: Underestimated Complexity**
- Probability: Medium
- Impact: Medium
- Mitigation: Phased approach, MVP focus, buffer time

## Dependencies

### Internal Dependencies

- `src/rag/domain/citation/citation_validator.py` - Citation validation
- `src/rag/application/synonym_generator_service.py` - Synonym generation
- `src/rag/infrastructure/llm_adapter.py` - LLM client
- `src/rag/infrastructure/chroma_store.py` - Vector store
- `src/rag/infrastructure/reranker.py` - Result re-ranking

### External Dependencies

- OpenAI GPT-4o API - LLM generation
- ChromaDB - Vector database
- rank_bm25 - Sparse retrieval (if needed)
- pytest - Testing framework

### Resource Requirements

- Development: 1 senior developer (full-time for 2 weeks)
- Testing: 1 QA engineer (part-time for 1 week)
- Infrastructure: No additional infrastructure required
- Budget: OpenAI API costs (~$50-100 for testing)

## Rollout Plan

### Phase 1 Rollout (Week 1)
1. Deploy citation enhancement to staging environment
2. Run evaluation suite to measure improvement
3. Deploy factual consistency prompts to staging
4. Run evaluation suite to measure improvement
5. Deploy query expansion to staging
6. Run full evaluation suite
7. If all targets met, deploy to production

### Phase 2 Rollout (Week 2-3)
1. Deploy factual consistency validator to staging
2. Run evaluation suite to measure improvement
3. Deploy hybrid retriever to staging
4. Run evaluation suite to measure improvement
5. Deploy persona-aware generator to staging
6. Run full evaluation suite with all personas
7. If all targets met, deploy to production with feature flags

### Phase 3 Rollout (Week 4-5) [Optional]
1. Deploy multi-hop retrieval to staging
2. Deploy self-consistency checker to staging
3. Deploy adaptive retrieval strategy to staging
4. Run comprehensive evaluation
5. Gradual rollout to production with monitoring

## Monitoring and Validation

### Quality Metrics Dashboard

**Real-time Metrics:**
- Accuracy score (rolling 100 queries)
- Completeness score (rolling 100 queries)
- Citation quality (rolling 100 queries)
- Context relevance (rolling 100 queries)

**Per-Persona Metrics:**
- Pass rate by persona (updated hourly)
- Average score by persona (updated hourly)
- Common issues by persona (updated daily)

**Alerting:**
- Alert if Accuracy drops below 0.830
- Alert if Completeness drops below 0.740
- Alert if any persona pass rate drops below 70%

### Validation Process

**Daily Validation:**
- Run evaluation suite on sample queries (30 queries)
- Compare against baseline metrics
- Investigate any regressions

**Weekly Validation:**
- Run full ParallelPersonaEvaluator evaluation (30 queries)
- Generate comprehensive report
- Review failure patterns and adjust strategy

**Monthly Validation:**
- Run expanded evaluation (100+ queries)
- Conduct A/B testing for new features
- Update quality targets based on baseline

## Success Metrics

### Quantitative Metrics

**Primary Metrics (Must Achieve):**
- Accuracy: 0.812 → 0.850+ (+4.7% improvement)
- Completeness: 0.736 → 0.750+ (+1.9% improvement)
- Overall Pass Rate: 76.7% → 85%+ (+10.8% improvement)

**Secondary Metrics (Should Achieve):**
- Professor Pass Rate: 60% → 80%+ (+33% improvement)
- Parent Pass Rate: 60% → 80%+ (+33% improvement)
- International Pass Rate: 60% → 80%+ (+33% improvement)
- Citations: 0.743 → 0.780+ (+5% improvement)

**Tertiary Metrics (Nice to Have):**
- Context Relevance: Maintain 0.833+
- Average Response Latency: < 3 seconds
- API Cost per Query: < $0.02

### Qualitative Metrics

**User Satisfaction:**
- Positive feedback rate > 80%
- User-reported issues < 5%
- Feature adoption rate > 70%

**System Reliability:**
- Uptime > 99.5%
- Error rate < 1%
- Failed queries < 2%

## Definition of Done

A task is considered complete when:
- [ ] Code is implemented and follows coding standards
- [ ] Unit tests are written with >85% coverage
- [ ] Integration tests pass
- [ ] Documentation is updated
- [ ] Code review is approved
- [ ] Performance impact is assessed
- [ ] Quality metrics are measured
- [ ] Rollback plan is documented

A phase is considered complete when:
- [ ] All tasks in phase are complete
- [ ] Quality targets are achieved
- [ ] No regressions in existing functionality
- [ ] Stakeholder sign-off is obtained
- [ ] Monitoring and alerting are in place
