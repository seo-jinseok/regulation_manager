# Implementation Plan: SPEC-RAG-SEARCH-001

**Created**: 2026-02-07
**SPEC Version**: 1.0
**Agent in Charge**: manager-strategy

## 1. Overview

### SPEC Summary

RAG 검색 시스템의 Contextual Recall을 32%에서 55-60%로 개선하는 포괄적인 검색 강화 계획입니다.

### Implementation Scope

이 구현은 다음을 포함합니다:
- 6개 새로운 엔티티 유형에 대한 향상된 엔티티 인식
- 3단계 쿼리 확장 파이프라인
- 쿼리 복잡도에 따른 적응형 Top-K 선택
- 인용 추적을 위한 기본 멀티-홉 검색

### Exclusions

다음은 이 구현에서 제외됩니다:
- 규정 데이터 수정
- 완전한 재색인
- UI 개발
- LLM 프롬프트 변경

## 2. Technology Stack

### Existing Libraries (No Changes)

| Library | Current Version | Purpose |
|---------|----------------|---------|
| Python | 3.11+ | Runtime |
| llama-index | 0.14.10+ | RAG Framework |
| ChromaDB | 1.4.0+ | Vector Database |
| rank-bm25 | latest | BM25 Retrieval |

### New Components (Internal)

| Component | Purpose | Status |
|-----------|---------|--------|
| EntityRecognizer | Enhanced entity recognition | NEW |
| MultiStageQueryExpander | 3-stage query expansion | MODIFIED |
| AdaptiveTopKSelector | Dynamic Top-K selection | NEW |
| MultiHopRetriever | Citation following | NEW |

### Environment Requirements

- Python: 3.11+
- Memory: 4GB+ (for citation graph caching)
- Latency Budget: < 500ms per query
- Test Coverage: 85%+

## 3. TAG Chain Design

### TAG List

**TAG-001: Entity Recognition Enhancement**
- Purpose: 6개 새로운 엔티티 유형 인식
- Scope: RegulationEntityRecognizer 구현
- Completion Condition: 모든 엔티티 유형 테스트 통과 (90%+ coverage)
- Dependency: None

**TAG-002: Query Expansion Pipeline**
- Purpose: 3단계 쿼리 확장 구현
- Scope: MultiStageQueryExpander 구현 및 통합
- Completion Condition: 3단계 확장 테스트 통과 (85%+ coverage)
- Dependency: TAG-001 (entity recognition)

**TAG-003: Adaptive Top-K Implementation**
- Purpose: 동적 Top-K 선택 구현
- Scope: AdaptiveTopKSelector 구현
- Completion Condition: 모든 복잡도 수준 테스트 통과 (85%+ coverage)
- Dependency: None (can be parallel with TAG-001, TAG-002)

**TAG-004: Multi-Hop Retrieval**
- Purpose: 인용 추적 기능 구현
- Scope: MultiHopRetriever 구현
- Completion Condition: 2홉 추적 테스트 통과 (80%+ coverage)
- Dependency: None (can be parallel with other TAGs)

**TAG-005: Integration & Testing**
- Purpose: 모든 컴포넌트 통합 및 평가
- Scope: EnhancedSearch orchestration, end-to-end testing
- Completion Condition: Contextual Recall 50%+ 달성, 모든 지표 회귀 없음
- Dependency: TAG-001, TAG-002, TAG-003, TAG-004

### TAG Dependency Diagram

```
[TAG-001: Entity Recognition] ──┐
                                ├──> [TAG-005: Integration]
[TAG-002: Query Expansion]   ───┘
[TAG-003: Adaptive Top-K]    ───────> [TAG-005: Integration]
[TAG-004: Multi-Hop]        ───────> [TAG-005: Integration]
```

## 4. Step-by-Step Implementation Plan

### Week 1: Entity Recognition & Query Expansion

**TAG-001: Entity Recognition Enhancement** (Days 1-2)

Goal: 6개 새로운 엔티티 유형 구현

Tasks:
- [ ] Day 1 AM: EntityRecognizer class skeleton 구현
- [ ] Day 1 PM: Section pattern recognition 구현
- [ ] Day 2 AM: Procedure/Requirement/Benefit/Deadline recognition 구현
- [ ] Day 2 PM: Hypernym expansion 구현 및 테스트

Acceptance Criteria:
- [ ] All 6 entity types recognized correctly
- [ ] Unit test coverage > 90%
- [ ] No regressions in existing functionality

**TAG-002: Query Expansion Pipeline** (Days 3-4)

Goal: 3단계 쿼리 확장 파이프라인 구현

Tasks:
- [ ] Day 3 AM: MultiStageQueryExpander skeleton 구현
- [ ] Day 3 PM: Stage 1 (synonym) expansion 구현
- [ ] Day 4 AM: Stage 2 (hypernym) expansion 구현
- [ ] Day 4 PM: Stage 3 (procedure) expansion 구현 및 테스트

Acceptance Criteria:
- [ ] All 3 stages work independently
- [ ] Combined expansion produces relevant terms
- [ ] Unit test coverage > 85%

### Week 2: Adaptive Top-K, Multi-Hop, Integration

**TAG-003: Adaptive Top-K** (Days 1-2)

Goal: 동적 Top-K 선택 구현

Tasks:
- [ ] Day 1 AM: Query complexity classifier 구현
- [ ] Day 1 PM: AdaptiveTopKSelector 구현
- [ ] Day 2 AM: Performance testing with different Top-K values
- [ ] Day 2 PM: Latency guardrails implementation

Acceptance Criteria:
- [ ] Correct complexity classification
- [ ] Top-K values appropriate for complexity
- [ ] Response time < 500ms maintained
- [ ] Unit test coverage > 85%

**TAG-004: Multi-Hop Retrieval** (Days 3-4)

Goal: 인용 추적 기능 구현

Tasks:
- [ ] Day 3 AM: Citation extractor 구현
- [ ] Day 3 PM: Citation graph builder 구현
- [ ] Day 4 AM: Multi-hop traversal logic 구현
- [ ] Day 4 PM: Relevance filtering and testing

Acceptance Criteria:
- [ ] Citations correctly extracted
- [ ] 2-hop traversal works without cycles
- [ ] Relevant citations only followed
- [ ] Unit test coverage > 80%

**TAG-005: Integration & Testing** (Days 5)

Goal: 모든 컴포넌트 통합 및 평가

Tasks:
- [ ] Day 5 AM: EnhancedSearch orchestration 구현
- [ ] Day 5 PM: End-to-end testing and evaluation

Acceptance Criteria:
- [ ] All components integrated successfully
- [ ] Contextual Recall improves to 50%+
- [ ] No regressions in other metrics
- [ ] Response time < 500ms

## 5. Risks and Response Plans

### Technical Risks

| Risk | Impact | Probability | Response Plan |
|------|--------|-------------|---------------|
| Entity recognition doesn't match actual queries | High | Medium | Analyze query logs first, use regex patterns with high precision |
| Query expansion creates noise | High | Medium | Use confidence thresholds, A/B test strategies |
| Adaptive Top-K increases latency | Medium | Medium | Set latency budget, implement performance monitoring |
| Multi-hop retrieval follows irrelevant citations | Medium | High | Limit to 2 hops, relevance threshold filtering |
| Implementation bugs cause regressions | High | Medium | Characterization tests, feature flags, gradual rollout |

### Quality Risks

| Risk | Impact | Probability | Response Plan |
|------|--------|-------------|---------------|
| Contextual Recall doesn't reach 50% | High | Low | Iterative enhancement, may need Alternative 3 (Aggressive) |
| Faithfulness or Answer Relevancy decreases | High | Low | Monitor metrics, rollback if regression > 5% |
| Performance degradation | Medium | Low | Latency monitoring, automatic disabling of new features |

### Mitigation Strategies

**Pre-Implementation:**
- Analyze actual query logs to validate assumptions
- Write characterization tests for existing behavior
- Implement feature flags for quick rollback

**During Implementation:**
- Daily testing to catch regressions early
- Performance monitoring with latency budgets
- Gradual rollout (canary deployment)

**Post-Implementation:**
- Monitor all metrics for 1 week
- A/B test new vs old behavior
- Collect user feedback

## 6. Approval Requests

### Decision-Making Requirements

**1. Implementation Approach: Alternative 2 (Balanced)**
- **Pros**: Addresses all root causes, moderate risk, 2-week timeline
- **Cons**: May not reach 65% target, requires careful testing
- **Recommendation**: Proceed with Alternative 2, iterate if needed

**2. Entity Recognition Patterns**
- **Option A**: Use regex patterns (faster, less accurate)
- **Option B**: Use NER model (slower, more accurate)
- **Recommendation**: Start with regex (Option A), consider NER if insufficient

**3. Top-K Values**
- **Option A**: Fixed values (5, 10, 15, 20)
- **Option B**: Dynamic calculation based on query
- **Recommendation**: Start with fixed (Option A), monitor effectiveness

### Approval Checklist

- [ ] Technology stack approved (no new external dependencies)
- [ ] TAG chain approved (5 TAGs, 2-week timeline)
- [ ] Implementation sequence approved (Week 1: ER+QE, Week 2: AT-K+MH+Integration)
- [ ] Risk response plan approved (mitigation strategies defined)
- [ ] Success criteria approved (Contextual Recall 50%+, no regressions)

## 7. Next Steps

After approval, hand over the following to workflow-ddd:

**TAG Chain**:
- TAG-001: Entity Recognition Enhancement
- TAG-002: Query Expansion Pipeline
- TAG-003: Adaptive Top-K Implementation
- TAG-004: Multi-Hop Retrieval
- TAG-005: Integration & Testing

**Key Decisions**:
- Alternative 2 (Balanced Enhancement) selected
- No new external dependencies
- 2-week implementation timeline
- Contextual Recall target: 50% (minimum), 65% (stretch goal)

**Success Criteria**:
- Contextual Recall: 32% → 50%+ (minimum), 55-60% (expected)
- Faithfulness: Maintain 51.7% → 55%+
- Answer Relevancy: Maintain 73.3% → 75%+
- Response Time: < 500ms
- Test Coverage: 85%+

**Risk Mitigation**:
- Feature flags for rollback
- Characterization tests for regression prevention
- Performance monitoring with latency budgets
- Gradual rollout plan
