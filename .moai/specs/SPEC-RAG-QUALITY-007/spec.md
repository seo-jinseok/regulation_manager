# SPEC-RAG-QUALITY-007: RAG System Quality Improvement - Citation and Context Relevance Enhancement

## Metadata

| Field | Value |
|-------|-------|
| ID | SPEC-RAG-QUALITY-007 |
| Status | Draft |
| Created | 2026-02-18 |
| Priority | High |
| Source | Evaluation Analysis (eval_20260218_104503) |

---

## Problem Statement

### Current State

RAG 시스템 품질 평가 결과, 다음 핵심 메트릭이 목표 임계값을 충족하지 못함:

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Citations | 0.50 | 0.70 | -0.20 |
| Context Relevance | 0.50 | 0.75 | -0.25 |
| Overall Score | 0.697 | 0.80 | -0.103 |
| Pass Rate | 0% | 80% | -80% |

### Impact

- **Citations (0.50)**: 규정 인용이 부정확하거나 누락되어 신뢰성 저하
- **Context Relevance (0.50)**: 검색된 문서의 관련성이 낮아 적절한 답변 생성 어려움
- **Pass Rate (0%)**: 150개 쿼리 중 0개만 합격하여 서비스 품질 기준 미달

---

## Requirements (EARS Format)

### REQ-001: Citation Accuracy Enhancement

**WHEN** 시스템이 규정 관련 질문에 답변할 때
**THE SYSTEM SHALL** 답변에 포함된 모든 규정 인용이 실제 규정 문서와 일치하는지 검증
**AND** 인용이 확인되지 않은 경우 사용자에게 명시적으로 표시

**Acceptance Criteria:**
- [ ] CitationValidator가 모든 응답에서 인용 정확성 검증
- [ ] 할루시네이션된 규정 인용 0건
- [ ] Citations 점수 0.70 이상 달성

### REQ-002: Context Relevance Improvement

**WHEN** 사용자가 질문을 입력할 때
**THE SYSTEM SHALL** IntentClassifier를 통해 정확한 의도 분류 수행
**AND** 리랭킹 모델을 통해 상위 5개 문서의 관련성 점수 0.75 이상 유지

**Acceptance Criteria:**
- [ ] IntentClassifier 정확도 90% 이상
- [ ] 리랭킹된 문서의 평균 관련성 점수 0.75 이상
- [ ] Context Relevance 점수 0.75 이상 달성

### REQ-003: Pass Rate Improvement

**WHEN** 품질 평가가 실행될 때
**THE SYSTEM SHALL** 80% 이상의 쿼리가 합격 기준(Overall Score 0.80+) 달성

**Acceptance Criteria:**
- [ ] 150개 테스트 쿼리 중 120개 이상 합격
- [ ] 전체 합격률 80% 이상
- [ ] 모든 페르소나에서 균형 잡힌 성능 (편차 5% 이내)

---

## Technical Approach

### Phase 1: Citation System Enhancement

1. **CitationValidator 강화** (이미 구현된 SPEC-RAG-QUALITY-006 활용)
   - 모든 응답에 대한 인용 검증 필수화
   - 검증되지 않은 인용에 대한 경고 표시

2. **강제 인용 생성 (Forced Citation Generation)**
   - 답변 생성 시 반드시 출처 명시
   - 출처를 찾을 수 없는 경우 "해당 정보를 찾을 수 없습니다" 응답

### Phase 2: Context Relevance Improvement

1. **IntentClassifier 튜닝**
   - 다중 의도 쿼리 처리 개선
   - 애매한 쿼리에 대한 분류 정확도 향상

2. **리랭킹 모델 최적화**
   - BAAI/bge-reranker-v2-m3 성능 평가
   - 한국어 특화 리랭커(A/B 테스트) 적용

### Phase 3: Integration Testing

1. **지속적 품질 모니터링**
   - 매주 자동 평가 실행
   - 회귀 테스트를 통한 품질 유지

---

## Implementation Plan

### Sprint 1: Citation Enhancement (Week 1)

| Task | Priority | Effort |
|------|----------|--------|
| CitationValidator 적용 범위 확대 | High | 2d |
| Forced Citation Generation 구현 | High | 1d |
| 인용 검증 테스트 케이스 작성 | Medium | 1d |

### Sprint 2: Context Relevance (Week 2)

| Task | Priority | Effort |
|------|----------|--------|
| IntentClassifier 튜닝 | High | 2d |
| 리랭킹 모델 A/B 테스트 설정 | Medium | 1d |
| Context Relevance 테스트 | High | 1d |

### Sprint 3: Validation (Week 3)

| Task | Priority | Effort |
|------|----------|--------|
| 전체 평가 재실행 | High | 1d |
| 결과 분석 및 보고서 생성 | Medium | 1d |
| 문서화 및 SPEC 종료 | Low | 0.5d |

---

## Success Metrics

| Metric | Before | After (Target) | Measurement |
|--------|--------|----------------|-------------|
| Citations | 0.50 | 0.70+ | LLM-as-Judge |
| Context Relevance | 0.50 | 0.75+ | RAGAS |
| Overall Score | 0.697 | 0.80+ | Weighted Average |
| Pass Rate | 0% | 80%+ | Evaluation |

---

## Dependencies

- SPEC-RAG-QUALITY-006: IntentClassifier and CitationValidator (COMPLETED)

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 리랭킹 모델 변경 시 latency 증가 | Medium | Medium | 캐싱 및 배치 처리 |
| IntentClassifier 오버피팅 | Low | High | 교차 검증 및 다양한 테스트 케이스 |
| 할루시네이션 완전 제거 어려움 | Medium | High | 강제 인용 정책 적용 |

---

## References

- Evaluation Data: `data/evaluations/rag_quality_full_eval_20260218_104545.json`
- Previous SPEC: SPEC-RAG-QUALITY-006
- Skill: `rag-quality-local`
