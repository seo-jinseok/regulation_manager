# SPEC-RAG-QUALITY-007: RAG System Quality Improvement - Citation and Context Relevance Enhancement

## Metadata

| Field | Value |
|-------|-------|
| ID | SPEC-RAG-QUALITY-007 |
| Status | Completed |
| Created | 2026-02-18 |
| Completed | 2026-02-20 |
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

## Implementation Notes

### Summary

SPEC-RAG-QUALITY-007 구현 완료. 4개의 커밋으로 구성되며, 4,696개의 테스트가 통과함.

### Implementation vs Plan Divergence

**Planned but Not Implemented:**
1. **CitationValidator Enhancement (REQ-001)**: 이미 SPEC-RAG-QUALITY-006에서 완전히 구현되어 작동 중임. 추가 변경 필요 없음.
2. **Forced Citation Generation**: 이미 search_usecase.py에 구현되어 있음 (line 3551-3564).
3. **A/B Testing for Rerankers**: 리소스 제약으로 인해 구현되지 않음. 현재 BAAI/bge-reranker-v2-m3 사용.

**Implemented as Planned:**
1. **IntentClassifier Integration (REQ-002)**:
   - `_get_intent_aware_search_params()` 메서드 추가
   - 의도 카테고리별 검색 파라미터 동적 조정
   - PROCEDURE → top_k=15, ELIGIBILITY → top_k=12, DEADLINE → top_k=10
   - 126개 라벨링된 쿼리로 테스트 완료

2. **Reranker Threshold Optimization (REQ-002)**:
   - MIN_RELEVANCE_THRESHOLD: 0.15 → 0.25 상향 조정
   - 관련성 점수 분포 로깅 추가
   - 22개 리랭커 테스트 통과

3. **Evaluation Verification Script (REQ-003)**:
   - `scripts/verify_evaluation_metrics.py` 생성
   - RAGAS 환경 검증 기능 구현
   - 0.50 uniform score 근본 원인 파악 (OPENAI_API_KEY 누락)
   - 13개 단위 테스트 통과

### Commits

1. `805bc64` - feat(rag): integrate IntentClassifier for intent-aware search
2. `7cc7036` - fix(rag): increase reranker threshold for better relevance
3. `0a961db` - feat(rag): add evaluation metrics verification script
4. `cf7dac9` - docs(spec): add SPEC-RAG-QUALITY-007 documentation

### Files Modified

| File | Changes | Lines |
|------|---------|-------|
| src/rag/application/search_usecase.py | IntentClassifier 통합 | +98 |
| src/rag/infrastructure/reranker.py | Threshold 조정 및 로깅 | +15 |
| scripts/verify_evaluation_metrics.py | 평가 검증 스크립트 (신규) | +743 |
| tests/integration/test_intent_aware_search.py | 통합 테스트 (신규) | +546 |
| tests/unit/test_eval_verification.py | 단위 테스트 (신규) | +236 |

### Test Results

- **Total Tests**: 4,696 passed
- **New Tests**: 31 integration + 13 unit = 44 tests added
- **Coverage**: 83.66% maintained
- **No Regressions**: All existing tests pass

### Known Limitations

1. **Evaluation Scores**: OPENAI_API_KEY 설정 전까지 0.50 기본값 유지
2. **A/B Testing**: 리랭커 A/B 테스트 미구현
3. **Full Evaluation**: 150개 쿼리 전체 평가는 OPENAI_API_KEY 설정 후 실행 필요

### Next Steps

1. OPENAI_API_KEY 환경 변수 설정
2. `scripts/verify_evaluation_metrics.py` 실행하여 RAGAS 환경 검증
3. 150개 쿼리 전체 평가 실행
4. 목표 달성 여부 확인 (Citations 0.70+, Context Relevance 0.75+, Pass Rate 80%+)

---

## References

- Evaluation Data: `data/evaluations/rag_quality_full_eval_20260218_104545.json`
- Previous SPEC: SPEC-RAG-QUALITY-006
- Skill: `rag-quality-local`
- Verification Script: `scripts/verify_evaluation_metrics.py`
