# Implementation Plan: SPEC-RAG-Q-011

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-Q-011 |
| **Title** | RAG System Quality Improvement |
| **Development Mode** | Hybrid (TDD for new code, DDD for modifications) |

---

## Implementation Milestones

### Milestone 1: Reranker Diagnosis and Fix (Priority: Critical)

**Goal**: FlagEmbedding Reranker 정상 작동 복구

**Tasks**:
1. [ ] 현재 FlagEmbedding import 에러 원인 분석
2. [ ] transformers와 FlagEmbedding 버전 호환성 매트릭스 작성
3. [ ] 호환 가능한 버전 조합 식별
4. [ ] ARM Mac 양자화 미지원 문제 해결 방안 적용
5. [ ] Reranker 로드 테스트 작성 및 통과

**Deliverables**:
- `src/rag/infrastructure/reranker/` 수정
- Reranker 로드 성공 로그 확인
- 통합 테스트 통과

**Success Criteria**:
- FlagEmbedding 에러 메시지 0건
- Reranker 정상 로드 확인
- 기존 검색 기능 정상 작동

### Milestone 2: Semantic Search Quality Verification (Priority: High)

**Goal**: Reranker 복구 후 검색 품질 메트릭 달성

**Tasks**:
1. [ ] Contextual Precision 측정 테스트 작성
2. [ ] BM25 + Reranker 하이브리드 검색 검증
3. [ ] 상위 K개 결과 재순위화 로직 검증
4. [ ] 검색 품질 메트릭 자동 수집 구현

**Deliverables**:
- 검색 품질 측정 스크립트
- Contextual Precision >= 0.75 달성

**Success Criteria**:
- Contextual Precision >= 0.75
- Faithfulness >= 0.70 (Reranker 효과로 인한 향상)

### Milestone 3: Evasive Response Prevention (Priority: High)

**Goal**: 회피성 답변 패턴 제거

**Tasks**:
1. [ ] 회피성 패턴 감지기 구현
2. [ ] 프롬프트에 구체적 답변 가이드라인 추가
3. [ ] 후처리 필터로 회피성 표현 대체
4. [ ] parent/professor 페르소나 테스트 케이스 작성

**Deliverables**:
- `src/rag/infrastructure/evasive_response_filter.py`
- 프롬프트 업데이트
- 회피성 응답 0건 달성

**Success Criteria**:
- "일반적으로", "상황에 따라" 패턴 0건
- 구체적 인용 포함 답변 비율 80%+

### Milestone 4: Multilingual Query Optimization (Priority: Medium)

**Goal**: international 페르소나 응답 품질 향상

**Tasks**:
1. [ ] 언어 감지 로직 강화
2. [ ] 영문 쿼리 추가 컨텍스트 검색 구현
3. [ ] 혼합 언어 쿼리 처리 로직 개선
4. [ ] international 페르소나 평가 테스트

**Deliverables**:
- 언어 감지 개선
- international 페르소나 점수 >= 0.60

**Success Criteria**:
- international 페르소나 점수 >= 0.60
- 영문/혼합 쿼리 변동성 감소

### Milestone 5: Final Validation and Integration (Priority: High)

**Goal**: 전체 품질 메트릭 달성 및 통합 테스트

**Tasks**:
1. [ ] 전체 페르소나 평가 실행
2. [ ] 회귀 테스트 수행
3. [ ] 성능 벤치마크 측정
4. [ ] 문서화 업데이트

**Deliverables**:
- 품질 평가 보고서
- 회귀 테스트 100% 통과
- 테스트 커버리지 >= 85%

**Success Criteria**:
- 모든 AC 충족
- 회귀 테스트 통과
- 커버리지 >= 85%

---

## File Changes Plan

### New Files

| File | Description |
|------|-------------|
| `src/rag/infrastructure/evasive_response_filter.py` | 회피성 응답 감지 및 대체 |
| `tests/rag/unit/infrastructure/test_evasive_response_filter.py` | 회피성 응답 필터 테스트 |

### Modified Files

| File | Changes |
|------|---------|
| `src/rag/infrastructure/reranker/reranker.py` | FlagEmbedding 호환성 수정 |
| `src/rag/config.py` | Reranker 설정 추가 |
| `src/rag/application/search_usecase.py` | 회피성 응답 필터 통합 |
| `data/prompts/prompts.json` | 구체적 답변 가이드라인 추가 |
| `src/rag/infrastructure/language_detector.py` | 언어 감지 로직 강화 |

---

## Testing Strategy

### Unit Tests (TDD for New Code)

- Reranker 로드 테스트
- 회피성 패턴 감지 테스트
- 언어 감지 테스트

### Integration Tests

- 검색 품질 메트릭 측정
- 페르소나별 평가
- End-to-end RAG 파이프라인

### Characterization Tests (DDD for Existing Code)

- 기존 검색 기능 보존
- Reranker 변경 전후 비교

---

## Rollback Plan

### Reranker Failure

```python
# Fallback to BM25 only if Reranker fails
if not reranker_available:
    logger.warning("Reranker unavailable, using BM25 fallback")
    return bm25_results
```

### Performance Degradation

- 이전 버전으로 빠른 롤백 가능하도록 feature flag 사용
- `ENABLE_SEMANTIC_RERANKING` 환경 변수로 제어

---

## Dependencies

### Internal

- SPEC-RAG-Q-002 (Hallucination Prevention)
- SPEC-RAG-Q-004 (Citation Verification)
- SearchUsecase

### External

- FlagEmbedding >= 1.2.0
- transformers >= 4.40.0 (호환 가능 버전)
- torch >= 2.0.0

---

## Environment Configuration

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_MODE` | `auto` | auto, semantic, bm25_only |
| `RERANKER_FALLBACK_ENABLED` | `true` | Reranker 실패 시 BM25 폴백 |
| `EVASIVE_RESPONSE_FILTER_ENABLED` | `true` | 회피성 응답 필터 활성화 |
| `EVASIVE_RESPONSE_FILTER_MODE` | `replace` | warn, replace, block |

---

## Progress Tracking

| Milestone | Status | Completion |
|-----------|--------|------------|
| M1: Reranker Fix | Not Started | 0% |
| M2: Search Quality | Not Started | 0% |
| M3: Evasive Response | Not Started | 0% |
| M4: Multilingual | Not Started | 0% |
| M5: Final Validation | Not Started | 0% |

---

**Last Updated:** 2026-02-22
