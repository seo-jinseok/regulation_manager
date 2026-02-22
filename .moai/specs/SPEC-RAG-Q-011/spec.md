# SPEC-RAG-Q-011: RAG System Quality Improvement

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-Q-011 |
| **Title** | RAG System Quality Improvement |
| **Status** | Planned |
| **Priority** | Critical |
| **Created** | 2026-02-22 |
| **Source** | RAG Quality Evaluation Analysis (Sequential Thinking) |

---

## Problem Analysis

### Current Issue

RAG 시스템의 품질 평가 결과 모든 메트릭이 목표치에 미달하고 있으며, 특히 Faithfulness가 0.44로 가장 낮습니다. 근본 원인 분석 결과 Reranker 작동 실패가 연쇄적으로 전체 품질 저하를 유발하고 있습니다.

### Evidence from Evaluation

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Faithfulness | 0.44 | 0.85+ | -0.41 |
| Answer Relevancy | 0.53 | 0.85+ | -0.32 |
| Contextual Precision | 0.46 | 0.80+ | -0.34 |
| Contextual Recall | 0.87 | 0.80+ | PASS |

### Persona-Based Scores

| Persona | Score | Status |
|---------|-------|--------|
| professor | 0.56 | Below target |
| staff | 0.55 | Below target |
| freshman | 0.55 | Below target |
| graduate | 0.53 | Below target |
| international | 0.41 | LOW |
| parent | 0.39 | CRITICAL |

---

## Root Cause Analysis (Priority Levels)

### Priority 1 (PRIMARY): Reranker Failure

**Problem**: FlagEmbedding import 실패로 인한 Reranker 작동 중단

**Error Log**:
```
FlagEmbedding import failed: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'
```

**Current State**: BM25 fallback만 사용 중, 의미적 검색 불가

**Impact Chain**:
1. Reranker 미작동
2. 낮은 검색 정밀도 (Contextual Precision 0.46)
3. 관련 없는 컨텍스트가 LLM에 전달
4. LLM이 부정확한 정보로 답변 생성
5. Faithfulness 저하 (0.44)

**Additional Context**:
- "Quantization is not supported for ArchType::neon" - ARM Mac에서 양자화 미지원

### Priority 2 (SECONDARY): Evasive Response Generation

**Problem**: 충분한 정보가 없을 때 구체적 응답 대신 회피성 답변 생성

**Detected Patterns**:
- "일반적으로..." (general statements)
- "상황에 따라 다릅니다..." (situational responses)
- "자세한 내용은 담당 부서에 문의하세요..." (unnecessary deflection)

**Detection Count**:
- parent 페르소나: 2회
- professor 페르소나: 1회

**Impact**: 사용자 신뢰 저하, 답변 완결성 부족

### Priority 3 (TERTIARY): Multilingual Query Handling

**Problem**: international 페르소나의 영문/혼합 쿼리에 대한 응답 품질 불안정

**Score Variation**:
- 영문 쿼리: 0.22 ~ 0.64 (높은 변동성)
- 혼합 언어 쿼리: 불안정한 처리

**Impact**: 외국인 사용자 경험 저하

---

## Requirements (EARS Format)

### REQ-001: Reranker Restoration

**WHEN** 사용자 쿼리가 검색 시스템에 전달될 때

**THE SYSTEM SHALL** FlagEmbedding 기반 Reranker가 정상 작동

**IF** FlagEmbedging import가 실패하면

**THEN** 대체 Reranker 또는 호환 가능한 버전으로 자동 전환

**SHALL NOT** BM25 fallback만으로 검색 수행 (의미적 검색 누락)

### REQ-002: Semantic Reranking Quality

**WHEN** Reranker가 활성화된 상태에서

**IF** 검색 결과가 반환되면

**THEN** 상위 K개 결과에 대해 의미적 재순위화 수행

**AND** Contextual Precision >= 0.75 달성

### REQ-003: Response Grounding

**WHEN** LLM이 답변을 생성할 때

**IF** 충분한 컨텍스트가 검색되면

**THEN** 구체적인 인용과 함께 답변 제공

**SHALL NOT** 회피성 표현("일반적으로", "상황에 따라") 사용

### REQ-004: Evasive Response Prevention

**WHEN** 검색 결과가 충분하지 않을 때

**THE SYSTEM SHALL** 다음 중 하나로 명확히 응답:
1. "해당 규정에서 구체적인 정보를 찾을 수 없습니다"
2. "관련 정보를 찾기 위해 다른 키워드로 검색해 보세요"
3. 검색된 부분 정보라도 구체적으로 제공

**SHALL NOT** 모호한 일반화로 답변 대체

### REQ-005: Multilingual Query Support

**WHEN** 영문 또는 혼합 언어 쿼리가 입력될 때

**THE SYSTEM SHALL** 언어를 자동 감지하여 적절히 처리

**IF** international 페르소나로 평가하면

**THEN** 점수 >= 0.60 달성

---

## Acceptance Criteria

### AC-001: Reranker Functionality

- [ ] 로그에 FlagEmbedding 에러 메시지 없음
- [ ] Reranker가 정상적으로 로드됨
- [ ] BM25 + Semantic Reranking 하이브리드 검색 작동

### AC-002: Search Quality Metrics

- [ ] Contextual Precision >= 0.75 (현재 0.46)
- [ ] Faithfulness >= 0.70 (현재 0.44)
- [ ] Answer Relevancy >= 0.75 (현재 0.53)

### AC-003: Persona Score Improvement

- [ ] parent 페르소나 점수 >= 0.60 (현재 0.39)
- [ ] international 페르소나 점수 >= 0.60 (현재 0.41)
- [ ] 모든 페르소나 평균 점수 >= 0.55

### AC-004: Evasive Response Elimination

- [ ] "일반적으로", "상황에 따라" 패턴 0건
- [ ] 불필요한 "담당 부서 문의" 안내 0건
- [ ] 구체적 인용이 포함된 답변 비율 80%+

### AC-005: Test Coverage

- [ ] 신규 코드 테스트 커버리지 >= 85%
- [ ] Reranker 관련 통합 테스트 통과
- [ ] 회귀 테스트 100% 통과

---

## Technical Approach

### Phase 1: Reranker Restoration (Priority 1)

**Objective**: FlagEmbedding 호환성 문제 해결

**Approach A**: 버전 호환성 수정 (Recommended)
1. `transformers` 버전 확인 및 호환 가능한 버전으로 고정
2. `FlagEmbedding` 버전 업데이트 또는 다운그레이드
3. `is_torch_fx_available` import 우회 방안 적용

**Approach B**: 대체 Reranker 사용
1. Cohere Reranker API 활용
2. Cross-Encoder 기반 로컬 Reranker (BGE-reranker-base)
3. 혼합 언어 지원 Reranker 선택

**Approach C**: ARM Mac 호환성
1. 양자화 없이 FP16/FP32 모드로 실행
2. MPS (Metal Performance Shaders) 백엔드 활용
3. CPU 기반 폴백 구현

### Phase 2: Evasive Response Prevention (Priority 2)

**Objective**: 회피성 답변 생성 방지

**Implementation**:
1. 프롬프트 엔지니어링: 구체적 답변 유도 가이드라인 추가
2. 후처리 필터: 회피성 패턴 감지 및 대체
3. 컨텍스트 검증: 충분한 정보가 있는지 사전 확인

**Prompt Enhancement**:
```
답변 가이드라인:
- 검색된 규정 내용을 구체적으로 인용하세요
- "일반적으로", "상황에 따라" 등의 모호한 표현을 피하세요
- 정보가 부족하면 명확히 "해당 규정에서 찾을 수 없습니다"라고 답변하세요
- 부분 정보라도 제공 가능한 내용은 구체적으로 설명하세요
```

### Phase 3: Multilingual Query Optimization (Priority 3)

**Objective**: international 페르소나 응답 품질 향상

**Implementation**:
1. 언어 감지 로직 강화
2. 영문 쿼리에 대한 추가 컨텍스트 검색
3. 혼합 언어 쿼리 처리 로직 개선

---

## Out of Scope

- RAGAS 버전 업그레이드 (이미 수정됨)
- LLM 모델 변경 (현재 모델 유지)
- 벡터 데이터베이스 변경 (ChromaDB 유지)
- 새로운 페르소나 추가

---

## Dependencies

- FlagEmbedding 라이브러리
- transformers 버전 호환성
- ARM Mac (M1/M2) 실행 환경
- 기존 HallucinationFilter (SPEC-RAG-Q-002)
- 기존 CitationVerificationService (SPEC-RAG-Q-004)

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FlagEmbedding 호환성 문제 지속 | Medium | High | 대체 Reranker 준비 |
| ARM Mac에서 성능 저하 | Medium | Medium | MPS 백엔드 활용 |
| 프롬프트 변경으로 인한 부작용 | Low | Medium | A/B 테스트 후 적용 |
| 다국어 처리 정확도 저하 | Medium | Low | 언어별 독립 처리 |

---

## Related Documents

- [SPEC-RAG-Q-002: Hallucination Prevention](../SPEC-RAG-Q-002/spec.md)
- [SPEC-RAG-Q-003: Deadline Information Enhancement](../SPEC-RAG-Q-003/spec.md)
- [SPEC-RAG-Q-004: Citation Verification](../SPEC-RAG-Q-004/spec.md)
- RAG Quality Evaluation Analysis (Sequential Thinking)

---

**Last Updated:** 2026-02-22
