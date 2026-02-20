# Acceptance Criteria: SPEC-RAG-QUALITY-007

## Metadata

| Field | Value |
|-------|-------|
| SPEC ID | SPEC-RAG-QUALITY-007 |
| Created | 2026-02-19 |
| Status | Draft |
| Version | 1.0 |

---

## Acceptance Criteria

### AC-001: Citation Accuracy Enhancement

**Given:** 시스템이 규정 관련 질문에 답변함
**When:** 답변이 생성될 때
**Then:** 모든 규정 인용이 실제 규정 문서와 일치해야 함

**Acceptance Tests:**

- [ ] **AT-001-1: Citation Format Validation**
  - Input: "복학 신청 방법"
  - Expected: 답변에 "「복학규정」 제X조" 형식 인용 포함
  - Validation: `grep "「.*」 제[0-9]+조" response`

- [ ] **AT-001-2: No Hallucinated Citations**
  - Input: 50개 다양한 쿼리
  - Expected: 할루시네이션된 인용 0건
  - Validation: CitationValidator.validate_answer() 결과에서 HALUCINATED status 0건

- [ ] **AT-001-3: Citation Score Target**
  - Input: 150개 평가 쿼리
  - Expected: Citations 점수 >= 0.70
  - Measurement: RAGAS LLM-as-Judge

**Failure Criteria:**
- Citations 점수 < 0.70
- 할루시네이션된 인용 발견
- 인용 형식이 "「규정명」 제X조" 패턴 불일치

---

### AC-002: Context Relevance Improvement

**Given:** 사용자가 질문을 입력함
**When:** 검색이 수행될 때
**Then:** 상위 문서들의 관련성 점수가 0.75 이상이어야 함

**Acceptance Tests:**

- [ ] **AT-002-1: IntentClassifier Accuracy**
  - Input: 100개 라벨링된 쿼리
  - Expected: IntentClassifier 정확도 >= 90%
  - Categories: PROCEDURE, ELIGIBILITY, DEADLINE, GENERAL

- [ ] **AT-002-2: Reranker Threshold Effectiveness**
  - Input: 50개 테스트 쿼리
  - Expected: 평균 관련성 점수 >= 0.75
  - Threshold: MIN_RELEVANCE_THRESHOLD = 0.25

- [ ] **AT-002-3: Context Relevance Score**
  - Input: 150개 평가 쿼리
  - Expected: Context Relevance 점수 >= 0.75
  - Measurement: RAGAS Context Precision

**Test Cases:**

```gherkin
Scenario: 절차 관련 쿼리의 컨텍스트 관련성
  Given 사용자가 "복학 신청 방법"을 질문함
  When IntentClassifier가 PROCEDURE로 분류함
  Then 검색 결과 top-15 문서가 반환됨
  And 평균 관련성 점수가 0.75 이상임
```

```gherkin
Scenario: 기한 관련 쿼리의 컨텍스트 관련성
  Given 사용자가 "등록금 납부 언제까지"를 질문함
  When IntentClassifier가 DEADLINE으로 분류함
  Then 검색 결과에 날짜/기간 관련 문서가 부스팅됨
  And 평균 관련성 점수가 0.75 이상임
```

**Failure Criteria:**
- IntentClassifier 정확도 < 90%
- Context Relevance 점수 < 0.75
- 평균 관련성 점수 < 0.75

---

### AC-003: Pass Rate Improvement

**Given:** 품질 평가가 실행됨
**When:** 150개 테스트 쿼리가 평가될 때
**Then:** 80% 이상의 쿼리가 합격 기준을 달성해야 함

**Acceptance Tests:**

- [ ] **AT-003-1: Overall Pass Rate**
  - Input: 150개 테스트 쿼리
  - Expected: 120개 이상 합격 (80%+)
  - Criteria: Overall Score >= 0.80

- [ ] **AT-003-2: Persona Balance**
  - Input: 10개 페르소나 × 15개 쿼리
  - Expected: 페르소나별 합격률 편차 < 5%
  - Personas: freshman, transfer, senior, graduate, professor, staff, international, adjunct, researcher, admin

- [ ] **AT-003-3: Metric Thresholds**
  - Expected: 모든 메트릭이 개별 임계값 통과
    - Faithfulness >= 0.6
    - Answer Relevancy >= 0.7
    - Contextual Precision >= 0.65
    - Contextual Recall >= 0.75

**Failure Criteria:**
- Pass Rate < 80%
- 페르소나별 합격률 편차 >= 5%
- 어느 하나의 메트릭이라도 임계값 미달

---

### AC-004: IntentClassifier Integration

**Given:** IntentClassifier가 구현됨
**When:** 검색 파이프라인이 실행될 때
**Then:** 의도 분류가 검색 파라미터에 반영되어야 함

**Acceptance Tests:**

- [ ] **AT-004-1: Integration in Search Pipeline**
  - Location: `src/rag/application/search_usecase.py`
  - Expected: `_get_intent_aware_search_params()` 메서드 존재
  - Expected: IntentClassifier.classify() 호출 확인

- [ ] **AT-004-2: Intent-Based Search Parameters**
  - Input:
    - PROCEDURE 쿼리 → top_k=15
    - DEADLINE 쿼리 → date/period boost
    - ELIGIBILITY 쿼리 → condition boost
  - Expected: 검색 파라미터가 의도에 따라 동적 조정됨

- [ ] **AT-004-3: Performance Impact**
  - Expected: IntentClassifier 처리 시간 < 10ms
  - Expected: 전체 검색 latency 증가 < 50ms

**Failure Criteria:**
- IntentClassifier 미통합
- 의도 기반 파라미터 조정 미작동
- Latency 증가 >= 100ms

---

### AC-005: Reranker Threshold Optimization

**Given:** Reranker가 구현됨
**When:** 문서 재랭킹이 수행될 때
**Then:** 낮은 관련성 문서가 필터링되어야 함

**Acceptance Tests:**

- [ ] **AT-005-1: Threshold Adjustment**
  - Location: `src/rag/infrastructure/reranker.py`
  - Expected: MIN_RELEVANCE_THRESHOLD = 0.25 (0.15에서 상향)
  - Validation: grep "MIN_RELEVANCE_THRESHOLD = 0.25" reranker.py

- [ ] **AT-005-2: Relevance Score Logging**
  - Expected: 각 쿼리에 대한 관련성 점수 분포 로깅
  - Log format: "avg_score=X.XXX, min_score=X.XXX, max_score=X.XXX, filtered=N"

- [ ] **AT-005-3: Filtering Effectiveness**
  - Input: 50개 테스트 쿼리
  - Expected: 평균 필터링 비율 10-30%
  - Expected: 필터링 후 Context Relevance 점수 향상

**Failure Criteria:**
- MIN_RELEVANCE_THRESHOLD 미조정
- 로깅 미추가
- 필터링 효과 없음 (Context Relevance 점수 변화 < 0.05)

---

## Test Scenarios

### Scenario 1: End-to-End Citation Flow

```gherkin
Feature: Citation Generation and Validation

  Scenario: LLM generates answer without citations
    Given 사용자가 "휴학 신청 조건"을 질문함
    And LLM이 인용 없이 답변을 생성함
    When _validate_and_enrich_citations()가 실행됨
    Then 강제 인용 생성이 트리거됨
    And 답변에 "「휴학규정」 제X조" 형식 인용이 추가됨
    And citation_count >= 1

  Scenario: LLM generates answer with valid citations
    Given 사용자가 "등록금 납부 기한"을 질문함
    And LLM이 "「등록금규정」 제5조" 인용과 함께 답변함
    When _validate_and_enrich_citations()가 실행됨
    Then 인용 형식이 검증됨
    And citation_density가 로깅됨
    And 답변이 수정되지 않음 (이미 유효함)
```

### Scenario 2: Intent-Based Search

```gherkin
Feature: Intent-Aware Search

  Scenario: Procedure query gets extended context
    Given 사용자가 "졸업 요건 어떻게 확인해"를 질문함
    When IntentClassifier가 PROCEDURE로 분류함 (confidence >= 0.7)
    Then top_k가 15로 설정됨
    And procedure boost factor가 1.5로 적용됨
    And 검색 결과에 절차 관련 문서가 상위에 위치함

  Scenario: Deadline query gets date boosting
    Given 사용자가 "성적 입력 언제까지야"를 질문함
    When IntentClassifier가 DEADLINE으로 분류함 (confidence >= 0.8)
    Then date와 period boost factor가 1.3으로 적용됨
    And 검색 결과에 기한/날짜 관련 문서가 부스팅됨
```

### Scenario 3: Reranker Threshold

```gherkin
Feature: Reranker Relevance Filtering

  Scenario: Low relevance documents filtered out
    Given 20개 문서가 초기 검색됨
    And 재랭킹 후 점수 분포가 [0.12, 0.18, 0.25, 0.32, 0.45, ...]
    When MIN_RELEVANCE_THRESHOLD=0.25가 적용됨
    Then 점수 < 0.25인 2개 문서가 필터링됨
    And 상위 문서들의 평균 점수 >= 0.35
    And Context Relevance 점수가 향상됨
```

---

## Evaluation Checklist

### Pre-Implementation
- [ ] Baseline metrics 측정 (50개 쿼리)
- [ ] Current IntentClassifier accuracy 측정
- [ ] Current reranker score distribution 분석

### During Implementation
- [ ] IntentClassifier 통합 unit test 통과
- [ ] Reranker threshold 조정 unit test 통과
- [ ] Integration test 통과 (10개 쿼리)

### Post-Implementation
- [ ] 50개 쿼리 평가 - 중간 검증
- [ ] 150개 쿼리 전체 평가
- [ ] Persona별 균형 검증
- [ ] Latency 영향 측정

---

## Quality Gates

### Gate 1: IntentClassifier Integration
- IntentClassifier 정확도 >= 90%
- 통합 테스트 통과
- Latency 증가 < 50ms

### Gate 2: Reranker Optimization
- MIN_RELEVANCE_THRESHOLD = 0.25
- 로깅 정상 작동
- Context Relevance 점수 향상 (>= 0.70 중간 목표)

### Gate 3: Full Evaluation
- Citations >= 0.70
- Context Relevance >= 0.75
- Pass Rate >= 80%
- Persona 편차 < 5%

---

## Rollback Criteria

**Critical Failure (즉시 롤백):**
- Pass Rate < 60%
- Context Relevance < 0.60
- Latency 증가 >= 200ms

**Performance Degradation (조사 후 결정):**
- Pass Rate 60-70%
- Context Relevance 0.65-0.75
- 일부 페르소나에서 심각한 저하 (편차 >= 10%)

---

## Sign-Off Requirements

- [ ] 모든 Acceptance Tests 통과
- [ ] Quality Gates 통과
- [ ] Performance regression 없음
- [ ] Documentation 업데이트 완료
- [ ] Code review 승인
- [ ] Stakeholder approval

---

## References

- SPEC Document: `.moai/specs/SPEC-RAG-QUALITY-007/spec.md`
- Implementation Plan: `.moai/specs/SPEC-RAG-QUALITY-007/plan.md`
- Test Data: `data/evaluations/evaluation_20260219_210750.json`
- RAGAS Metrics: https://docs.ragas.io/
