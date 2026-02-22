# Acceptance Criteria: SPEC-RAG-Q-011

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-Q-011 |
| **Title** | RAG System Quality Improvement |
| **Test Strategy** | Hybrid (TDD + DDD) |

---

## Acceptance Test Scenarios

### AT-001: Reranker Functionality Verification

**Given**: RAG 시스템이 초기화될 때
**When**: Reranker 모듈이 로드됨
**Then**: FlagEmbedding 에러 없이 정상 로드되어야 함

```gherkin
Scenario: Reranker loads without FlagEmbedding error
  Given the RAG system is initialized
  When the reranker module loads
  Then no FlagEmbedding import error should occur
  And the reranker should be ready for semantic reranking
```

**Verification Method**:
- [ ] 로그 파일에서 FlagEmbedding 에러 메시지 검색 (0건)
- [ ] Reranker 상태 API 응답 확인 (`status: "healthy"`)

---

### AT-002: Semantic Reranking Quality

**Given**: 사용자가 검색 쿼리를 입력할 때
**When**: 검색 결과가 반환됨
**Then**: Contextual Precision >= 0.75 달성

```gherkin
Scenario: Semantic reranking improves search precision
  Given a user submits a query "휴학 신청 방법"
  When the search pipeline executes
  Then BM25 retrieval returns candidate documents
  And semantic reranking reorders the candidates
  And contextual precision should be >= 0.75
```

**Test Cases**:

| Query | Expected Precision | Test Status |
|-------|-------------------|-------------|
| "휴학 신청 방법" | >= 0.75 | [ ] |
| "등록금 납부 기한" | >= 0.75 | [ ] |
| "졸업 요건" | >= 0.75 | [ ] |
| "연구년 제도" | >= 0.75 | [ ] |
| "수강신청 기간" | >= 0.75 | [ ] |

---

### AT-003: Faithfulness Improvement

**Given**: Reranker가 정상 작동할 때
**When**: LLM이 답변을 생성함
**Then**: Faithfulness >= 0.70 달성

```gherkin
Scenario: Reranker improves response faithfulness
  Given the reranker is functioning correctly
  When a user asks "등록금 분할납부 가능한가요?"
  Then the retrieved context should be relevant
  And the LLM response should be grounded in the context
  And faithfulness score should be >= 0.70
```

**Verification Method**:
- [ ] RAGAS Faithfulness 메트릭 측정
- [ ] 5개 이상의 테스트 쿼리로 평균 계산

---

### AT-004: Evasive Response Elimination

**Given**: 답변 생성 파이프라인이 실행될 때
**When**: 충분한 컨텍스트가 검색됨
**Then**: 회피성 표현 없이 구체적 답변 제공

```gherkin
Scenario: No evasive responses with sufficient context
  Given sufficient context is retrieved
  When the LLM generates a response
  Then the response should not contain "일반적으로"
  And the response should not contain "상황에 따라 다릅니다"
  And the response should include specific citations
```

**Evasive Pattern Detection**:

| Pattern | Detection Status | Test Status |
|---------|-----------------|-------------|
| "일반적으로" | Detected | [ ] Not present |
| "상황에 따라" | Detected | [ ] Not present |
| "담당 부서에 문의" (불필요한 경우) | Detected | [ ] Not present |
| "자세한 내용은" + 구체적 정보 없음 | Detected | [ ] Not present |

**Test Queries for Evasive Response**:

| Query | Expected Behavior | Test Status |
|-------|-------------------|-------------|
| "휴학은 몇 년까지 가능한가요?" | 구체적 연한 명시 | [ ] |
| "등록금 환불 규정이 있나요?" | 환불 조건 및 비율 명시 | [ ] |
| "졸업 필요 학점이 어떻게 되나요?" | 학점 기준 명시 | [ ] |

---

### AT-005: Parent Persona Score Improvement

**Given**: parent 페르소나로 평가할 때
**When**: 질문에 대한 답변이 생성됨
**Then**: 점수 >= 0.60 달성

```gherkin
Scenario: Parent persona achieves target score
  Given the parent persona is active
  When evaluating responses about student policies
  Then the average score should be >= 0.60
  And no evasive responses should be detected
```

**Test Cases (Parent Persona)**:

| Query | Expected Score | Test Status |
|-------|---------------|-------------|
| "자녀가 휴학하려는데 절차가 어떻게 되나요?" | >= 0.60 | [ ] |
| "등록금 납부 방법이 어떻게 되나요?" | >= 0.60 | [ ] |
| "장학금 신청 자격이 어떻게 되나요?" | >= 0.60 | [ ] |

---

### AT-006: International Persona Score Improvement

**Given**: international 페르소나로 평가할 때
**When**: 영문 또는 혼합 언어 쿼리가 입력됨
**Then**: 점수 >= 0.60 달성

```gherkin
Scenario: International persona achieves target score
  Given the international persona is active
  When a user submits queries in English or mixed language
  Then the language should be detected correctly
  And the response quality score should be >= 0.60
```

**Test Cases (International Persona)**:

| Query (Language) | Expected Score | Test Status |
|------------------|---------------|-------------|
| "How do I apply for leave of absence?" (EN) | >= 0.60 | [ ] |
| "What are the graduation requirements?" (EN) | >= 0.60 | [ ] |
| "휴학 how to apply?" (Mixed) | >= 0.60 | [ ] |
| "등록금 tuition payment methods?" (Mixed) | >= 0.60 | [ ] |

---

### AT-007: Answer Relevancy Improvement

**Given**: 검색 결과가 적절히 순위화될 때
**When**: LLM이 답변을 생성함
**Then**: Answer Relevancy >= 0.75 달성

```gherkin
Scenario: Answer relevancy meets target
  Given properly reranked search results
  When the LLM generates a response
  Then the response should directly address the query
  And answer relevancy should be >= 0.75
```

**Verification Method**:
- [ ] RAGAS Answer Relevancy 메트릭 측정
- [ ] 10개 테스트 쿼리로 평균 계산

---

### AT-008: Test Coverage Compliance

**Given**: 모든 구현이 완료될 때
**When**: 테스트 스위트가 실행됨
**Then**: 커버리지 >= 85% 달성

```gherkin
Scenario: Test coverage meets target
  Given all implementation is complete
  When the test suite runs
  Then unit test coverage should be >= 85%
  And all integration tests should pass
  And no regression in existing tests
```

**Coverage Targets**:

| Component | Target Coverage | Test Status |
|-----------|----------------|-------------|
| `evasive_response_filter.py` | >= 90% | [ ] |
| `reranker.py` (modified) | >= 85% | [ ] |
| `language_detector.py` (modified) | >= 85% | [ ] |
| Overall | >= 85% | [ ] |

---

### AT-009: No Regression in Existing Features

**Given**: 기존 SPEC들이 구현된 상태
**When**: 새로운 변경사항이 적용됨
**Then**: 기존 기능이 정상 작동해야 함

```gherkin
Scenario: No regression in existing features
  Given SPEC-RAG-Q-002, Q-003, Q-004 are implemented
  When the new changes are applied
  Then hallucination prevention should still work
  And citation verification should still work
  And deadline information enhancement should still work
```

**Regression Test Checklist**:

| Feature | Test Status |
|---------|-------------|
| Hallucination Prevention (SPEC-RAG-Q-002) | [ ] Pass |
| Deadline Information (SPEC-RAG-Q-003) | [ ] Pass |
| Citation Verification (SPEC-RAG-Q-004) | [ ] Pass |

---

## Quality Gate Checklist

### Pre-Implementation

- [ ] FlagEmbedding 버전 호환성 확인
- [ ] ARM Mac 테스트 환경 준비
- [ ] 기존 테스트 스위트 통과

### During Implementation

- [ ] TDD 사이클 준수 (RED-GREEN-REFACTOR)
- [ ] 각 마일스톤 완료 시 통합 테스트 실행
- [ ] 코드 리뷰 완료

### Post-Implementation

- [ ] 모든 AC 테스트 통과
- [ ] 커버리지 >= 85% 달성
- [ ] 성능 벤치마크 통과
- [ ] 회귀 테스트 100% 통과

---

## Test Execution Commands

```bash
# Run unit tests with coverage
pytest tests/rag/unit/ -v --cov=src/rag --cov-report=term-missing

# Run integration tests
pytest tests/rag/integration/ -v

# Run persona evaluation
python scripts/evaluate_rag_quality.py --persona all

# Check for regression
pytest tests/rag/ -v --tb=short
```

---

## Success Metrics Summary

| Metric | Current | Target | Test Method |
|--------|---------|--------|-------------|
| Reranker Status | Failed | Healthy | Log inspection |
| Contextual Precision | 0.46 | >= 0.75 | RAGAS evaluation |
| Faithfulness | 0.44 | >= 0.70 | RAGAS evaluation |
| Answer Relevancy | 0.53 | >= 0.75 | RAGAS evaluation |
| Parent Score | 0.39 | >= 0.60 | Persona evaluation |
| International Score | 0.41 | >= 0.60 | Persona evaluation |
| Evasive Responses | 3 | 0 | Pattern detection |
| Test Coverage | - | >= 85% | pytest-cov |

---

**Last Updated:** 2026-02-22
