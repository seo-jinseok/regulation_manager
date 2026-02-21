# Acceptance Criteria: SPEC-RAG-QUALITY-008

## Faithfulness Enhancement for RAG System

**SPEC ID**: SPEC-RAG-QUALITY-008
**Created**: 2026-02-20

---

## Acceptance Criteria Overview

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| AC-001 | Faithfulness >= 60% | Critical | Pending |
| AC-002 | Recall >= 80% maintained | High | Pending |
| AC-003 | Pass Rate >= 50% | High | Pending |
| AC-004 | Answer Relevancy >= 70% | Medium | Pending |
| AC-005 | Contextual Precision >= 65% | Medium | Pending |

---

## Detailed Acceptance Criteria

### AC-001: Faithfulness Score >= 60%

**Description**: 생성된 답변의 Faithfulness 점수가 60% 이상이어야 함

**Given-When-Then Scenarios**:

```
Scenario 1: High Faithfulness Answer Generation
GIVEN 사용자가 "휴학 신청 기간은 언제인가요?"라고 질문하고
AND 검색된 컨텍스트에 "학칙 제40조: 휴학은 학기 시작 30일 전까지 신청"이 포함되어 있고
WHEN 시스템이 답변을 생성하면
THEN 답변의 Faithfulness 점수는 0.6 이상이어야 한다
AND 답변은 "30일 전까지"라는 컨텍스트 내용을 포함해야 한다
AND 답변에 컨텍스트에 없는 전화번호가 포함되지 않아야 한다
```

```
Scenario 2: Low Faithfulness Detection and Regeneration
GIVEN 사용자가 질문하고
AND 검색된 컨텍스트가 질문에 충분히 답할 수 없고
WHEN 첫 번째 답변의 Faithfulness가 0.6 미만이면
THEN 시스템은 더 엄격한 프롬프트로 답변을 재생성해야 한다
AND 최대 2회까지 재시도해야 한다
```

```
Scenario 3: Fallback Response for Ungroundable Questions
GIVEN 사용자가 "2025년 등록금은 얼마인가요?"라고 질문하고
AND 검색된 컨텍스트에 등록금 금액 정보가 없고
WHEN 모든 재시도 후에도 Faithfulness가 0.6 미만이면
THEN 시스템은 "제공된 규정에서 해당 정보를 찾을 수 없습니다" 응답을 반환해야 한다
AND 응답에 관련 부서 문의 안내가 포함되어야 한다
```

**Test Cases**:

| Test ID | Input Query | Expected Faithfulness | Expected Behavior |
|---------|-------------|----------------------|-------------------|
| TC-001 | "휴학 신청 기간" | >= 0.6 | 컨텍스트 기반 답변 |
| TC-002 | "장학금 문의처" | >= 0.6 (with regen) | 재생성 후 적절한 답변 또는 fallback |
| TC-003 | "등록금 납부 방법" | >= 0.6 | 컨텍스트 기반 답변 |
| TC-004 | "교원 연봉 인상률" | N/A | Fallback (컨텍스트 부족) |

**Measurement Method**:
```bash
# RAGAS Faithfulness 메트릭으로 측정
python scripts/verify_evaluation_metrics.py --metric faithfulness

# 평가 결과에서 Faithfulness 평균 확인
# 목표: mean(faithfulness) >= 0.60
```

---

### AC-002: Contextual Recall >= 80% Maintained

**Description**: Recall 점수가 현재 87% 수준에서 80% 이상을 유지해야 함

**Given-When-Then Scenario**:

```
Scenario: Recall Maintenance During Faithfulness Improvement
GIVEN 현재 시스템의 Recall이 87%이고
WHEN Faithfulness 개선을 위한 변경이 적용되면
THEN Recall 점수는 80% 이상을 유지해야 한다
AND Recall 저하가 7%p 이내여야 한다 (87% → 80%+)
```

**Test Cases**:

| Test ID | Input Query | Expected Recall | Notes |
|---------|-------------|-----------------|-------|
| TC-101 | "자녀 장학금" | >= 0.8 | 다중 컨텍스트 필요 |
| TC-102 | "등록금 납부" | >= 0.8 | 관련 규정 다수 |
| TC-103 | "교원 휴직" | >= 0.8 | 복합 조건 |

**Measurement Method**:
```bash
# RAGAS ContextRecall 메트릭으로 측정
python scripts/verify_evaluation_metrics.py --metric recall

# 평가 결과에서 Recall 평균 확인
# 목표: mean(contextual_recall) >= 0.80
```

---

### AC-003: Overall Pass Rate >= 50%

**Description**: 전체 평가 쿼리의 50% 이상이 합격 기준을 달성해야 함

**Given-When-Then Scenario**:

```
Scenario: Overall Pass Rate Improvement
GIVEN 164개 평가 쿼리가 존재하고
WHEN Faithfulness 개선 후 평가를 실행하면
THEN 최소 82개 쿼리 (50%)가 합격해야 한다
AND 합격 기준은 모든 메트릭이 임계값 이상인 것
```

**Pass Criteria per Query**:
- Faithfulness >= 0.60
- Answer Relevancy >= 0.70
- Contextual Precision >= 0.65
- Contextual Recall >= 0.80

**Measurement Method**:
```bash
# 전체 평가 실행
python scripts/verify_evaluation_metrics.py --full-eval

# 합격률 확인
# Pass Rate = (passed_queries / total_queries) * 100
# 목표: Pass Rate >= 50%
```

---

### AC-004: Answer Relevancy >= 70%

**Description**: 답변의 질문 관련성이 70% 이상이어야 함

**Given-When-Then Scenario**:

```
Scenario: Answer Relevancy Maintenance
GIVEN 사용자가 구체적인 질문을 하고
WHEN 시스템이 답변을 생성하면
THEN 답변은 질문에 직접적으로 관련되어야 한다
AND Answer Relevancy 점수는 0.7 이상이어야 한다
```

**Measurement Method**:
```bash
# RAGAS AnswerRelevancy 메트릭으로 측정
python scripts/verify_evaluation_metrics.py --metric relevancy

# 목표: mean(answer_relevancy) >= 0.70
```

---

### AC-005: Contextual Precision >= 65%

**Description**: 검색된 컨텍스트의 정밀도가 65% 이상이어야 함

**Given-When-Then Scenario**:

```
Scenario: Contextual Precision Maintenance
GIVEN 사용자가 질문하고
WHEN 시스템이 컨텍스트를 검색하면
THEN 상위 컨텍스트의 관련성이 높아야 한다
AND Contextual Precision 점수는 0.65 이상이어야 한다
```

**Measurement Method**:
```bash
# RAGAS ContextPrecision 메트릭으로 측정
python scripts/verify_evaluation_metrics.py --metric precision

# 목표: mean(contextual_precision) >= 0.65
```

---

## Component-Level Acceptance Criteria

### CAC-001: FaithfulnessValidator

**Description**: FaithfulnessValidator가 올바르게 작동해야 함

| Test Case | Input | Expected Output |
|-----------|-------|-----------------|
| 인용 추출 | "제40조에 따르면" | Claim(text="제40조", type=CITATION) |
| 수치 추출 | "30일 이내, 50%" | Claims for "30일", "50%" |
| 근거 확인 | Claim + Context with claim | is_grounded = True |
| 미근거 확인 | Claim + Context without claim | is_grounded = False |
| 점수 계산 | 3 grounded, 2 ungrounded | score = 0.6 |

**Test Commands**:
```bash
# 단위 테스트 실행
pytest tests/unit/test_faithfulness_validator.py -v

# 커버리지 확인
pytest tests/unit/test_faithfulness_validator.py --cov=src/rag/domain/evaluation/faithfulness_validator
```

### CAC-002: Enhanced Prompts

**Description**: 강화된 프롬프트가 적용되어야 함

| Test Case | Verification Method |
|-----------|---------------------|
| Context-only 규칙 포함 | 프롬프트 문자열 검색 |
| External knowledge ban 포함 | 프롬프트 문자열 검색 |
| Fallback 템플릿 포함 | 프롬프트 문자열 검색 |
| Context delimiter 포함 | [CONTEXT START/END] 검색 |

**Test Commands**:
```bash
# 프롬프트 내용 확인
python -c "from src.rag.application.search_usecase import REGULATION_QA_PROMPT; print('CONTEXT' in REGULATION_QA_PROMPT)"
```

### CAC-003: Regeneration Loop

**Description**: 낮은 Faithfulness 시 재생성 루프가 작동해야 함

| Test Case | Expected Behavior |
|-----------|-------------------|
| Faithfulness >= 0.6 | 재생성 없이 반환 |
| Faithfulness < 0.6, 1st retry | 더 엄격한 프롬프트로 재생성 |
| Faithfulness < 0.6, 2nd retry | 가장 엄격한 프롬프트로 재생성 |
| All retries failed | Fallback 응답 반환 |

**Test Commands**:
```bash
# 통합 테스트 실행
pytest tests/integration/test_faithfulness_flow.py -v
```

---

## Quality Gates

### Pre-Merge Quality Gate

```yaml
quality_gates:
  unit_tests:
    - pytest tests/unit/test_faithfulness_validator.py
    - minimum_coverage: 85%
  integration_tests:
    - pytest tests/integration/test_faithfulness_flow.py
    - all_tests_pass: true
  lint:
    - ruff check src/rag/domain/evaluation/faithfulness_validator.py
    - ruff check src/rag/application/search_usecase.py
```

### Post-Deploy Quality Gate

```yaml
quality_gates:
  evaluation:
    - faithfulness: ">= 0.60"
    - recall: ">= 0.80"
    - pass_rate: ">= 50%"
  regression:
    - no_existing_test_failures: true
    - coverage_maintained: true
```

---

## Test Execution Plan

### Phase 1: Component Testing

```bash
# 1. FaithfulnessValidator 단위 테스트
pytest tests/unit/test_faithfulness_validator.py -v --cov

# 2. 프롬프트 검증
python -c "
from src.rag.application.search_usecase import REGULATION_QA_PROMPT
assert 'CONTEXT' in REGULATION_QA_PROMPT
assert '절대' in REGULATION_QA_PROMPT or '금지' in REGULATION_QA_PROMPT
print('Prompt validation passed')
"
```

### Phase 2: Integration Testing

```bash
# 1. 통합 테스트
pytest tests/integration/test_faithfulness_flow.py -v

# 2. 샘플 쿼리 테스트
python scripts/test_faithfulness_improvement.py --sample-queries
```

### Phase 3: Full Evaluation

```bash
# 전체 평가 실행
python scripts/verify_evaluation_metrics.py --full-eval

# 결과 분석
python scripts/analyze_evaluation_results.py --compare-with-baseline
```

---

## Sign-Off Checklist

- [ ] **AC-001**: Faithfulness >= 60% 달성
- [ ] **AC-002**: Recall >= 80% 유지
- [ ] **AC-003**: Pass Rate >= 50% 달성
- [ ] **AC-004**: Answer Relevancy >= 70% 달성
- [ ] **AC-005**: Contextual Precision >= 65% 달성
- [ ] **CAC-001**: FaithfulnessValidator 단위 테스트 통과
- [ ] **CAC-002**: Enhanced Prompts 적용 확인
- [ ] **CAC-003**: Regeneration Loop 작동 확인
- [ ] 모든 품질 게이트 통과
- [ ] 회귀 테스트 통과 (기존 기능 정상)

---

## Notes

- 평가는 OPENAI_API_KEY 설정 후 실행 가능
- Fallback 응답의 Faithfulness는 1.0으로 처리 (의도적 응답)
- Recall 저하 시 즉시 프롬프트 조정 필요
