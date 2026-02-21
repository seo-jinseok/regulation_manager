# Acceptance Criteria: SPEC-RAG-QUALITY-010

## Overview

본 문서는 SPEC-RAG-QUALITY-010의 수락 기준을 Given-When-Then (Gherkin) 형식으로 정의합니다.

---

## REQ-004: Enhance Query Intent Detection

### AC-004-1: Intent Classification Accuracy

**Given** 사용자가 질문을 입력한다
**When** IntentClassifier가 질문을 분류한다
**Then** 분류 정확도가 85% 이상이어야 한다

**테스트 케이스**:
```
| Query | Expected Intent | Confidence |
|-------|-----------------|------------|
| "휴학 신청 어떻게 해요?" | PROCEDURE | >= 0.7 |
| "장학금 받을 수 있나요?" | ELIGIBILITY | >= 0.7 |
| "수강신청 언제까지인가요?" | DEADLINE | >= 0.7 |
| "학교 규정이 궁금해요" | GENERAL | >= 0.5 |
```

### AC-004-2: Intent-Specific Search Configuration

**Given** 질문의 의도가 분류된다
**When** IntentHandler가 검색 설정을 결정한다
**Then** 의도에 맞는 최적화된 검색 설정이 적용되어야 한다

**검색 설정 매핑**:
```
| Intent | top_k | Procedure Boost | Eligibility Boost | Deadline Boost |
|--------|-------|-----------------|-------------------|----------------|
| PROCEDURE | 8 | 1.5x | 1.0x | 1.0x |
| ELIGIBILITY | 6 | 1.0x | 1.3x | 1.0x |
| DEADLINE | 5 | 1.0x | 1.0x | 1.4x |
| GENERAL | 5 | 1.0x | 1.0x | 1.0x |
```

### AC-004-3: Clarification Question Generation

**Given** 질문의 의도 분류 신뢰도가 0.5 미만이다
**When** IntentHandler가 모호성을 감지한다
**Then** 사용자에게 명확화 질문이 제시되어야 한다

**명확화 질문 예시**:
```gherkin
Scenario: 모호한 질문에 대한 명확화
  Given 사용자가 "이거 뭐예요?"라고 질문함
  And IntentClassifier confidence < 0.5
  When 시스템이 명확화 질문을 생성함
  Then 다음 중 하나의 질문이 반환됨:
    - "구체적으로 어떤 절차를 알고 싶으신가요?"
    - "어떤 자격 요건을 확인하고 싶으신가요?"
    - "어떤 기간에 대해 궁금하신가요?"
```

### AC-004-4: Intent-Specific Response Structure

**Given** 질문의 의도가 분류된다
**When** 답변이 생성된다
**Then** 의도에 맞는 구조화된 답변 포맷이 적용되어야 한다

**답변 구조 검증**:
```gherkin
Scenario: 절차 질문에 대한 구조화된 답변
  Given 질문 의도가 PROCEDURE로 분류됨
  When 답변이 생성됨
  Then 답변에 다음 섹션이 포함됨:
    | Section | Required |
    |---------|----------|
    | 핵심 절차 요약 | Yes |
    | 단계별 안내 | Yes |
    | 필요 서류 | If applicable |
    | 관련 부서 | If applicable |
```

### AC-004-5: Answer Relevancy Target

**Given** Query Intent Enhancement가 구현된다
**When** RAG 품질 평가를 실행한다
**Then** Answer Relevancy 점수가 0.70 이상이어야 한다

**측정 방법**:
- RAGAS `answer_relevancy` 메트릭
- 또는 deepeval `AnswerRelevancyMetric`
- 평가셋: 50개 표준 질문

---

## REQ-005: Remove Evasive Response Patterns

### AC-005-1: Evasive Pattern Detection

**Given** 답변이 생성된다
**When** EvasiveResponseDetector가 답변을 분석한다
**Then** 5가지 회피성 패턴이 정확히 감지되어야 한다

**감지 대상 패턴**:
```gherkin
Scenario Outline: 회피성 패턴 감지
  Given 답변에 "<pattern>"이 포함됨
  And 컨텍스트에 관련 정보가 존재함
  When EvasiveResponseDetector가 감지함
  Then is_evasive = true
  And detected_patterns에 "<pattern>" 포함

  Examples:
    | pattern |
    | 홈페이지 참고하세요 |
    | 관련 부서에 문의하세요 |
    | 제공된 컨텍스트에서 확인되지 않습니다 |
    | 규정에서 확인되지 않습니다 |
    | 정확한 내용은 확인 바랍니다 |
```

### AC-005-2: Context Information Verification

**Given** 회피성 패턴이 감지된다
**When** 시스템이 컨텍스트를 재확인한다
**Then** 실제 정보 존재 여부가 정확히 판단되어야 한다

**검증 로직**:
```gherkin
Scenario: 컨텍스트 정보 존재 검증
  Given 답변에 "관련 부서에 문의하세요" 포함
  When 시스템이 컨텍스트에서 부서 정보 검색
  Then 다음 중 하나:
    | Context Has Info | Action |
    | true | 재생성 시도 |
    | false | 답변 유지 (정당한 회피) |
```

### AC-005-3: Automatic Regeneration

**Given** 회피성 답변이 감지된다
**And** 컨텍스트에 정보가 존재한다
**When** 시스템이 답변을 재생성한다
**Then** 재생성된 답변이 회피성 패턴을 포함하지 않아야 한다

**재생성 플로우**:
```gherkin
Scenario: 회피성 답변 자동 재생성
  Given EvasiveDetectionResult.is_evasive = true
  And EvasiveDetectionResult.context_has_info = true
  When 시스템이 답변 재생성
  Then 재생성 횟수는 최대 1회
  And 재생성된 답변은 구체적 정보를 포함
  And 재생성된 답변은 회피성 패턴 미포함
```

### AC-005-4: Evasive Response Rate Target

**Given** Evasive Response Detection이 구현된다
**When** 평가셋 100개 질문에 대해 테스트한다
**Then** 최종 회피성 답변 비율이 5% 미만이어야 한다

**측정 지표**:
```
evasive_rate = (evasive_detected_count - false_positive_count) / total_responses
target: evasive_rate < 0.05
```

### AC-005-5: Metrics Collection

**Given** 회피성 감지 시스템이 운영된다
**When** 답변이 생성될 때마다
**Then** 다음 메트릭이 수집되어야 한다

**수집 메트릭**:
| Metric | Description | Storage |
|--------|-------------|---------|
| evasive_detection_total | 전체 감지 시도 | `.metrics/evasive/total.json` |
| evasive_detected_count | 감지된 회피성 | `.metrics/evasive/detected.json` |
| evasive_regenerated_count | 재생성 횟수 | `.metrics/evasive/regenerated.json` |
| evasive_false_positive | 오탐 횟수 | `.metrics/evasive/false_positive.json` |

---

## REQ-006: Implement Persona-Based Evaluation System

### AC-006-1: Persona Definition Completeness

**Given** 6가지 페르소나가 정의된다
**When** PersonaDefinition을 조회한다
**Then** 모든 페르소나가 완전한 속성을 가져야 한다

**필수 속성**:
```gherkin
Scenario Outline: 페르소나 정의 완전성
  Given "<persona_id>" 페르소나
  Then 다음 속성이 모두 존재함:
    | Attribute | Type |
    | id | str |
    | name | str |
    | description | str |
    | language_level | str |
    | citation_preference | str |
    | key_requirements | List[str] |
    | query_templates | List[str] |

  Examples:
    | persona_id |
    | freshman |
    | student |
    | professor |
    | staff |
    | parent |
    | international |
```

### AC-006-2: Query Template Coverage

**Given** 각 페르소나별 쿼리 템플릿이 생성된다
**When** 템플릿을 검증한다
**Then** 각 페르소나당 최소 10개의 템플릿이 존재해야 한다

**템플릿 검증**:
```gherkin
Scenario: 쿼리 템플릿 커버리지
  Given persona_queries.json 파일
  When 각 페르소나의 템플릿 수 확인
  Then 모든 페르소나에 대해:
    | Persona | Min Templates |
    |---------|---------------|
    | freshman | >= 10 |
    | student | >= 10 |
    | professor | >= 10 |
    | staff | >= 10 |
    | parent | >= 10 |
    | international | >= 10 |
```

### AC-006-3: Persona Evaluation Execution

**Given** PersonaEvaluator가 구현된다
**When** 특정 페르소나에 대해 평가를 실행한다
**Then** 4가지 점수가 정확히 계산되어야 한다

**평가 점수**:
```gherkin
Scenario: 페르소나 평가 점수 계산
  Given freshman 페르소나로 평가 실행
  When PersonaEvaluator.evaluate_persona() 호출
  Then 반환된 PersonaEvaluationResult에 포함:
    | Score | Range | Description |
    |-------|-------|-------------|
    | relevancy | 0.0-1.0 | 페르소나 요구 적합성 |
    | clarity | 0.0-1.0 | 언어 수준 적절성 |
    | completeness | 0.0-1.0 | 정보 완전성 |
    | citation_quality | 0.0-1.0 | 인용 품질 |
```

### AC-006-4: Persona Score Dashboard

**Given** 모든 페르소나에 대한 평가가 완료된다
**When** 대시보드 데이터를 생성한다
**Then** 페르소나별 평균 점수가 정확히 집계되어야 한다

**대시보드 데이터 검증**:
```gherkin
Scenario: 페르소나 점수 집계
  Given 6개 페르소나 평가 완료
  When 대시보드 데이터 생성
  Then 각 페르소나에 대해:
    - avg_overall 계산됨
    - avg_relevancy 계산됨
    - avg_clarity 계산됨
    - avg_completeness 계산됨
    - avg_citation 계산됨
    - issue_count 포함됨
```

### AC-006-5: Weak Persona Identification

**Given** 페르소나별 점수가 집계된다
**When** 취약 페르소나를 식별한다
**Then** overall_score < 0.65인 페르소나가 정확히 식별되어야 한다

**식별 기준**:
```gherkin
Scenario: 취약 페르소나 식별
  Given 페르소나 점수 집계 완료
  When weak_personas 리스트 생성
  Then overall_score < 0.65인 모든 페르소나 포함
  And recommendations에 개선 방향 포함
```

### AC-006-6: All Persona Score Target

**Given** Persona Evaluation System이 완성된다
**When** 전체 페르소나 평가를 실행한다
**Then** 모든 페르소나의 overall_score가 0.65 이상이어야 한다

**목표 점수**:
| Persona | Target Overall Score |
|---------|---------------------|
| freshman | >= 0.65 |
| student | >= 0.65 |
| professor | >= 0.65 |
| staff | >= 0.65 |
| parent | >= 0.65 |
| international | >= 0.65 |

---

## Integration Tests

### IT-001: End-to-End Query Intent Flow

**Given** 사용자가 "휴학 신청 방법 알려주세요"라고 질문한다
**When** 전체 RAG 파이프라인이 실행된다
**Then** 다음이 검증되어야 한다:
1. IntentClassifier → PROCEDURE 분류 (confidence >= 0.7)
2. IntentHandler → top_k=8, procedure_boost=1.5x
3. SearchUseCase → 의도별 최적화된 검색
4. Answer Generation → 절차 중심 구조화된 답변
5. EvasiveResponseDetector → 회피성 패턴 미감지

### IT-002: Evasive Response Regeneration Flow

**Given** LLM이 회피성 답변을 생성한다 ("홈페이지 참고하세요")
**And** 컨텍스트에 실제 정보가 존재한다
**When** 전체 파이프라인이 실행된다
**Then** 다음이 검증되어야 한다:
1. EvasiveResponseDetector → is_evasive=true
2. Context verification → context_has_info=true
3. Regeneration → 재생성 시도
4. Final Answer → 회피성 패턴 미포함, 구체적 정보 포함

### IT-003: Persona Evaluation Pipeline

**Given** 6가지 페르소나와 각 10개 쿼리 템플릿이 준비된다
**When** PersonaEvaluator가 전체 평가를 실행한다
**Then** 다음이 검증되어야 한다:
1. 60개 쿼리에 대한 평가 완료
2. 각 쿼리당 4개 점수 계산
3. 페르소나별 평균 점수 집계
4. 취약 페르소나 식별
5. 개선 recommendation 생성

---

## Quality Gates

### Gate 1: Milestone 4 Completion
- [ ] AC-004-1 ~ AC-004-5 모두 통과
- [ ] 의도 분류 정확도 >= 85%
- [ ] Answer Relevancy >= 0.70

### Gate 2: Milestone 5 Completion
- [ ] AC-005-1 ~ AC-005-5 모두 통과
- [ ] 회피성 답변 비율 < 5%
- [ ] 재생성 성공률 >= 80%

### Gate 3: Milestone 6 Completion
- [ ] AC-006-1 ~ AC-006-6 모두 통과
- [ ] 모든 페르소나 overall_score >= 0.65
- [ ] 취약 페르소나 식별 기능 동작

### Gate 4: Integration Tests
- [ ] IT-001 ~ IT-003 모두 통과
- [ ] End-to-end 플로우 정상 동작

---

## Test Coverage Requirements

### Unit Tests
| Module | Target Coverage |
|--------|-----------------|
| intent_handler.py | >= 85% |
| evasive_detector.py | >= 90% |
| persona_evaluator.py | >= 85% |
| persona_definition.py | >= 95% |

### Integration Tests
| Integration | Required |
|-------------|----------|
| IntentHandler + SearchUseCase | Yes |
| EvasiveDetector + FaithfulnessValidator | Yes |
| PersonaEvaluator + QualityEvaluator | Yes |

---

## Validation Checklist

### Pre-Implementation
- [ ] SPEC-RAG-QUALITY-009 Milestone 1-3 완료 확인
- [ ] IntentClassifier 기존 구현체 분석 완료
- [ ] prompts.json v2.3 구조 파악 완료

### During Implementation
- [ ] 각 Milestone별 AC 충족 확인
- [ ] Unit Tests 작성 및 통과
- [ ] Integration Tests 작성 및 통과

### Post-Implementation
- [ ] 전체 Quality Gates 통과
- [ ] 메트릭 수집 확인
- [ ] 문서화 완료 (SPEC update)
