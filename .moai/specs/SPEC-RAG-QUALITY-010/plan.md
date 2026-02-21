# Implementation Plan: SPEC-RAG-QUALITY-010

## Overview

본 문서는 SPEC-RAG-QUALITY-010 (Integrated Query Enhancement & Persona Evaluation)의 구현 계획을 정의합니다.

**Predecessor**: SPEC-RAG-QUALITY-009 (Milestone 1-3 완료)

**Scope**:
- Milestone 4: Query Intent Enhancement
- Milestone 5: Evasive Response Detection
- Milestone 6: Persona Evaluation System

---

## Milestone 4: Query Intent Enhancement (P1-High)

### 목표
질문 의도 분류를 강화하여 Answer Relevancy 0.53 → 0.70+ 달성

### 작업 분해

#### M4-1: IntentHandler 클래스 구현
**파일**: `src/rag/application/intent_handler.py` (신규)

**주요 기능**:
- IntentClassificationResult 기반 검색 설정 결정
- 의도별 SearchConfig 매핑
- SearchUseCase에 설정 전달

**인터페이스**:
```python
@dataclass
class IntentSearchConfig:
    top_k: int
    procedure_boost: float
    eligibility_boost: float
    deadline_boost: float
    use_clarification: bool

class IntentHandler:
    def get_search_config(self, intent: IntentClassificationResult) -> IntentSearchConfig
    def should_generate_clarification(self, intent: IntentClassificationResult) -> bool
```

**의존성**: IntentClassifier (기존)

#### M4-2: ClarificationGenerator 구현
**파일**: `src/rag/application/intent_handler.py` (통합)

**주요 기능**:
- confidence < 0.5인 질문 감지
- 의도별 명확화 질문 템플릿
- 사용자 응답 기반 재검색

**명확화 질문 예시**:
- 절차 의심: "구체적으로 어떤 절차를 알고 싶으신가요? (신청 방법, 필요 서류, 기한)"
- 자격 의심: "어떤 자격 요건을 확인하고 싶으신가요? (지원 자격, 대상, 조건)"
- 기한 의심: "어떤 기간에 대해 궁금하신가요? (신청 기간, 처리 기간, 유효 기간)"

#### M4-3: 의도별 답변 템플릿 추가
**파일**: `data/config/prompts.json` (v2.4)

**추가 템플릿**:
```json
{
  "intent_templates": {
    "PROCEDURE": {
      "structure": ["핵심 절차 요약", "단계별 안내", "필요 서류", "관련 부서"],
      "emphasis": "절차와 방법 중심"
    },
    "ELIGIBILITY": {
      "structure": ["자격 요건", "대상 구분", "필요 조건", "확인 방법"],
      "emphasis": "조건과 자격 중심"
    },
    "DEADLINE": {
      "structure": ["기한 요약", "세부 일정", "예외 사항", "확인 방법"],
      "emphasis": "기간과 날짜 중심"
    },
    "GENERAL": {
      "structure": ["핵심 답변", "상세 내용", "관련 정보"],
      "emphasis": "균형 잡힌 답변"
    }
  }
}
```

#### M4-4: IntentClassifier 규칙 보강
**파일**: `src/rag/application/intent_classifier.py` (수정)

**추가 키워드**:
- PROCEDURE: "절차는", "어떻게 되나", "진행 방법", "진행 절차"
- ELIGIBILITY: "자격이 되나", "대상이 되나", "조건이 뭐", "누가 받을 수"
- DEADLINE: "언제까지인가", "기한이 언제", "마감일이", "신청 기간"

**Colloquial Pattern 추가**:
- "~나요?" → ELIGIBILITY
- "~까요?" → DEADLINE (timing)

#### M4-5: 의도 분류 정확도 테스트
**파일**: `tests/rag/unit/application/test_intent_accuracy.py` (신규)

**테스트 케이스**:
- 절차 질문 20개 (PROCEDURE 예상)
- 자격 질문 20개 (ELIGIBILITY 예상)
- 기한 질문 20개 (DEADLINE 예상)
- 일반 질문 20개 (GENERAL 예상)

**목표**: 정확도 >= 85%

### 완료 기준
- [ ] IntentHandler 구현 및 테스트
- [ ] ClarificationGenerator 구현
- [ ] prompts.json v2.4 업데이트
- [ ] IntentClassifier 키워드 추가
- [ ] 의도 분류 정확도 85% 달성

---

## Milestone 5: Evasive Response Detection (P1-High)

### 목표
회피성 답변 비율 < 5% 달성

### 작업 분해

#### M5-1: EvasiveResponseDetector 클래스 구현
**파일**: `src/rag/domain/evaluation/evasive_detector.py` (신규)

**주요 기능**:
- 회피성 패턴 감지
- 컨텍스트 정보 존재 여부 검증
- 재생성 필요성 판단

**인터페이스**:
```python
@dataclass
class EvasiveDetectionResult:
    is_evasive: bool
    detected_patterns: List[str]
    context_has_info: bool
    confidence: float

class EvasiveResponseDetector:
    EVASIVE_PATTERNS = [
        r"홈페이지.*참고",
        r"관련 부서.*문의",
        r"제공된.*컨텍스트.*확인.*없",
        r"규정에서.*확인.*없",
        r"정확한.*확인.*바랍니다",
    ]

    def detect(self, response: str, context: str) -> EvasiveDetectionResult
    def should_regenerate(self, result: EvasiveDetectionResult) -> bool
```

#### M5-2: 5가지 회피성 패턴 정의

| Pattern | Regex | Condition |
|---------|-------|-----------|
| Homepage Deflection | `홈페이지.*참고` | 컨텍스트에 정보 있음 |
| Department Deflection | `관련 부서.*문의` | 컨텍스트에 부서 정보 있음 |
| Context Denial | `제공된.*컨텍스트.*확인.*없` | 실제로 정보 존재 |
| Regulation Denial | `규정에서.*확인.*없` | 실제로 규정에 존재 |
| Vague Confirmation | `정확한.*확인.*바랍니다` | 구체적 정보 있음에도 회피 |

#### M5-3: SearchUseCase에 회피성 검사 통합
**파일**: `src/rag/application/search_usecase.py` (수정)

**통합 위치**: `_generate_answer_with_validation()` 내부

**플로우**:
1. LLM 답변 생성
2. FaithfulnessValidator 검사
3. EvasiveResponseDetector 검사 (NEW)
4. 회피성 감지 + 컨텍스트 정보 존재 → 재생성
5. 최대 1회 재생성

#### M5-4: 회피성 답변 재생성 로직
**로직**:
```python
if evasive_result.is_evasive and evasive_result.context_has_info:
    # 재생성 프롬프트에 힌트 추가
    hint = f"다음 정보가 컨텍스트에 있습니다: {evasive_result.detected_patterns}"
    regenerated = regenerate_with_hint(response, hint)
    return regenerated
```

#### M5-5: 회피성 답변 비율 메트릭 수집
**메트릭**:
- `evasive_detection_total`: 전체 감지 시도
- `evasive_detected_count`: 감지된 회피성 답변
- `evasive_regenerated_count`: 재생성된 답변
- `evasive_rate`: 회피성 답변 비율

**저장**: `.metrics/evasive_detection/` 디렉토리

### 완료 기준
- [ ] EvasiveResponseDetector 구현
- [ ] 5가지 패턴 감지 테스트
- [ ] SearchUseCase 통합
- [ ] 재생성 로직 구현
- [ ] 회피성 비율 < 5% 달성

---

## Milestone 6: Persona Evaluation System (P2-Medium)

### 목표
6가지 페르소나별 품질 측정 시스템 구축

### 작업 분해

#### M6-1: PersonaDefinition 데이터 클래스 구현
**파일**: `src/rag/domain/evaluation/persona_definition.py` (신규)

**데이터 구조**:
```python
@dataclass
class PersonaDefinition:
    id: str
    name: str
    description: str
    language_level: str  # simple, normal, formal, technical
    citation_preference: str  # minimal, normal, extensive
    key_requirements: List[str]
    query_templates: List[str]

PERSONAS = {
    "freshman": PersonaDefinition(
        id="freshman",
        name="신입생",
        description="대학 규정에 익숙하지 않은 1학년 학생",
        language_level="simple",
        citation_preference="minimal",
        key_requirements=["간단명료한 설명", "친절한 안내", "전문 용어 설명"]
    ),
    # ... 5 more personas
}
```

#### M6-2: 페르소나별 쿼리 템플릿 생성
**파일**: `data/config/persona_queries.json` (신규)

**템플릿 구조**:
```json
{
  "freshman": [
    {
      "query": "휴학하려면 뭐 해야 해요?",
      "expected_intent": "PROCEDURE",
      "key_elements": ["간단한 절차", "학생정보시스템", "지도교수"]
    },
    // ... 9 more
  ],
  "professor": [
    {
      "query": "연구비 집행 규정에 따른 처리 절차를 알고 싶습니다.",
      "expected_intent": "PROCEDURE",
      "key_elements": ["규정 조항", "담당 부서", "처리 기한"]
    },
    // ... 9 more
  ]
  // ... 4 more personas
}
```

#### M6-3: PersonaEvaluator 클래스 구현
**파일**: `src/rag/domain/evaluation/persona_evaluator.py` (신규)

**인터페이스**:
```python
@dataclass
class PersonaEvaluationResult:
    persona_id: str
    query: str
    response: str
    scores: Dict[str, float]  # relevancy, clarity, completeness, citation_quality
    overall_score: float
    issues: List[str]

class PersonaEvaluator:
    def __init__(self, personas: Dict[str, PersonaDefinition]):
        self.personas = personas

    def evaluate_persona(
        self,
        persona_id: str,
        queries: List[str],
        rag_pipeline
    ) -> List[PersonaEvaluationResult]

    def generate_report(
        self,
        results: List[PersonaEvaluationResult]
    ) -> PersonaEvaluationReport
```

**평가 기준**:
- `relevancy`: 페르소나 요구에 맞는 답변인가?
- `clarity`: 페르소나의 언어 수준에 적절한가?
- `completeness`: 필수 정보가 모두 포함되었는가?
- `citation_quality`: 인용이 페르소나 요구사항에 맞는가?

#### M6-4: 페르소나별 점수 대시보드 데이터 구조
**파일**: `.metrics/persona_evaluation/` (신규 디렉토리)

**데이터 구조**:
```json
{
  "evaluation_date": "2026-02-21T10:00:00",
  "personas": {
    "freshman": {
      "avg_overall": 0.72,
      "avg_relevancy": 0.75,
      "avg_clarity": 0.80,
      "avg_completeness": 0.68,
      "avg_citation": 0.65,
      "issue_count": 3
    },
    // ... 5 more
  },
  "weak_personas": ["freshman", "international"],
  "recommendations": [
    "신입생 페르소나: 전문 용어 설명 보강 필요",
    "외국인 유학생: 간단한 한국어 템플릿 추가 필요"
  ]
}
```

#### M6-5: 취약 페르소나 식별 및 리포트
**식별 기준**:
- overall_score < 0.65 → 취약 페르소나
- 특정 점수 < 0.60 → 개선 필요 영역

**리포트 형식**:
```markdown
# Persona Evaluation Report

## Summary
- Evaluated: 6 personas, 60 queries
- Average Overall Score: 0.72
- Weak Personas: freshman (0.62), international (0.58)

## Persona Details

### Freshman (Score: 0.62)
- Issues: 전문 용어 설명 부족, 복잡한 절차 설명
- Recommendations: 간단한 용어 설명 추가, 단계별 시각화

### International (Score: 0.58)
- Issues: 한국어 복잡도 높음, 문화적 맥락 부재
- Recommendations: 간단한 한국어 템플릿, 영문 병행 안내
```

### 완료 기준
- [ ] PersonaDefinition 6종 구현
- [ ] 쿼리 템플릿 60개 생성
- [ ] PersonaEvaluator 구현
- [ ] 대시보드 데이터 구조 정의
- [ ] 취약 페르소나 식별 기능 구현

---

## Dependencies Graph

```
M4-1 (IntentHandler)
  └── M4-2 (ClarificationGenerator)
        └── M4-3 (Templates)
              └── M4-4 (Classifier)
                    └── M4-5 (Tests)

M5-1 (Detector)
  └── M5-2 (Patterns)
        └── M5-3 (Integration)
              └── M5-4 (Regeneration)
                    └── M5-5 (Metrics)

M6-1 (PersonaDefinition)
  └── M6-2 (QueryTemplates)
        └── M6-3 (Evaluator)
              └── M6-4 (Dashboard)
                    └── M6-5 (Report)
```

---

## Execution Order

### Phase 1: Query Intent (1-2일)
1. M4-1 → M4-2 → M4-3 (병렬 가능)
2. M4-4 → M4-5

### Phase 2: Evasive Detection (1일)
1. M5-1 → M5-2
2. M5-3 → M5-4 → M5-5

### Phase 3: Persona Evaluation (2일)
1. M6-1 → M6-2 (병렬 가능)
2. M6-3 → M6-4 → M6-5

---

## Test Strategy

### Unit Tests
- `test_intent_handler.py`: IntentHandler 로직 테스트
- `test_evasive_detector.py`: 패턴 감지 테스트
- `test_persona_evaluator.py`: 페르소나 평가 로직 테스트

### Integration Tests
- `test_intent_integration.py`: SearchUseCase + IntentHandler 통합
- `test_evasive_integration.py`: 회피성 재생성 플로우
- `test_persona_integration.py`: 전체 페르소나 평가 파이프라인

### Quality Validation
- 의도 분류 정확도 테스트 (85% 목표)
- 회피성 답변 비율 측정 (<5% 목표)
- 페르소나별 점수 측정 (모든 페르소나 >= 0.65)

---

## Rollback Plan

### M4 Rollback
- IntentHandler 비활성화 시 기존 IntentClassifier만 사용
- `use_intent_handler: false` 설정으로 전환

### M5 Rollback
- EvasiveResponseDetector 비활성화
- `use_evasive_detection: false` 설정으로 전환

### M6 Rollback
- PersonaEvaluator는 독립 실행형, RAG 파이프라인에 영향 없음

---

## Monitoring

### M4 Metrics
- `intent_classification_accuracy`: 의도 분류 정확도
- `clarification_generated_count`: 명확화 질문 생성 수

### M5 Metrics
- `evasive_detection_rate`: 회피성 감지율
- `evasive_regeneration_rate`: 재생성율
- `evasive_final_rate`: 최종 회피성 비율

### M6 Metrics
- `persona_avg_score`: 페르소나별 평균 점수
- `persona_weak_count`: 취약 페르소나 수
