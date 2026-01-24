# SPEC-TEST-001: 인수 조건

## TAG BLOCK
```yaml
spec_id: SPEC-TEST-001
related_spec: spec.md
phase: Acceptance
created: 2025-01-24
```

## 정의 (Definition)

**용어 정의:**
- **페르소나 (Persona):** 가상의 사용자 유형으로 특정 특성, 관심사, 질문 스타일을 가짐
- **멀티턴 대화 (Multi-turn Conversation):** 3턴 이상의 후속 질문이 포함된 대화 시나리오
- **팩트체크 (Fact Check):** 답변의 핵심 주장을 규정과 대조하여 검증하는 절차
- **의도 추론 (Intent Inference):** 사용자의 표면적/숨겨진/행동 의도를 3단계로 분석
- **RAG 컴포넌트 (RAG Component):** Self-RAG, HyDE, Corrective RAG 등 검색 파이프라인 구성 요소

**성공 기준:**
- 정적 테스트 통과율 ≥ 85%
- 동적 쿼리 성공률 ≥ 80%
- 멀티턴 대화 성공률 ≥ 80%
- 후속 질문 성공률 ≥ 85%
- 답변 품질 점수 ≥ 4.0/5.0
- 팩트체크 수행률 100%
- 의도 충족률 ≥ 90%
- 의도 추론 정확도 ≥ 85%

## 인수 테스트 시나리오

### AC-001: 페르소나 생성

**Given:** 시스템이 초기화되고 10종 페르소나 정의가 로드됨

**When:** 사용자가 3개 페르소나 생성을 요청

**Then:**
- 시스템은 3개의 서로 다른 페르소나를 생성해야 함
- 각 페르소나는 고유 ID, 이름, 특성, 관심사를 가져야 함
- 생성된 페르소나는 data/output/test_sessions/personas/에 저장되어야 함

### AC-002: 쿼리 생성 (난이도 분포)

**Given:** 신입생 페르소나가 선택됨

**When:** QueryGenerator가 3개 쿼리 생성을 요청받음

**Then:**
- 1개는 "쉬움" 난이도여야 함 (단일 규정, 명확한 키워드)
- 1개는 "중간" 난이도여야 함 (여러 규정 연계)
- 1개는 "어려움" 난이도여야 함 (모호한 표현, 감정적)
- 각 쿼리는 의도 추론 3단계 분석을 포함해야 함

### AC-003: 단일 쿼리 테스트 실행

**Given:** 테스트 케이스가 준비됨 (쿼리: "휴학 신청 기간이 언제야?")

**When:** TestExecutor가 테스트를 실행

**Then:**
- RAG 시스템이 호출되어 답변을 생성해야 함
- RAG 파이프라인 로그가 캡처되어야 함
- 결과가 TestResult로 저장되어야 함
- 실행 시간이 기록되어야 함

### AC-004: 팩트체크 자동화

**Given:** 답변이 생성됨 ("수업일수 2/3 이전까지 신청 가능")

**When:** AutoFactChecker가 팩트체크 실행

**Then:**
- 핵심 주장 3개가 추출되어야 함
- 각 주장에 대해 검증 쿼리가 생성되어야 함
- 검색 결과와 대조되어야 함
- 검증 결과 (통과/실패)가 기록되어야 함

### AC-005: 품질 평가

**Given:** 테스트 결과와 팩트체크 결과가 존재

**When:** QualityEvaluator가 평가 실행

**Then:**
- 6항목 각각에 점수가 부과되어야 함
- 총점이 산출되어야 함 (5.0 만점)
- 성공/부분성공/실패 판정이 내려져야 함
- 팩트체크 실패 시 자동 감점되어야 함

### AC-006: 멀티턴 대화 시뮬레이션

**Given:** 5턴 시나리오가 정의됨

**When:** MultiTurnSimulator가 시나리오 실행

**Then:**
- 각 Turn이 순차적으로 실행되어야 함
- 이전 Turn의 맥락이 유지되어야 함
- 각 Turn별 의도 추론이 검증되어야 함
- 맥락 유지율이 측정되어야 함

### AC-007: 맥락 유지 검증

**Given:** 3턴 대화가 진행됨
- Turn 1: "휴학하고 싶어요"
- Turn 2: "일반휴학이요. 신청 기간은?"
- Turn 3: "장학금 받고 있는데 어떻게 돼요?"

**When:** ContextTracker가 맥락 유지 검증

**Then:**
- Turn 2에서 "일반휴학" 맥락이 유지되어야 함
- Turn 3에서 "일반휴학" + "장학금 수혜" 맥락이 누적되어야 함
- 맥락 연결 점수가 산출되어야 함

### AC-008: RAG 컴포넌트 기여도 분석

**Given:** 테스트 실행이 완료되고 RAG 파이프라인 로그가 존재

**When:** ComponentAnalyzer가 기여도 분석

**Then:**
- 각 RAG 컴포넌트의 동작 여부가 식별되어야 함
- 각 컴포넌트의 기여도 점수가 산출되어야 함 (-2 ~ +2)
- 종합 기여도 리포트가 생성되어야 함

### AC-009: 실패 분석 (5-Why)

**Given:** 답변이 "부분성공"으로 판정됨 (장학금 유형별 처리 누락)

**When:** FailureAnalyzer가 5-Why 분석 실행

**Then:**
- Why 1: 장학금 유형별 처리 방법 누락
- Why 2: 검색 결과에 장학금-휴학 연계 정보 부재
- Why 3: "장학금 휴학" 키워드 조합 인텐트 미존재
- Why 4: 복합 상황(A+B) 시나리오 설계 미반영
- Why 5: 인텐트 설계가 단일 주제 중심
- 개선 방안이 제시되어야 함 (intents.json에 복합 인텐트 추가)

### AC-010: 개선 적용

**Given:** 개선 제안이 생성됨 (유형: intent)

**When:** ImprovementApplier가 적용 실행

**Then:**
- data/config/intents.json에 복합 인텐트가 추가되어야 함
- 기존 인텐트는 삭제되지 않아야 함
- 패치 로그가 기록되어야 함

### AC-011: 테스트 리포트 생성

**Given:** 테스트 세션이 완료됨

**When:** TestReportGenerator가 리포트 생성

**Then:**
- 마크다운 파일이 생성되어야 함 (data/output/test_sessions/test_session_YYYYMMDD.md)
- 9개 항목이 포함되어야 함:
  1. 시작 상태
  2. RAG 컴포넌트 단위 테스트 결과
  3. 동적 테스트 결과
  4. 멀티턴 테스트 결과
  5. 답변 품질 점수
  6. RAG 컴포넌트 기여도 종합
  7. 적용된 개선
  8. 최종 상태
  9. 다음 단계 권장사항

### AC-012: Clean Architecture 준수

**Given:** domain 계층 코드가 존재

**When:** 의존성 검사 실행

**Then:**
- domain 계층은 외부 라이브러리를 직접 의존하지 않아야 함
- infrastructure 계층이 domain 인터페이스를 구현해야 함
- 의존성 방향: Interface → Application → Domain ← Infrastructure

### AC-013: 일반론 답변 감지

**Given:** 답변이 생성됨

**When:** QualityEvaluator가 평가

**Then:**
- "대학마다 다를 수 있습니다" 포함 시 자동 실패 판정
- "확인이 필요합니다" 포함 시 자동 실패 판정
- 판정理由가 기록되어야 함

### AC-014: 팩트체크 수행률

**Given:** 10개 답변이 생성됨

**When:** 팩트체크 수행률 측정

**Then:**
- 10개 모두 팩트체크가 수행되어야 함
- 수행률이 100%여야 함
- 미수행 답변은 실패로 기록되어야 함

## 품질 게이트 (Quality Gates)

### Gate 1: 정적 테스트 통과
- 조건: pytest tests/rag/automation/ 통과
- 기준: 100% 통과

### Gate 2: 코드 품질
- 조건: ruff check src/rag/automation/
- 기준: 0 errors, 0 warnings

### Gate 3: domain 계층 순수성
- 조건: domain 계층의 import 검사
- 기준: 순수 Python만 사용 (외부 라이브러리 없음)

### Gate 4: 통합 테스트
- 조건: 전체 테스트 세션 실행
- 기준: 성공률 ≥ 80%

### Gate 5: Clean Architecture 준수
- 조건: 의존성 방향 검사
- 기준: 모든 의존성이 올바른 방향

## Definition of Done

**구현 완료 기준:**
- [ ] 모든 Priority 1 작업 완료
- [ ] 모든 인수 테스트 통과
- [ ] 품질 게이트 5개 모두 통과
- [ ] 테스트 리포트가 생성됨
- [ ] Clean Architecture 준수 검증됨
- [ ] 기존 RAG 시스템과 호환성 확인됨

**문서화 기준:**
- [ ] API 문서 작성됨
- [ ] 사용자 가이드 작성됨
- [ ] 코드 주석 추가됨
- [ ] README.md 업데이트됨
