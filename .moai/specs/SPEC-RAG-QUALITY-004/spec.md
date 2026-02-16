# SPEC-RAG-QUALITY-004: RAG 품질 긴급 개선 - Stage 1

## Metadata

| 항목 | 값 |
|------|-----|
| SPEC ID | SPEC-RAG-QUALITY-004 |
| Status | Draft |
| Priority | P0 (Critical) |
| Created | 2026-02-16 |
| Author | RAG Quality Evaluator |
| Related | SPEC-RAG-QUALITY-001, SPEC-RAG-QUALITY-002, SPEC-RAG-QUALITY-003 |

## Problem Statement

### Current State

2026-02-15 평가 결과:
- **통과율**: 0% (30개 시나리오 모두 실패)
- **Overall Score**: 0.416 (임계값 0.6 미달)
- **Critical Issues**: Faithfulness 0.313, Contextual Recall 0.247

### Target State

Stage 1 목표 (Week 1):
- 통과율: 60% 이상
- Overall Score: 0.6 이상
- Faithfulness: 0.6 이상
- Contextual Recall: 0.65 이상

### Gap Analysis

| 메트릭 | 현재 | 목표 | 격차 |
|--------|------|------|------|
| Faithfulness | 0.313 | 0.60 | -0.287 |
| Answer Relevancy | 0.687 | 0.70 | -0.013 |
| Contextual Precision | 0.427 | 0.65 | -0.223 |
| Contextual Recall | 0.247 | 0.65 | -0.403 |

## Root Cause Analysis

### 1. Faithfulness 저하 원인 (0.313)

**발생 빈도**: 30개 중 14개 시나리오에서 Faithfulness < 0.5

**패턴 분석**:
- CRITICAL 상태 (Faithfulness < 0.3): 8건
  - freshman_001: "휴학 어떻게 해요?" → 0.0
  - freshman_003: "성적이 나쁘면 휴학해야 하나요?" → 0.4
  - graduate_003: "논문 제출 기한 연장 가능한가요?" → 0.0
  - graduate_004: "조교 근무 시간과 장학금 혜택" → 0.0
  - parent 쿼리 다수: 0.2~0.4

**원인**:
1. LLM이 컨텍스트 부족 시 추측성 답변 생성
2. Hallucination filter가 sanitize만 수행, 검증 미흡
3. "모르겠습니다" 답변 회피 패턴

### 2. Contextual Recall 저하 원인 (0.247)

**발생 빈도**: 30개 중 26개 시나리오에서 Recall < 0.65

**패턴 분석**:
- Recall 0.0인 쿼리: 5건 (완전 검색 실패)
- Recall 0.3~0.5인 쿼리: 12건 (부분 검색 실패)

**원인**:
1. 구어체 쿼리 → 전문 용어 매핑 실패
2. 복합 질문 분해 후 병합 과정에서 정보 손실
3. Reranker가 중요 문서를 제외

### 3. 페르소나별 성능 격차

| 페르소나 | 점수 | 문제점 |
|----------|------|--------|
| parent | 0.152 | 일상 언어 → 규정 용어 변환 실패 |
| international | 0.408 | 영어/한국어 혼용 쿼리 처리 미흡 |
| professor | 0.447 | 복잡한 다중 조건 쿼리 분해 실패 |

## Requirements

### REQ-001: Faithfulness 개선 (P0)

**WHEN** Faithfulness 점수가 0.5 미만으로 예측될 때
**THE SYSTEM SHALL** 다음 조치 중 하나를 수행한다:
1. 추가 컨텍스트 검색 (top_k 증가)
2. "해당 정보를 찾을 수 없습니다" 답변
3. 신뢰도 점수와 함께 답변 제공

**Acceptance Criteria**:
- [ ] Faithfulness < 0.3인 답변 생성 차단
- [ ] 할루시네이션 감지 시 자동 재시도 (최대 2회)
- [ ] 모든 인용에 대해 source verification 수행

### REQ-002: 검색 품질 개선 (P0)

**WHEN** 사용자 쿼리가 검색 시스템에 전달될 때
**THE SYSTEM SHALL** 다음을 보장한다:
1. 최소 3개 이상의 관련 문서 검색
2. 검색된 문서의 총 토큰 수 > 500
3. 질문 키워드와 매칭되는 문서 포함

**Acceptance Criteria**:
- [ ] Contextual Recall 평균 0.5 이상 달성
- [ ] 검색 실패율 10% 미만
- [ ] 구어체 쿼리에 대한 동의어 확장 적용

### REQ-003: 페르소나별 최적화 (P1)

**WHEN** parent 또는 international 페르소나 감지 시
**THE SYSTEM SHALL** 다음을 수행한다:
1. 일상 언어 → 규정 용어 매핑
2. 영어 쿼리에 대한 한국어 번역 후 검색
3. 간단한 언어로 답변 생성

**Acceptance Criteria**:
- [ ] parent 페르소나 평균 점수 0.4 이상
- [ ] international 페르소나 평균 점수 0.5 이상

## Technical Approach

### Phase 1: Faithfulness 개선 (2일)

1. **Hallucination Filter 강화**
   ```python
   # 파일: src/rag/application/search_usecase.py
   # 변경: _apply_hallucination_filter()

   def _apply_hallucination_filter(self, answer, sources, threshold=0.3):
       if self._calculate_faithfulness(answer, sources) < threshold:
           return self._generate_safe_response(sources)
       return answer
   ```

2. **Safe Response Generator 추가**
   - 컨텍스트 기반 안전 답변 생성
   - "정보를 찾을 수 없습니다" + 관련 규정명 제안

### Phase 2: 검색 품질 개선 (3일)

1. **동의어 확장기 추가**
   ```yaml
   # 파일: data/synonyms.yaml
   휴학:
     - 학기 휴업
     - 등록 휴부
     - 학업 중단
   장학금:
     - 학비 지원
     - 장학 혜택
     - 등록금 감면
   ```

2. **Reranker 임계값 조정**
   - 현재: 상위 5개
   - 변경: 상위 10개 후 필터링

### Phase 3: 페르소나 최적화 (2일)

1. **Parent 페르소나 핸들러**
   - 일상 언어 → 규정 용어 매핑
   - 간단한 설명 우선

2. **International 페르소나 핸들러**
   - 영어 쿼리 한국어 번역
   - 이중 언어 검색

## Test Plan

### Unit Tests

1. `test_faithfulness_filter.py`
   - Faithfulness < 0.3인 답변 차단 테스트
   - Safe response 생성 테스트

2. `test_synonym_expansion.py`
   - 구어체 → 규정 용어 매핑 테스트
   - 확장된 쿼리로 검색 품질 테스트

### Integration Tests

1. `test_persona_handling.py`
   - 6개 페르소나별 응답 품질 테스트
   - Parent/International 특화 테스트

### Regression Tests

1. `scripts/custom_llm_judge_eval.py`
   - 개선 후 재평가
   - 목표: 통과율 60% 이상

## Success Metrics

| 메트릭 | 현재 | Stage 1 목표 | Stage 2 목표 |
|--------|------|-------------|-------------|
| 통과율 | 0% | 60% | 80% |
| Faithfulness | 0.313 | 0.60 | 0.75 |
| Answer Relevancy | 0.687 | 0.70 | 0.80 |
| Contextual Precision | 0.427 | 0.65 | 0.75 |
| Contextual Recall | 0.247 | 0.50 | 0.65 |

## Risks and Mitigations

| 위험 | 가능성 | 영향 | 완화책 |
|------|--------|------|--------|
| Hallucination filter 과도 차단 | 중 | 중 | 임계값 조정, A/B 테스트 |
| 동의어 확장 오버헤드 | 낮 | 낮 | 캐싱, 사전 로딩 |
| 페르소나 오분류 | 중 | 중 | 명시적 페르소나 선택 옵션 |

## References

- 평가 결과: `data/evaluations/custom_llm_judge_eval_stage1_latest.json`
- 이전 SPEC: SPEC-RAG-QUALITY-003
- 관련 코드: `src/rag/application/search_usecase.py`

---

## Change Log

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-02-16 | 1.0.0 | 초기 작성 |

<moai>DONE</moai>
