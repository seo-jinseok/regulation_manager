# SPEC-RAG-QUALITY-008: Faithfulness Enhancement for RAG System

## Metadata

| Field | Value |
|-------|-------|
| ID | SPEC-RAG-QUALITY-008 |
| Status | Completed |
| Created | 2026-02-20 |
| Completed | 2026-02-21 |
| Priority | Critical |
| Source | Evaluation Analysis (2026-02-20) |
| Predecessor | SPEC-RAG-QUALITY-007 |

---

## Problem Statement

### Current State

RAG 시스템 품질 평가 결과, **Faithfulness 메트릭이 30%로 심각하게 낮음**:

| Metric | SPEC-007 Before | Current (2026-02-20) | Target | Gap |
|--------|-----------------|---------------------|--------|-----|
| Faithfulness | 50.3% | **30%** | 60%+ | **-30%** |
| Contextual Recall | 32% | 87% | 80%+ | +7% |
| Answer Relevancy | N/A | 53% | 70%+ | -17% |
| Contextual Precision | N/A | 50% | 65%+ | -15% |
| Pass Rate | 0% | 0% | 50%+ | -50% |

### Critical Issue: Faithfulness Regression

**Faithfulness가 50.3%에서 30%로 20.3%p 감소** - 가장 심각한 문제

- **Recall은 87%로 매우 높음**: 시스템이 관련 컨텍스트를 효과적으로 검색
- **Faithfulness는 30%로 매우 낮음**: 생성된 답변이 검색된 컨텍스트에 기반하지 않음 (할루시네이션)

**결론**: 시스템은 관련 컨텍스트를 잘 찾지만, LLM이 컨텍스트에 없는 정보를 추가 생성함

### Sample Failure Analysis

평가 결과 샘플에서 발견된 실패 패턴:

1. **evaluation_20260220_170228.json** (질문: "자녀 장학금 관련해서 알고 싶어요")
   - faithfulness: 0.5 (임계값 0.6 미만)
   - failure_reasons: "CRITICAL: Faithfulness below critical threshold - high hallucination risk"

2. **evaluation_20260220_170128.json** (질문: "등록금 부모님도 알아야 하나요?")
   - faithfulness: 0.5 (임계값 0.6 미만)
   - failure_reasons: "CRITICAL: Faithfulness below critical threshold - high hallucination risk"

### Root Cause Analysis (Five Whys)

1. **Why Faithfulness is low?** → LLM이 컨텍스트에 없는 정보를 생성
2. **Why LLM generates external info?** → 프롬프트가 "컨텍스트만 사용"을 충분히 강제하지 않음
3. **Why prompt is not strict enough?** → 현재 프롬프트는 할루시네이션 방지를 포함하지만 후처리 검증이 없음
4. **Why no post-validation?** → 생성 후 답변이 컨텍스트에 근거하는지 검증하는 메커니즘 부재
5. **Root Cause**: **컨텍스트 근거성 검증 메커니즘 부재 + 프롬프트 강제력 부족**

---

## Requirements (EARS Format)

### REQ-001: Strict Context-Only Prompt Engineering

**THE SYSTEM SHALL** RAG 답변 생성 시 다음 규칙을 프롬프트에 강제:
- 제공된 컨텍스트에 명시된 정보만 답변에 포함
- 컨텍스트에 없는 정보는 절대 생성하지 않음
- 정보가 부족한 경우 명시적으로 "제공된 규정에서 찾을 수 없습니다" 선언
- 모든 사실적 주장에 출처 인용 필수

**WHEN** LLM이 답변을 생성할 때
**THE SYSTEM SHALL** 다음 프롬프트 규칙을 적용:
1. "당신은 제공된 문맥에서 찾을 수 있는 정보만으로 답변해야 합니다"
2. "문맥에 없는 정보는 절대 추가하지 마세요"
3. "확실하지 않은 경우 '제공된 규정에서 해당 정보를 찾을 수 없습니다'라고 답변하세요"

**Acceptance Criteria:**
- [ ] 모든 RAG 프롬프트에 context-only 강제 조항 추가
- [ ] 외부 지식 사용 금지 명시적 선언
- [ ] 불확실한 경우 fallback 응답 템플릿 제공

### REQ-002: Post-Generation Faithfulness Validation

**WHEN** 답변이 생성된 후
**THE SYSTEM SHALL** 다음 검증을 수행:
1. 답변의 각 핵심 주장이 컨텍스트 문서에 존재하는지 확인
2. 검증되지 않은 주장이 있는 경우 낮은 faithfulness 점수 부여
3. Faithfulness < 0.6인 경우 답변 재생성 또는 경고

**IF** Faithfulness 점수가 0.6 미만이면
**THEN** 시스템은 다음 중 하나를 수행:
- 답변 재생성 (최대 2회)
- 또는 "정보를 찾을 수 없습니다" 응답으로 대체

**Acceptance Criteria:**
- [ ] FaithfulnessValidator 클래스 구현
- [ ] 모든 답변에 대해 post-generation 검증 수행
- [ ] Faithfulness < 0.6인 답변 자동 재생성 또는 fallback

### REQ-003: Answer Grounding with Citations

**WHEN** 답변에 사실적 주장이 포함될 때
**THE SYSTEM SHALL** 각 주장에 대해 다음을 수행:
1. 주장과 매칭되는 컨텍스트 구간 식별
2. 해당 구간의 규정명과 조항 번호 추출
3. 주장 뒤에 인용 형식으로 출처 표기

**Acceptance Criteria:**
- [ ] 모든 핵심 주장에 출처 인용 자동 생성
- [ ] 인용이 불가능한 주장은 "확인 필요"로 표시
- [ ] 인용 품질 평가 메트릭 0.70+ 달성

### REQ-004: Fallback Response for Insufficient Context

**WHEN** 검색된 컨텍스트가 질문에 충분히 답할 수 없을 때
**THE SYSTEM SHALL** 다음 fallback 응답을 제공:
- "제공된 규정에서 해당 정보를 찾을 수 없습니다"
- 관련될 수 있는 다른 질문 제안
- 담당 부서 문의 안내

**Acceptance Criteria:**
- [ ] 컨텍스트 관련성 점수 < 0.5인 경우 fallback 트리거
- [ ] 명확한 정보 부족 메시지 제공
- [ ] 사용자 안내를 위한 대안 제시

---

## Technical Approach

### Phase 1: Prompt Engineering Enhancement

**목표**: 프롬프트를 통해 LLM의 외부 지식 사용 방지

**변경 파일**:
- `src/rag/application/search_usecase.py` - `_get_fallback_regulation_qa_prompt()` 강화
- `data/config/prompts.json` - regulation_qa 프롬프트 업데이트

**접근 방식**:
1. **System Prompt 강화**:
   ```python
   # 추가할 프롬프트 섹션
   """
   ## 절대 규칙 (Violation Detection)

   당신은 제공된 [CONTEXT] 섹션의 정보만 사용하여 답변해야 합니다.
   다음은 절대 금지됩니다:

   1. [CONTEXT]에 없는 전화번호, 이메일, 부서명 생성
   2. [CONTEXT]에 없는 규정 인용 (제X조)
   3. [CONTEXT]에 없는 수치, 날짜, 기간 언급
   4. 일반적인 대학 규정이나 타 학교 사례 언급
   5. "일반적으로", "보통", "통상적으로" 등의 회피 표현

   정보가 [CONTEXT]에 없으면 반드시 다음과 같이 답변하세요:
   "제공된 규정에서 해당 정보를 찾을 수 없습니다. 관련 부서에 문의해 주시기 바랍니다."
   """
   ```

2. **Context Delimiter 명확화**:
   ```
   [CONTEXT START - 반드시 이 내용만 참조하세요]
   {context_chunks}
   [CONTEXT END - 이 범위 밖의 정보는 사용하지 마세요]
   ```

### Phase 2: Post-Generation Faithfulness Validation

**목표**: 생성된 답변의 컨텍스트 근거성 검증

**새 파일**:
- `src/rag/domain/evaluation/faithfulness_validator.py`

**접근 방식**:
```python
class FaithfulnessValidator:
    """답변의 Faithfulness를 검증하는 클래스"""

    def validate_answer(
        self,
        answer: str,
        context: List[str],
        threshold: float = 0.6
    ) -> FaithfulnessValidationResult:
        """
        답변이 컨텍스트에 근거하는지 검증

        Returns:
            - score: 0.0 ~ 1.0 (1.0 = 완전히 근거)
            - is_acceptable: threshold 이상인지 여부
            - ungrounded_claims: 근거 없는 주장 목록
        """
        # 1. 답변에서 핵심 주장 추출
        claims = self._extract_claims(answer)

        # 2. 각 주장이 컨텍스트에 존재하는지 확인
        grounded_claims = []
        ungrounded_claims = []

        for claim in claims:
            if self._is_claim_in_context(claim, context):
                grounded_claims.append(claim)
            else:
                ungrounded_claims.append(claim)

        # 3. Faithfulness 점수 계산
        score = len(grounded_claims) / len(claims) if claims else 1.0

        return FaithfulnessValidationResult(
            score=score,
            is_acceptable=score >= threshold,
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            suggestion=self._generate_suggestion(ungrounded_claims)
        )
```

### Phase 3: Answer Regeneration Loop

**목표**: 낮은 Faithfulness 답변 자동 재생성

**변경 파일**:
- `src/rag/application/search_usecase.py` - `_generate_answer_with_validation()` 메서드 추가

**접근 방식**:
```python
async def _generate_answer_with_validation(
    self,
    query: str,
    context: List[str],
    max_retries: int = 2
) -> Answer:
    """Faithfulness 검증을 포함한 답변 생성"""

    for attempt in range(max_retries + 1):
        # 1. 답변 생성
        answer = await self._generate_answer(query, context)

        # 2. Faithfulness 검증
        validation = self.faithfulness_validator.validate_answer(
            answer.content, context
        )

        if validation.is_acceptable:
            return answer

        # 3. 재시도 시 더 엄격한 프롬프트 사용
        if attempt < max_retries:
            logger.warning(
                f"Low faithfulness ({validation.score:.2f}), "
                f"regenerating answer (attempt {attempt + 1})"
            )
            # 더 엄격한 프롬프트로 재생성
            answer = await self._generate_answer_strict(query, context, validation)

    # 4. 모든 시도 실패 시 fallback
    return Answer(
        content=FALLBACK_MESSAGE_KO,
        confidence=0.0,
        metadata={"reason": "faithfulness_below_threshold"}
    )
```

### Phase 4: Citation Grounding Enforcement

**목표**: 모든 사실적 주장에 인용 연결

**접근 방식**:
- 답변 생성 시 각 문장별 출처 컨텍스트 매핑
- 출처가 확인되지 않은 문장은 "확인 필요" 태그 추가
- 기존 `HallucinationFilter`와 통합

---

## Implementation Plan

### Milestone 1: Prompt Enhancement (Primary Goal)

| Task | Priority | Description |
|------|----------|-------------|
| P1-1 | High | `_get_fallback_regulation_qa_prompt()` 강화 - context-only 규칙 추가 |
| P1-2 | High | `prompts.json` regulation_qa 프롬프트 업데이트 |
| P1-3 | Medium | Context delimiter 명확화 ([CONTEXT START/END]) |
| P1-4 | Medium | 프롬프트 변경 테스트 케이스 작성 |

### Milestone 2: Faithfulness Validation (Primary Goal)

| Task | Priority | Description |
|------|----------|-------------|
| P2-1 | High | `FaithfulnessValidator` 클래스 구현 |
| P2-2 | High | 주장 추출 (`_extract_claims`) 로직 구현 |
| P2-3 | High | 컨텍스트 매칭 (`_is_claim_in_context`) 로직 구현 |
| P2-4 | Medium | 검증 결과 데이터 구조 정의 |

### Milestone 3: Regeneration Loop (Secondary Goal)

| Task | Priority | Description |
|------|----------|-------------|
| P3-1 | High | `_generate_answer_with_validation()` 메서드 구현 |
| P3-2 | Medium | 더 엄격한 프롬프트 버전 (`_generate_answer_strict`) 구현 |
| P3-3 | Medium | 최대 재시도 횟수 설정 (기본값: 2) |

### Milestone 4: Integration & Testing (Secondary Goal)

| Task | Priority | Description |
|------|----------|-------------|
| P4-1 | High | `SearchUseCase`에 FaithfulnessValidator 통합 |
| P4-2 | High | 통합 테스트 작성 (샘플 질문으로 검증) |
| P4-3 | Medium | 기존 `HallucinationFilter`와 통합 |
| P4-4 | Medium | 평가 스크립트로 Faithfulness 개선 확인 |

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Faithfulness | 30% | 60%+ | RAGAS Faithfulness metric |
| Contextual Recall | 87% | 80%+ (maintain) | RAGAS ContextRecall metric |
| Pass Rate | 0% | 50%+ | Overall pass rate |
| Answer Relevancy | 53% | 70%+ | RAGAS AnswerRelevancy metric |
| Contextual Precision | 50% | 65%+ | RAGAS ContextPrecision metric |

### Quality Gates

- **Faithfulness >= 0.60**: Critical - 이것이 본 SPEC의 핵심 목표
- **Recall >= 0.80**: Must maintain - 현재 87% 수준 유지
- **Pass Rate >= 50%**: Target - 전체 품질 개선 지표

---

## Dependencies

- SPEC-RAG-QUALITY-006: IntentClassifier and CitationValidator (COMPLETED)
- SPEC-RAG-QUALITY-007: Context Relevance Improvement (COMPLETED)
- `HallucinationFilter`: 이미 구현됨, 통합 필요

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 프롬프트 강화로 인한 답변 과단축 | Medium | Medium | 적절한 fallback 메시지 제공 |
| 재생성 루프로 인한 응답 시간 증가 | High | Medium | 캐싱 및 최대 2회 재시도 제한 |
| FaithfulnessValidator 정확도 | Medium | High | 한국어 NLP 최적화, 테스트 케이스 다양화 |
| Recall 저하 가능성 | Low | High | 프롬프트 변경 후 Recall 모니터링 |

---

## References

- Evaluation Data: `data/evaluations/` (2026-02-20)
- Previous SPEC: SPEC-RAG-QUALITY-007
- Related Files:
  - `src/rag/application/search_usecase.py`
  - `src/rag/application/hallucination_filter.py`
  - `src/rag/domain/evaluation/quality_evaluator.py`
  - `data/config/prompts.json`

---

## Implementation Notes

### Completed: 2026-02-21

### Implementation Summary

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| REQ-001: Strict Context-Only Prompt Engineering | Completed | prompts.json v2.3, Context Delimiter 추가 |
| REQ-002: Post-Generation Faithfulness Validation | Completed | FaithfulnessValidator 클래스 구현 |
| REQ-003: Answer Grounding with Citations | Completed | SearchUseCase 통합 |
| REQ-004: Fallback Response for Insufficient Context | Completed | 재생성 루프 구현 |

### Files Created

| File | Description |
|------|-------------|
| `src/rag/domain/evaluation/faithfulness_validator.py` | Faithfulness 검증 클래스 (97.37% 커버리지) |
| `tests/rag/unit/domain/evaluation/test_faithfulness_validator.py` | 29개 단위 테스트 |
| `tests/rag/unit/application/test_regeneration_loop.py` | 17개 재생성 루프 테스트 |
| `tests/integration/test_faithfulness_flow.py` | 15개 통합 테스트 |

### Files Modified

| File | Changes |
|------|---------|
| `data/config/prompts.json` | 버전 2.2 → 2.3, Context Delimiter 및 Strict Rules 추가 |
| `src/rag/application/search_usecase.py` | 재생성 루프, FaithfulnessValidator 통합 |

### Test Results

- **Total Tests**: 61 new tests
- **Pass Rate**: 100% (61/61)
- **Coverage**: FaithfulnessValidator 97.37%

### Commit

```
a77772f feat(rag): implement FaithfulnessValidator for hallucination prevention (SPEC-RAG-QUALITY-008)
```

### Next Steps

RAG 평가 실행으로 Faithfulness 개선 효과 검증:
```bash
uv run python scripts/verify_evaluation_metrics.py --full-eval
```

