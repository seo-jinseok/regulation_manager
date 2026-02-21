# Implementation Plan: SPEC-RAG-QUALITY-008

## Faithfulness Enhancement for RAG System

**SPEC ID**: SPEC-RAG-QUALITY-008
**Created**: 2026-02-20
**Priority**: Critical

---

## Overview

본 계획은 RAG 시스템의 Faithfulness(신뢰도)를 30%에서 60% 이상으로 개선하기 위한 구현 전략을 정의합니다. 핵심 문제는 높은 Recall(87%)에도 불구하고 낮은 Faithfulness(30%)로, LLM이 컨텍스트에 없는 정보를 생성하는 할루시네이션 문제입니다.

---

## Technical Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SearchUseCase                            │
├─────────────────────────────────────────────────────────────┤
│  1. Query → Retrieval (Recall 87% - Good)                   │
│  2. Context + Enhanced Prompt → LLM                         │
│  3. Generated Answer → FaithfulnessValidator (NEW)          │
│  4. If score < 0.6: Regenerate with strict prompt           │
│  5. Final Answer → HallucinationFilter → Output             │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────┐              ┌───────────────────────────┐
│ FaithfulnessValidator           │ Enhanced Prompts          │
│ (NEW)                           │ - Context-only rules      │
│ - Claim extraction              │ - External knowledge ban  │
│ - Context matching              │ - Fallback templates      │
│ - Score calculation             │ - Clear delimiters        │
└──────────────────┘              └───────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Prompt Engineering Enhancement

**Duration**: Priority High
**Files to Modify**:
- `src/rag/application/search_usecase.py`
- `data/config/prompts.json`

**Tasks**:

1. **Context-Only 프롬프트 규칙 추가**
   - `_get_fallback_regulation_qa_prompt()` 함수 수정
   - 새로운 "절대 규칙" 섹션 추가:
     ```
     ## 절대 규칙 (Strict Grounding)
     당신은 제공된 [CONTEXT] 섹션의 정보만 사용하여 답변해야 합니다.
     [CONTEXT]에 없는 정보는 절대 생성하지 마세요.
     ```

2. **Context Delimiter 명확화**
   - `[CONTEXT START]` / `[CONTEXT END]` 태그 추가
   - LLM이 컨텍스트 경계를 명확히 인식하도록 개선

3. **Fallback 메시지 강화**
   - "제공된 규정에서 찾을 수 없습니다" 명확화
   - 관련 부서 문의 안내 추가

**Expected Impact**: Faithfulness +10-15%p

### Phase 2: FaithfulnessValidator Implementation

**Duration**: Priority High
**New Files**:
- `src/rag/domain/evaluation/faithfulness_validator.py`

**Design**:

```python
@dataclass
class Claim:
    """단일 주장을 나타내는 데이터 클래스"""
    text: str
    claim_type: ClaimType  # FACTUAL, NUMERICAL, CITATION, CONTACT
    source_span: Optional[Tuple[int, int]] = None
    is_grounded: bool = False

@dataclass
class FaithfulnessValidationResult:
    """검증 결과"""
    score: float  # 0.0 ~ 1.0
    is_acceptable: bool
    claims: List[Claim]
    grounded_count: int
    ungrounded_count: int
    suggestions: List[str]

class FaithfulnessValidator:
    """답변의 Faithfulness 검증"""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._claim_patterns = self._init_claim_patterns()

    def validate(
        self,
        answer: str,
        context: List[str]
    ) -> FaithfulnessValidationResult:
        """답변 검증 수행"""
        # 1. 주장 추출
        claims = self._extract_claims(answer)

        # 2. 각 주장의 근거 확인
        context_text = " ".join(context)
        for claim in claims:
            claim.is_grounded = self._check_groundedness(
                claim, context_text
            )

        # 3. 점수 계산
        grounded = sum(1 for c in claims if c.is_grounded)
        score = grounded / len(claims) if claims else 1.0

        return FaithfulnessValidationResult(
            score=score,
            is_acceptable=score >= self.threshold,
            claims=claims,
            grounded_count=grounded,
            ungrounded_count=len(claims) - grounded,
            suggestions=self._generate_suggestions(claims)
        )

    def _extract_claims(self, text: str) -> List[Claim]:
        """텍스트에서 핵심 주장 추출"""
        claims = []

        # 1. 인용 패턴 (제X조)
        citations = re.findall(r'제\d+조(?:제\d+항)?', text)
        for c in citations:
            claims.append(Claim(text=c, claim_type=ClaimType.CITATION))

        # 2. 수치 패턴 (날짜, 기간, 퍼센트)
        numbers = re.findall(r'\d+(?:일|개월|%|원|명|점)', text)
        for n in numbers:
            claims.append(Claim(text=n, claim_type=ClaimType.NUMERICAL))

        # 3. 연락처 패턴
        contacts = re.findall(r'\d{2,3}-\d{3,4}-\d{4}', text)
        for c in contacts:
            claims.append(Claim(text=c, claim_type=ClaimType.CONTACT))

        # 4. 문장 단위 핵심 주장
        sentences = text.split('다.')
        for s in sentences:
            if self._is_factual_claim(s):
                claims.append(Claim(text=s + '다', claim_type=ClaimType.FACTUAL))

        return claims

    def _check_groundedness(self, claim: Claim, context: str) -> bool:
        """주장이 컨텍스트에 근거하는지 확인"""
        if claim.claim_type == ClaimType.CITATION:
            # 인용은 정확히 일치해야 함
            return claim.text in context

        elif claim.claim_type == ClaimType.NUMERICAL:
            # 수치는 근처 문맥과 함께 확인
            return self._fuzzy_match(claim.text, context)

        elif claim.claim_type == ClaimType.CONTACT:
            # 연락처는 정확히 일치
            normalized = re.sub(r'\D', '', claim.text)
            return normalized in re.sub(r'\D', '', context)

        else:  # FACTUAL
            # 문장은 키워드 기반 매칭
            return self._semantic_match(claim.text, context)
```

**Expected Impact**: Faithfulness 검증 정확도 90%+

### Phase 3: Regeneration Loop

**Duration**: Priority Medium
**Files to Modify**:
- `src/rag/application/search_usecase.py`

**Design**:

```python
class SearchUseCase:
    # ... existing code ...

    async def _generate_answer_with_validation(
        self,
        query: str,
        context: List[str],
        max_retries: int = 2
    ) -> Answer:
        """Faithfulness 검증을 포함한 답변 생성"""

        for attempt in range(max_retries + 1):
            # 프롬프트 선택 (재시도 시 더 엄격한 버전)
            if attempt == 0:
                prompt = self._get_standard_prompt()
            else:
                prompt = self._get_strict_prompt(attempt)

            # 답변 생성
            answer = await self._generate_answer(query, context, prompt)

            # Faithfulness 검증
            validation = self.faithfulness_validator.validate(
                answer.content, context
            )

            if validation.is_acceptable:
                answer.metadata["faithfulness_score"] = validation.score
                answer.metadata["validation_attempts"] = attempt
                return answer

            logger.warning(
                f"Low faithfulness ({validation.score:.2f}), "
                f"ungrounded claims: {validation.ungrounded_count}, "
                f"attempt {attempt + 1}/{max_retries}"
            )

        # 모든 시도 실패 시 fallback
        return self._create_fallback_answer(query, validation)

    def _get_strict_prompt(self, attempt: int) -> str:
        """재시도용 더 엄격한 프롬프트"""
        strictness_levels = [
            "제공된 문맥의 정보만 사용하세요.",
            "⚠️ 중요: 반드시 문맥에 명시된 내용만 답변하세요. "
            "문맥에 없는 내용은 절대 추가하지 마세요.",
            "🚨 절대 규칙: 문맥에 없는 정보는 생성하지 마세요. "
            "확실하지 않으면 '제공된 규정에서 찾을 수 없습니다'라고만 답변하세요."
        ]
        return strictness_levels[min(attempt, len(strictness_levels) - 1)]

    def _create_fallback_answer(
        self,
        query: str,
        validation: FaithfulnessValidationResult
    ) -> Answer:
        """Fallback 응답 생성"""
        return Answer(
            content=(
                "제공된 규정에서 해당 질문에 대한 명확한 정보를 찾을 수 없습니다.\n\n"
                "관련 부서에 직접 문의해 주시기 바랍니다:\n"
                "- 학적팀: 학사 관련 문의\n"
                "- 장학팀: 장학금 관련 문의\n"
                "- 교무처: 교원 관련 문의"
            ),
            confidence=0.0,
            metadata={
                "faithfulness_score": validation.score,
                "fallback": True,
                "reason": "faithfulness_below_threshold"
            }
        )
```

**Expected Impact**: Faithfulness < 0.6인 답변 자동 재처리

### Phase 4: Integration & Testing

**Duration**: Priority Medium
**Tasks**:

1. **SearchUseCase 통합**
   - `_generate_answer_with_validation()` 메서드 호출 지점 변경
   - 기존 `generate_answer()` 호출을 새 메서드로 대체

2. **HallucinationFilter 통합**
   - `FaithfulnessValidator` 결과를 `HallucinationFilter`에 전달
   - 검증 결과에 따른 필터링 동작 조정

3. **테스트 케이스 작성**
   - FaithfulnessValidator 단위 테스트
   - SearchUseCase 통합 테스트
   - 샘플 질문으로 end-to-end 검증

4. **평가 스크립트 실행**
   - `scripts/verify_evaluation_metrics.py`로 개선 확인
   - Faithfulness 60%+ 달성 여부 검증

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/rag/application/search_usecase.py` | Modify | Prompt 강화, 재생성 루프 추가 |
| `src/rag/domain/evaluation/faithfulness_validator.py` | New | Faithfulness 검증 클래스 |
| `src/rag/domain/evaluation/__init__.py` | Modify | FaithfulnessValidator export 추가 |
| `data/config/prompts.json` | Modify | regulation_qa 프롬프트 업데이트 |
| `tests/unit/test_faithfulness_validator.py` | New | 검증 클래스 단위 테스트 |
| `tests/integration/test_faithfulness_flow.py` | New | 통합 테스트 |

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_faithfulness_validator.py

class TestFaithfulnessValidator:

    def test_citation_extraction(self):
        """인용 패턴 추출 테스트"""
        validator = FaithfulnessValidator()
        claims = validator._extract_claims(
            "휴학은 학칙 제40조에 따라 신청해야 합니다."
        )
        assert any(c.text == "제40조" for c in claims)

    def test_numerical_extraction(self):
        """수치 패턴 추출 테스트"""
        validator = FaithfulnessValidator()
        claims = validator._extract_claims(
            "30일 이내에 신청해야 하며, 50%의 장학금이 지급됩니다."
        )
        assert any("30일" in c.text for c in claims)
        assert any("50%" in c.text for c in claims)

    def test_groundedness_check(self):
        """근거 확인 테스트"""
        validator = FaithfulnessValidator()
        context = ["학칙 제40조에 따르면 휴학은 학기 시작 전에 신청해야 합니다."]
        claims = [Claim(text="제40조", claim_type=ClaimType.CITATION)]

        is_grounded = validator._check_groundedness(claims[0], context[0])
        assert is_grounded == True

    def test_validation_score(self):
        """전체 검증 점수 테스트"""
        validator = FaithfulnessValidator(threshold=0.6)
        context = ["학칙 제40조: 휴학은 학기 시작 전에 신청해야 합니다."]

        result = validator.validate(
            "휴학은 제40조에 따라 학기 시작 전에 신청해야 합니다.",
            context
        )
        assert result.score >= 0.6
        assert result.is_acceptable == True

    def test_ungrounded_claim_detection(self):
        """근거 없는 주장 감지 테스트"""
        validator = FaithfulnessValidator(threshold=0.6)
        context = ["학칙 제40조: 휴학 관련 내용"]

        result = validator.validate(
            "휴학은 제40조에 따르며, 문의처는 02-1234-5678입니다.",
            context
        )
        # 전화번호가 컨텍스트에 없으므로 점수 감소
        assert "02-1234-5678" not in context[0]
        # 인용은 있지만 전화번호가 없으므로 점수 < 1.0
        assert result.score < 1.0
```

### Integration Tests

```python
# tests/integration/test_faithfulness_flow.py

class TestFaithfulnessFlow:

    @pytest.mark.asyncio
    async def test_regenerate_on_low_faithfulness(self):
        """낮은 Faithfulness 시 재생성 테스트"""
        usecase = SearchUseCase(...)

        # 의도적으로 낮은 Faithfulness를 유발하는 컨텍스트
        result = await usecase.search("장학금 문의처는?")

        # Fallback 메시지 확인
        if result.answer.metadata.get("fallback"):
            assert "찾을 수 없습니다" in result.answer.content

    @pytest.mark.asyncio
    async def test_high_faithfulness_passes(self):
        """높은 Faithfulness 답변 통과 테스트"""
        usecase = SearchUseCase(...)

        result = await usecase.search("휴학 신청 기간은?")

        # Faithfulness 점수가 기록되어야 함
        assert "faithfulness_score" in result.answer.metadata
        assert result.answer.metadata["faithfulness_score"] >= 0.6
```

---

## Rollback Plan

1. **Prompt 변경 롤백**
   - `prompts.json` 이전 버전 복원
   - `_get_fallback_regulation_qa_prompt()` 원복

2. **Validator 비활성화**
   - `FaithfulnessValidator` 호출 주석 처리
   - 기존 `generate_answer()` 직접 호출로 복원

3. **재생성 루프 비활성화**
   - `max_retries = 0`으로 설정
   - 즉시 fallback 반환

---

## Success Criteria

- [ ] FaithfulnessValidator 구현 완료
- [ ] 모든 단위 테스트 통과 (커버리지 85%+)
- [ ] 통합 테스트 통과
- [ ] Faithfulness 점수 60%+ 달성
- [ ] Recall 점수 80%+ 유지
- [ ] Pass Rate 50%+ 달성

---

## Timeline

| Phase | Priority | Dependencies |
|-------|----------|--------------|
| Phase 1: Prompt Enhancement | Primary Goal | None |
| Phase 2: Validator Implementation | Primary Goal | None |
| Phase 3: Regeneration Loop | Secondary Goal | Phase 2 |
| Phase 4: Integration & Testing | Secondary Goal | Phase 1-3 |

---

## Notes

- 본 SPEC은 SPEC-RAG-QUALITY-007의 후속으로, Faithfulness 저하 문제를 직접 해결
- 기존 `HallucinationFilter`와 중복되지 않도록 역할 분담:
  - `HallucinationFilter`: 생성 후 패턴 기반 필터링
  - `FaithfulnessValidator`: 생성 전/후 의미 기반 검증
