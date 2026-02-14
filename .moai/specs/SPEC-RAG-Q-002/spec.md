# SPEC-RAG-Q-002: Hallucination Prevention

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-Q-002 |
| **Title** | Hallucination Prevention |
| **Status** | Draft |
| **Priority** | High |
| **Created** | 2026-02-13 |
| **Source** | RAG Quality Evaluation (rag_quality_local_20260213) |

---

## Problem Analysis

### Current Issue

RAG 시스템의 LLM이 검색된 문맥에 없는 정보를 생성(hallucination)하여 답변의 신뢰성을 저하시키고 있습니다.

### Evidence from Evaluation

| Issue | Frequency | Severity |
|-------|-----------|----------|
| 가짜 연락처 생성 (02-XXXX-XXXX) | 8 queries | HIGH |
| 문맥에 없는 부서명 언급 | 5 queries | MEDIUM |
| 존재하지 않는 규정 조항 인용 | 4 queries | MEDIUM |

### Example Cases

```
Query: "연구년 신청 문의처가 어디인가요?"
Bad Response: "연구년 신청은 학술연구지원팀(02-1234-5678)로 문의하세요."
Issues:
  - '학술연구지원팀'이 검색 결과에 없음
  - '02-1234-5678'은 실제 연락처가 아님
```

---

## Requirements (EARS Format)

### REQ-001: Contact Information Validation

**WHEN** LLM이 연락처 정보(전화번호, 이메일, 팩스)를 생성하려 할 때

**IF** 해당 정보가 검색된 문맥에 존재하면

**THEN** 문맥에서 확인된 연락처만 사용

**ELSE** "자세한 연락처는 해당 부서에 직접 문의해 주시기 바랍니다"로 응답

### REQ-002: Department Name Validation

**WHEN** LLM이 부서명이나 조직명을 언급할 때

**IF** 해당 부서명이 검색된 문맥에 명시되어 있으면

**THEN** 문맥에서 확인된 부서명 사용

**ELSE** "담당 부서"와 같은 일반적 표현 사용

**SHALL NOT** 문맥에 없는 구체적 부서명 생성

### REQ-003: Citation Grounding

**WHEN** LLM이 규정 조항을 인용할 때

**IF** 인용된 조항이 검색 결과에 존재하면

**THEN** 해당 조항 내용도 함께 제공

**ELSE** 인용을 제외하거나 "관련 규정"으로 일반화

---

## Acceptance Criteria

### AC-001: No Hallucinated Contact Information

- [ ] 모든 연락처는 실제 검색 결과에서 출처 확인 가능
- [ ] "02-XXXX-XXXX" 형식의 가짜 연락처 생성率为 0%
- [ ] 없는 연락처는 "담당 부서 문의" 안내로 대체

### AC-002: Verified Department References

- [ ] 언급된 모든 부서명은 검색 결과에서 확인 가능
- [ ] 문맥에 없는 부서명 생성率为 0%

### AC-003: Grounded Citations

- [ ] 모든 규정 인용은 검색 결과에 존재하는 조항
- [ ] Hallucination 관련 평가 실패 케이스 80% 감소

### AC-004: Quality Metrics Improvement

- [ ] Accuracy 점수 0.85+ 달성 (현재 0.807)
- [ ] 전체 Pass Rate 60%+ 달성 (현재 43.3%)

---

## Technical Approach

### Option A: Post-Processing Filter (Recommended)

1. LLM 응답에서 연락처, 부서명, 조항 인용 패턴 추출
2. 각 항목이 검색 결과 문맥에 존재하는지 검증
3. 검증되지 않은 항목은 제거 또는 일반화

```python
class HallucinationFilter:
    def validate_response(self, response: str, context: list[str]) -> str:
        # Extract phone numbers, emails, department names
        # Validate against context
        # Return sanitized response
        pass
```

### Option B: Prompt Engineering

시스템 프롬프트에 환각 방지 지침 추가:

```
중요 규칙:
- 검색 결과에 없는 연락처는 절대 생성하지 마세요.
- 검색 결과에 없는 부서명은 구체적으로 언급하지 마세요.
- 확실하지 않은 정보는 "해당 규정에 명시되어 있지 않습니다"라고 답변하세요.
```

### Option C: Hybrid Approach

프롬프트 엔지니어링 + 후처리 필터 조합

---

## Dependencies

- 검색 결과(context) 접근 가능한 LLM 응답 생성 파이프라인
- 연락처/부서명 정규표현식 패턴 라이브러리
- 기존 부서/연락처 데이터베이스 (선택적)

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 과도한 필터링으로 유용한 정보 손실 | Medium | Medium | 신중한 검증 로직 설계 |
| 응답 생성 속도 저하 | Low | Low | 캐싱 및 비동기 처리 |
| 거짓 음성(정상 정보 차단) | Medium | High | 화이트리스트 기반 예외 처리 |

---

## Related Documents

- [RAG Quality Evaluation Report](../../data/evaluations/rag_quality_local_report_20260213.md)
- [SPEC-RAG-Q-003: Deadline Information Enhancement](../SPEC-RAG-Q-003/spec.md)
- [SPEC-RAG-Q-004: Citation Verification](../SPEC-RAG-Q-004/spec.md)

---

**Last Updated:** 2026-02-13
