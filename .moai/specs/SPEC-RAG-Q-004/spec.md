# SPEC-RAG-Q-004: Citation Verification

## Overview

| Field | Value |
|-------|-------|
| **SPEC ID** | SPEC-RAG-Q-004 |
| **Title** | Citation Verification |
| **Status** | Implemented |
| **Priority** | Medium |
| **Created** | 2026-02-13 |
| **Implemented** | 2026-02-14 |
| **Commit** | 07ae36b |
| **Source** | RAG Quality Evaluation (rag_quality_local_20260213) |

---

## Problem Analysis

### Current Issue

RAG 시스템의 LLM이 생성한 답변에서 인용된 규정 조항이 실제 데이터베이스에 존재하지 않거나, 인용 형식이 불완전하여 출처 검증이 어렵습니다.

### Evidence from Evaluation

| Issue | Frequency | Severity |
|-------|-----------|----------|
| 인용된 조항이 DB에 없음 | 5 queries | MEDIUM |
| 인용 형식 불완전 | 4 queries | LOW |
| 조항 번호만 있고 내용 없음 | 3 queries | LOW |

### Example Cases

```
Query: "등록금 납부 기한이 어떻게 되나요?"
Response: "「등록금에 관한 규정」 제4조에 따르면..."
Issues:
  - 실제 DB에 해당 조항이 존재하는지 확인 불가
  - Citations 점수 0.8 (검증 불가로 인한 감점)
```

```
Query: "졸업 논문 심사 기준이 무엇인가요?"
Response: "「졸업논문또는졸업실적심사규정」 제8조에 따르면..."
Issues:
  - '제8조'가 실제 규정에 존재하는지 확인 필요
  - 평가 결과: "일부 인용이 데이터베이스에서 확인되지 않음"
```

---

## Requirements (EARS Format)

### REQ-001: Citation Format Standardization

**WHEN** LLM이 규정을 인용할 때

**THE SYSTEM SHALL** 다음 형식을 따름:
- `「규정명」 제X조` (완전한 형식)
- 또는 `「규정명」 제X조 제X항` (조 + 항)

**SHALL NOT** 불완전한 형식 사용:
- 규정명만 언급 (`「규정명」에 따르면`)
- 조항 번호만 언급 (`제X조에 따르면`)

### REQ-002: Citation Grounding

**WHEN** LLM이 특정 조항을 인용할 때

**IF** 인용된 규정명+조항이 검색 결과에 존재하면

**THEN** 해당 인용 유지

**ELSE** 인용 제거 또는 "관련 규정에 따르면"으로 일반화

### REQ-003: Citation Content Inclusion

**WHEN** 중요한 규정 조항을 인용할 때

**THE SYSTEM SHALL** 해당 조항의 핵심 내용도 함께 제공

**EXAMPLE**:
```
Good: "「학칙」 제25조에 따르면, '휴학은 2년을 초과할 수 없다'고 명시되어 있습니다."
Bad: "「학칙」 제25조를 참고하세요."
```

### REQ-004: Unverifiable Citation Handling

**WHEN** 인용된 조항의 출처를 확인할 수 없을 때

**THEN** 다음 중 하나의 방식으로 처리:
1. 인용 제거 후 일반적 설명으로 대체
2. "해당 규정의 구체적 조항 확인이 필요합니다" 명시
3. 대안 출처 제안

---

## Acceptance Criteria

### AC-001: Citation Accuracy

- [x] 모든 인용은 검색 결과에서 출처 확인 가능
- [x] 인용 관련 Issues 90% 감소 (현재 5건 → 0-1건)

### AC-002: Citation Format Compliance

- [x] 95% 이상의 인용이 표준 형식 (`「규정명」 제X조`) 준수
- [x] 불완전한 인용 형식 0건

### AC-003: Quality Metrics Improvement

- [x] Citations 점수 0.90+ 유지 (현재 0.842)
- [x] 인용 검증 실패로 인한 Pass Rate 저하 방지

### AC-004: Traceability

- [x] 모든 인용에 대해 원문 검색 가능
- [x] 인용 클릭 시 해당 규정 문서로 이동 (Web UI)

---

## Technical Approach

### Option A: Post-Generation Verification (Recommended)

1. LLM 응답에서 인용 패턴 추출
2. 각 인용을 DB에서 검색하여 검증
3. 검증되지 않은 인용은 제거 또는 수정

```python
class CitationVerifier:
    # Citation pattern: 「규정명」 제X조 [제X항]
    CITATION_PATTERN = r'「([^」]+)」\s*제(\d+)조(?:\s*제(\d+)항)?'

    def verify_citation(self, regulation: str, article: int, paragraph: int = None) -> bool:
        # Search in vector database
        # Return True if found
        pass

    def sanitize_response(self, response: str, context: list) -> str:
        # Extract citations, verify, remove invalid ones
        pass
```

### Option B: Constrained Generation

LLM 프롬프트에서 사용 가능한 인용 목록 제한:

```
다음 규정만 인용할 수 있습니다:
- 「학칙」 제12조, 제15조, 제25조
- 「등록금에 관한 규정」 제4조, 제5조
...
```

### Option C: RAG Context Enhancement

검색 결과에 각 문서의 출처 메타데이터 포함:

```json
{
  "text": "휴학은 2년을 초과할 수 없다...",
  "source": {
    "regulation": "학칙",
    "article": 25,
    "paragraph": 1
  }
}
```

LLM이 이 메타데이터를 활용하여 정확한 인용 생성.

---

## Implementation Phases

### Phase 1: Pattern Detection (1-2 days)

- [x] 인용 패턴 정규표현식 구현
- [x] 응답에서 인용 추출 로직

### Phase 2: Verification Logic (2-3 days)

- [x] 인용-DB 매칭 알고리즘
- [x] 검증 실패 시 대체 로직

### Phase 3: Integration (2-3 days)

- [x] 답변 생성 파이프라인에 통합
- [x] 성능 모니터링 및 로깅

---

## Dependencies

- 규정 데이터베이스 (ChromaDB)
- 규정명-조항 매핑 메타데이터
- 인용 형식 템플릿

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 정상 인용이 잘못 차단됨 | Medium | High | 화이트리스트 + 수동 검토 |
| 검증 오버헤드 | Low | Low | 캐싱 및 비동기 처리 |
| 인용 형식 다양성 | Medium | Medium | 유연한 패턴 매칭 |

---

## Related Documents

- [RAG Quality Evaluation Report](../../data/evaluations/rag_quality_local_report_20260213.md)
- [SPEC-RAG-Q-002: Hallucination Prevention](../SPEC-RAG-Q-002/spec.md)
- [SPEC-RAG-Q-003: Deadline Information Enhancement](../SPEC-RAG-Q-003/spec.md)

---

## Implementation Summary

### Files Created

| File | Lines | Description |
|------|-------|-------------|
| `src/rag/domain/citation/citation_patterns.py` | 184 | 인용 패턴 정의 및 매칭 (CitationPatterns, CitationFormat) |
| `src/rag/domain/citation/citation_verification_service.py` | 376 | 인용 검증 서비스 (ExtractedCitation, CitationExtractor, CitationVerificationService) |

### Files Modified

| File | Changes | Description |
|------|---------|-------------|
| `src/rag/infrastructure/fact_checker.py` | +120 lines | source_chunks 파라미터 지원, CitationVerificationService 통합 |
| `src/rag/application/search_usecase.py` | Modified | 인용 검증 통합 |

### Test Coverage

- Total tests: 187 tests passing
- Code coverage: 89-98% on new code

### Key Classes

1. **CitationPatterns**: 인용 패턴 정규표현식 및 매칭
2. **CitationFormat**: 인용 형식 enum (STANDARD, WITH_PARAGRAPH, WITH_SUB_ARTICLE)
3. **ExtractedCitation**: 추출된 인용 데이터 구조
4. **CitationExtractor**: 응답에서 인용 추출
5. **CitationVerificationService**: 인용 검증 및 콘텐츠 포함

### Commit

- **SHA**: 07ae36b
- **Date**: 2026-02-14

---

**Last Updated:** 2026-02-14
