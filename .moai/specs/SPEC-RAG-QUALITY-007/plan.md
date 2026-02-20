# Implementation Plan: SPEC-RAG-QUALITY-007

## Metadata

| Field | Value |
|-------|-------|
| SPEC ID | SPEC-RAG-QUALITY-007 |
| Created | 2026-02-19 |
| Status | Ready for Implementation |
| Priority | High |
| Estimated Effort | 4.5 days |
| Dependencies | SPEC-RAG-QUALITY-006 (Completed) |

---

## Executive Summary

RAG 시스템 품질 개선을 위한 실행 계획으로, 핵심 메트릭(인용 정확성, 문맥 관련성, 합격률)을 목표치까지 개선합니다.

**Critical Findings from Code Review:**
1. ✅ CitationValidator는 이미 통합되어 작동 중
2. ❌ IntentClassifier는 검색 파이프라인에 미통합 (Root Cause #1)
3. ⚠️ Reranker 임계값이 너무 낮음 (0.15 → 0.25-0.30 필요)
4. ❓ 평가 메트릭 설정 확인 필요

---

## Requirements Mapping

### REQ-001: Citation Accuracy Enhancement (0.50 → 0.70)

**Current Status:** ✅ CitationValidator 통합됨 (search_usecase.py line 2504-2506)
- 강제 인용 생성 구현됨 (line 3551-3564)
- 형식 검증 "「규정명」 제X조" 구현됨

**Gap Analysis:**
- 인용은 생성되지만 평가 점수가 0.50인 이유:
  1. RAGAS 평가 메트릭 설정 문제 가능성
  2. 생성된 인용 형식이 평가 기대치와 불일치 가능성

**Action Items:**
- [ ] RAGAS 인용 메트릭 설정 검증
- [ ] 생성된 인용 형식 vs 평가 기대치 비교 분석
- [ ] 인용 밀도(citation density) 로깅 추가

### REQ-002: Context Relevance Improvement (0.50 → 0.75)

**Root Cause Identified:**
1. IntentClassifier 미통합 → 의도 기반 검색 부스팅 없음
2. MIN_RELEVANCE_THRESHOLD = 0.15 (너무 낮음)
   - 관련성 없는 문서가 컨텍스트에 포함됨
   - Line 40 in reranker.py: `MIN_RELEVANCE_THRESHOLD = 0.15`

**Action Items:**
- [ ] IntentClassifier를 검색 파이프라인에 통합
- [ ] 의도 카테고리별 검색 파라미터 조정 로직 구현
- [ ] MIN_RELEVANCE_THRESHOLD를 0.25-0.30으로 상향 조정
- [ ] 관련성 점수 로깅 및 모니터링 추가

### REQ-003: Pass Rate Improvement (0% → 80%)

**Current Failure Analysis (from evaluation_20260219_210750.json):**
```json
{
  "faithfulness": 0.5,           // FAIL (< 0.6)
  "answer_relevancy": 0.5,       // FAIL (< 0.7)
  "contextual_precision": 0.5,   // FAIL (< 0.65)
  "contextual_recall": 0.87      // PASS
}
```

**0.50 Uniform Scores → Evaluation Default Values:**
- 세 메트릭이 모두 0.50인 것은 실제 측정값이 아닌 기본값일 가능성
- 원인: RAGAS 평가 인프라 문제 (chromadb 미설치 등)

**Action Items:**
- [ ] RAGAS 평가 환경 검증 (chromadb 설치 확인)
- [ ] 0.50 점수의 실제 원인 파악
- [ ] 소규모 테스트(10개 쿼리)로 평가 메트릭 검증

---

## Technical Approach

### Phase 1: IntentClassifier Integration (2 days)

**Files to Modify:**
1. `src/rag/application/search_usecase.py`
   - IntentClassifier import 추가
   - `_get_search_parameters()` 메서드에 의도 기반 로직 추가
   - 검색 파라미터(top_k, rerank_threshold 등) 동적 조정

**Implementation Steps:**

```python
# search_usecase.py
from .intent_classifier import IntentClassifier, IntentCategory

class SearchUseCase:
    def __init__(self, ...):
        # ... existing init ...
        self._intent_classifier = IntentClassifier(confidence_threshold=0.5)

    def _get_intent_aware_search_params(
        self, query: str
    ) -> dict:
        """의도에 따른 검색 파라미터 조정"""
        result = self._intent_classifier.classify(query)

        params = {
            "top_k": 10,
            "rerank_threshold": 0.25,
            "boost_factors": {}
        }

        if result.category == IntentCategory.PROCEDURE:
            # 절차 관련 쿼리는 더 많은 컨텍스트 필요
            params["top_k"] = 15
            params["boost_factors"] = {"procedure": 1.5}

        elif result.category == IntentCategory.DEADLINE:
            # 기한 관련 쿼리는 날짜 정보 강조
            params["boost_factors"] = {"date": 1.3, "period": 1.3}

        elif result.category == IntentCategory.ELIGIBILITY:
            # 자격 관련 쿼리는 조건문 강조
            params["boost_factors"] = {"condition": 1.4}

        return params
```

**Testing:**
- [ ] IntentClassifier 정확도 테스트 (90% 목표)
- [ ] 의도별 검색 결과 품질 비교
- [ ] Context Relevance 점수 변화 측정

### Phase 2: Reranker Threshold Optimization (0.5 days)

**Files to Modify:**
1. `src/rag/infrastructure/reranker.py`
   - `MIN_RELEVANCE_THRESHOLD` 상수 조정
   - 관련성 점수 로깅 추가

**Implementation:**

```python
# reranker.py
# SPEC-RAG-QUALITY-007: Increase threshold to filter irrelevant documents
# Previous: 0.15 (too low, allows irrelevant docs)
# Target: 0.25-0.30 (balances precision vs recall)
MIN_RELEVANCE_THRESHOLD = 0.25  # Adjusted from 0.15

def rerank(self, query: str, documents: List, top_k: int = 10) -> List:
    # ... existing rerank logic ...

    # SPEC-RAG-QUALITY-007: Log relevance score distribution
    if reranked_docs:
        scores = [doc.score for doc in reranked_docs]
        logger.info(
            f"Reranking stats: "
            f"avg_score={sum(scores)/len(scores):.3f}, "
            f"min_score={min(scores):.3f}, "
            f"max_score={max(scores):.3f}, "
            f"filtered={original_count - len(reranked_docs)}"
        )

    return reranked_docs
```

**Validation:**
- [ ] 평균 관련성 점수 측정 (0.75 목표)
- [ ] 필터링된 문서 수 모니터링
- [ ] Context Relevance 점수 변화 확인

### Phase 3: Evaluation Verification (1 day)

**Tasks:**
1. **RAGAS 환경 검증**
   ```bash
   # Check chromadb installation
   python -c "import chromadb; print(chromadb.__version__)"

   # Verify RAGAS metrics
   python -c "from ragas.metrics import context_precision; print('OK')"
   ```

2. **Citation Format 검증**
   - 생성된 인용 형식: "「규정명」 제X조"
   - RAGAS 기대 형식 확인

3. **소규모 테스트**
   - 10개 대표 쿼리로 평가 실행
   - 메트릭별 실제 측정값 확인
   - 0.50이 기본값인지 실제값인지 판별

**Files to Create:**
- `scripts/verify_evaluation_metrics.py` - 평가 메트릭 검증 스크립트

### Phase 4: Integration Testing (1 day)

**Test Plan:**
1. **Baseline 측정**
   - 현재 상태로 50개 쿼리 평가
   - 메트릭별 기준값 확보

2. **Phase 1-2 적용 후 측정**
   - IntentClassifier 통합
   - Reranker 임계값 조정
   - 50개 쿼리 재평가

3. **전체 평가 (150개 쿼리)**
   - 모든 개선 사항 적용
   - 목표 달성 여부 확인:
     - Citations: 0.70+
     - Context Relevance: 0.75+
     - Pass Rate: 80%+ (120/150 queries)

**Success Criteria:**
- [ ] IntentClassifier 정확도 >= 90%
- [ ] Context Relevance >= 0.75
- [ ] Citations >= 0.70
- [ ] Pass Rate >= 80%
- [ ] 모든 페르소나에서 균형 잡힌 성능 (편차 < 5%)

---

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/rag/application/search_usecase.py` | IntentClassifier 통합, 의도 기반 검색 로직 | P0 |
| `src/rag/infrastructure/reranker.py` | MIN_RELEVANCE_THRESHOLD 조정, 로깅 추가 | P0 |
| `scripts/evaluate_rag_quality.py` | 평가 메트릭 검증 로직 추가 | P1 |

## Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `scripts/verify_evaluation_metrics.py` | RAGAS 메트릭 설정 검증 | P1 |
| `tests/integration/test_intent_aware_search.py` | IntentClassifier 통합 테스트 | P1 |
| `tests/integration/test_reranker_threshold.py` | Reranker 임계값 테스트 | P2 |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| IntentClassifier 오버피팅 | Low | Medium | 교차 검증, 다양한 테스트 케이스 |
| Reranker 임계값 조정 후 정밀도 저하 | Medium | High | A/B 테스트, 점진적 조정 |
| RAGAS 메트릭 설정 문제 | Medium | High | 소규모 테스트로 사전 검증 |
| Latency 증가 | Low | Low | 캐싱, 배치 처리 |

---

## Success Metrics

| Metric | Before | Target | Measurement Method |
|--------|--------|--------|-------------------|
| Citations | 0.50 | 0.70+ | RAGAS LLM-as-Judge |
| Context Relevance | 0.50 | 0.75+ | RAGAS Context Precision |
| Overall Score | 0.697 | 0.80+ | Weighted Average |
| Pass Rate | 0% | 80%+ | 150-query evaluation |
| IntentClassifier Accuracy | N/A | 90%+ | Unit test |

---

## Next Steps

1. **즉시 실행:** IntentClassifier 통합 (Phase 1)
2. **병렬 실행:** Reranker 임계값 조정 (Phase 2)
3. **검증:** 평가 메트릭 확인 (Phase 3)
4. **최종:** 통합 테스트 및 검증 (Phase 4)

**Handover to manager-ddd:**
- TAG Chain: TAG-001 (IntentClassifier) → TAG-002 (Reranker) → TAG-003 (Validation)
- Library Versions: 변경 없음 (기존 컴포넌트 활용)
- Key Decisions:
  1. IntentClassifier를 검색 파이프라인에 통합 (신규)
  2. MIN_RELEVANCE_THRESHOLD 0.15 → 0.25 조정
  3. CitationValidator는 이미 작동 중, 평가 메트릭 검증 필요

---

## References

- SPEC Document: `.moai/specs/SPEC-RAG-QUALITY-007/spec.md`
- Previous SPEC: `.moai/specs/SPEC-RAG-QUALITY-006/spec.md`
- Evaluation Data: `data/evaluations/evaluation_20260219_210750.json`
- Related Code:
  - `src/rag/application/intent_classifier.py` (IntentClassifier)
  - `src/rag/domain/citation/citation_validator.py` (CitationValidator)
  - `src/rag/infrastructure/reranker.py` (Reranker)
  - `src/rag/application/search_usecase.py` (Main Pipeline)
