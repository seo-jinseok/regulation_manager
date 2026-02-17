# RAG Quality Evaluation Report

**Evaluation ID**: eval_stage1_2026-02-15
**Date**: 2026-02-15
**Source**: custom_llm_judge_eval_stage1_latest.json

---

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Total Scenarios** | 30 | - |
| **Passed** | 4 | ⚠️ |
| **Failed** | 26 | 🔴 |
| **Pass Rate** | 13.33% | 🔴 Critical |
| **Overall Score** | 0.526 | Below Target |

### Quick Assessment

현재 RAG 시스템은 **심각한 품질 문제**를 겪고 있습니다. 합격률 13.33%는 목표(80%+)에 크게 미달하며, 특히 **할루시네이션 위험**과 **문서 검색 실패**가 주요 문제입니다.

---

## Metric Analysis

### Score Distribution

| Metric | Average | Threshold | Gap | Trend |
|--------|---------|-----------|-----|-------|
| Faithfulness | 0.50 | 0.60 | -0.10 | 🔴 |
| Answer Relevancy | 0.71 | 0.70 | +0.01 | 🟢 |
| Contextual Precision | 0.54 | 0.65 | -0.11 | 🔴 |
| Contextual Recall | 0.32 | 0.65 | **-0.33** | 🔴 |

### Key Finding

**Contextual Recall이 가장 큰 문제**입니다. 시스템이 관련 문서를 찾지 못하고 있으며, 이로 인해 할루시네이션이 발생합니다.

---

## Persona Performance

### Overview

| Persona | Total | Passed | Failed | Pass Rate | Avg Score |
|---------|-------|--------|--------|-----------|-----------|
| Freshman | 5 | 2 | 3 | **40%** | 0.735 |
| International | 5 | 1 | 4 | 20% | 0.663 |
| Parent | 5 | 1 | 4 | 20% | 0.291 |
| Graduate | 5 | 0 | 5 | **0%** | 0.576 |
| Professor | 5 | 0 | 5 | **0%** | 0.479 |
| Staff | 5 | 0 | 5 | **0%** | 0.410 |

### Analysis

- **Best**: Freshman (40%) - 단순 질문에 상대적으로 잘 대응
- **Worst**: Graduate, Professor, Staff (0%) - 전문/행정 용어 질문 완전 실패
- **Critical**: Parent (20%, avg 0.291) - 평균 점수가 매우 낮음

---

## Category Performance

| Category | Total | Passed | Failed | Pass Rate | Avg Score |
|----------|-------|--------|--------|-----------|-----------|
| Simple | 15 | 2 | 13 | 13.33% | 0.603 |
| Complex | 10 | 2 | 8 | 20% | 0.487 |
| Edge | 5 | 0 | 5 | **0%** | 0.372 |

### Analysis

모든 카테고리에서 성능이 저조하며, 특히 **Edge 케이스**에서 완전 실패합니다.

---

## Failure Analysis

### Pattern 1: Hallucination Risk (14 cases)

**Faithfulness = 0.0**인 쿼리들 - 시스템이 근거 없는 답변 생성

| Query | Persona | Issue |
|-------|---------|-------|
| 장학금 신청 방법 알려주실까요? | Freshman | No relevant docs found |
| 논문 제출 기한 연장 가능한가요? | Graduate | Generated fake policy |
| 승진 심의 기준과 편장조 구체적 근거 | Professor | No citations provided |
| 사무용품 사용 규정과 승인 권한자 확인 | Staff | Fabricated procedures |
| 자녀 성적 확인 어떻게 하면 돼요? | Parent | Wrong information |

### Pattern 2: Low Contextual Recall (20 cases)

관련 문서를 찾지 못함

| Query | Recall | Expected |
|-------|--------|----------|
| 교원인사규정 제8조 확인 필요 | 0.0 | 0.65+ |
| 복무 규정 확인 부탁드립니다 | 0.0 | 0.65+ |
| 등록금 납부 기간과 방법 알려주세요 | 0.0 | 0.65+ |

### Pattern 3: Low Contextual Precision (8 cases)

관련 없는 문서까지 검색

---

## Passed Scenarios (4)

| ID | Persona | Query | Score |
|----|---------|-------|-------|
| freshman_003 | Freshman | 졸업 요건이 뭔가요? | Pass |
| freshman_004 | Freshman | 성적 확인은 어디서 하나요? | Pass |
| graduate_005 | Graduate | 등록금 면제 기준이 대학원마다 달라요? | 0.85 |
| parent_004 | Parent | 장학금 종류가 어떻게 되나요? | Pass |

---

## Recommendations

### Priority 0 (Immediate)

1. **Reranker 수정**
   - FlagEmbedding/transformers 호환성 문제 해결
   - 또는 대체 reranker 구현

2. **신뢰도 임계값 추가**
   - Contextual Recall < 0.3인 경우 "정보를 찾을 수 없습니다" 반환
   - 할루시네이션 방지

### Priority 1 (This Week)

3. **청킹 전략 개선**
   - 청크 크기 축소 (512 토큰)
   - 규정 구조 보존

4. **동의어 매핑**
   - 학술/행정 용어 동의어 사전 구축
   - 한국어 형태소 분석 추가

### Priority 2 (Next Sprint)

5. **페르소나 지원**
   - 질문 스타일 기반 페르소나 감지
   - 기술적 깊이 조절

---

## Next Steps

1. **SPEC-RAG-QUALITY-001** 실행하여 개선 작업 진행
2. FlagEmbedding 호환성 해결 후 재평가
3. 개선 후 30개 시나리오 재테스트
4. 목표: Pass Rate 80%+ 달성

---

## Files Generated

- SPEC Document: `.moai/specs/SPEC-RAG-QUALITY-001/spec.md`
- Evaluation Data: `data/evaluations/custom_llm_judge_eval_stage1_latest.json`
- This Report: `data/evaluations/evaluation_report_2026-02-15.md`

---

<moai>DONE</moai>
