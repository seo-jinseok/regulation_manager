# LLM-as-Judge RAG Quality Evaluation Summary

## Overview

**Date:** 2026-02-07
**Stage:** 1 - Initial (Week 1)
**Evaluation Method:** Custom LLM-as-Judge (OpenAI GPT-4o)
**Total Scenarios:** 30 (6 personas × 5 scenarios each)

---

## Overall Results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Total Scenarios** | 30 | - | - |
| **Passed** | 3 | - | - |
| **Failed** | 27 | - | - |
| **Pass Rate** | **10.0%** | 60.0% | ❌ FAIL |
| **Faithfulness** | 0.517 | 0.600 | ❌ FAIL |
| **Answer Relevancy** | 0.733 | 0.700 | ✅ PASS |
| **Contextual Precision** | 0.557 | 0.650 | ❌ FAIL |
| **Contextual Recall** | 0.320 | 0.650 | ❌ FAIL |
| **Overall Score** | **0.540** | 0.600 | ❌ FAIL |

---

## Persona Performance Breakdown

| Persona | Total | Passed | Failed | Pass Rate | Avg Score |
|---------|-------|--------|--------|-----------|-----------|
| **Freshman** | 5 | 2 | 3 | **40.0%** | 0.699 |
| **Graduate** | 5 | 0 | 5 | 0.0% | 0.556 |
| **Professor** | 5 | 0 | 5 | 0.0% | 0.515 |
| **Staff** | 5 | 0 | 5 | 0.0% | 0.508 |
| **Parent** | 5 | 1 | 4 | **20.0%** | 0.333 |
| **International** | 5 | 0 | 5 | 0.0% | 0.626 |

**Best Performing Persona:** Freshman (40% pass rate)
**Worst Performing Persona:** Graduate/Professor/Staff/International (0% pass rate)

---

## Category Performance Breakdown

| Category | Total | Passed | Failed | Pass Rate | Avg Score |
|----------|-------|--------|--------|-----------|-----------|
| **Simple** | 15 | 1 | 14 | 6.7% | 0.577 |
| **Complex** | 10 | 1 | 9 | 10.0% | 0.550 |
| **Edge** | 5 | 1 | 4 | 20.0% | 0.405 |

**Best Category:** Edge cases (20% pass rate)
**Worst Category:** Simple queries (6.7% pass rate)

---

## Passing Scenarios

### 1. freshman_004 - "처음이라 수강 신청 절차를 잘 몰라요"
- **Overall Score:** 0.775
- **Scores:**
  - Faithfulness: 0.70 ✅
  - Answer Relevancy: 1.00 ✅
  - Contextual Precision: 0.70 ✅
  - Contextual Recall: 0.70 ✅

### 2. freshman_005 - "복학 신청도 따로 해야 하나요?"
- **Overall Score:** 0.815
- **Scores:**
  - Faithfulness: 0.70 ✅
  - Answer Relevancy: 1.00 ✅
  - Contextual Precision: 0.90 ✅
  - Contextual Recall: 0.70 ✅

### 3. parent_004 - "장학금 받는 조건이 뭐예요?"
- **Overall Score:** 0.925 ⭐ (Best)
- **Scores:**
  - Faithfulness: 0.90 ✅
  - Answer Relevancy: 1.00 ✅
  - Contextual Precision: 1.00 ✅
  - Contextual Recall: 0.80 ✅

---

## Critical Issues Identified

### 1. Contextual Recall (Biggest Issue) - 32% vs 65% threshold
**Impact:** High
**Description:** The RAG system consistently fails to retrieve all relevant information needed to answer queries completely.
**Examples:**
- "장학금 신청 방법" - Retrieved contexts about registration, not scholarships
- "연구년 자격 요건" - Missing key information about research sabbatical criteria
- "교원인사규정 제8조" - Retrieved general info, missing specific article content

### 2. Hallucination Issues (Faithfulness) - 51.7% vs 60% threshold
**Impact:** Critical
**Description:** Several responses contain significant unsupported information or complete hallucinations.
**Critical Failures (0.0 Faithfulness):**
- freshman_002: Scholarship application (retrieved registration info, hallucinated scholarship process)
- graduate_004: TA work hours and benefits (no relevant contexts found)
- professor_003: Promotion criteria (no relevant contexts found)
- professor_005: Sabbatical vs research year differences (no relevant contexts found)
- staff_003: Salary payment dates (no relevant contexts found)
- international_005: Housing options (no relevant contexts found)

### 3. Contextual Precision - 55.7% vs 65% threshold
**Impact:** Medium
**Description:** Retrieved contexts often contain irrelevant information or poor ranking.
**Issue:** Many queries retrieve contexts about registration/enrollment when asking about scholarships or other topics.

---

## Recommendations for Improvement

### Immediate Actions (Priority: High)

1. **Improve Retrieval Quality**
   - Enhance query expansion to better understand user intent
   - Improve entity recognition for regulation-specific terms (제8조, 연구년, etc.)
   - Add query-type detection (procedure vs. factual vs. conditional)

2. **Fix Hallucination Issues**
   - Implement stricter citation verification
   - Add "unknown" response when contexts are insufficient
   - Improve prompt engineering to reduce unsupported claims

3. **Increase Contextual Recall**
   - Expand top_k from 5 to 10 for complex queries
   - Implement multi-hop retrieval for cross-regulation queries
   - Add semantic search improvements for Korean academic terms

### Medium-Term Improvements (Priority: Medium)

1. **Persona-Specific Optimization**
   - Freshman queries are working relatively well (40% pass rate)
   - International student queries need English language support
   - Professor queries require precise article/section retrieval

2. **Hyde (Hypothetical Document Embeddings)**
   - Implement query expansion using hypothetical answers
   - Improve retrieval for complex, multi-part questions

3. **Reranker Fine-tuning**
   - Current reranker may not be optimal for Korean academic regulations
   - Consider fine-tuning on domain-specific data

### Long-Term Improvements (Priority: Low)

1. **Knowledge Graph Integration**
   - Build relationships between regulations, articles, and topics
   - Enable cross-references and hierarchical navigation

2. **Multi-Stage Retrieval**
   - First stage: Broad topic identification
   - Second stage: Specific article/section retrieval
   - Third stage: Detail extraction and synthesis

---

## Stage 1 Threshold Progress

The evaluation was conducted using **Stage 1 (Initial)** thresholds:

| Metric | Stage 1 Threshold | Target (Stage 3) | Current | Gap |
|--------|------------------|------------------|---------|-----|
| Faithfulness | 0.60 | 0.80 | 0.517 | -0.083 |
| Answer Relevancy | 0.70 | 0.80 | 0.733 | +0.033 ✅ |
| Contextual Precision | 0.65 | 0.75 | 0.557 | -0.093 |
| Contextual Recall | 0.65 | 0.75 | 0.320 | -0.330 ❌ |
| Overall | 0.60 | 0.75 | 0.540 | -0.060 |

**Status:** 1/5 metrics passing Stage 1 thresholds
**Progress to Stage 2:** Not ready
**Confidence:** Low - Significant improvements needed

---

## Test Execution Details

- **Test Duration:** 441 seconds (7.4 minutes)
- **Average Time per Scenario:** 14.7 seconds
- **Evaluation Model:** GPT-4o (OpenAI)
- **Evaluation Method:** Custom LLM-as-Judge with structured JSON output
- **RAG Configuration:**
  - Vector Store: ChromaDB with 17,254 documents
  - Embeddings: paraphrase-multilingual-MiniLM-L12-v2
  - Reranker: BGE-Reranker-v2-m3 (enabled)
  - Top-K: 5 contexts per query

---

## Conclusion

The RAG system is **not yet ready for production use** at Stage 1 thresholds. While Answer Relevancy is passing (73.3%), the other three critical metrics are below threshold:

1. **Contextual Recall** is the most significant issue (32% vs 65% threshold)
2. **Hallucination problems** exist in multiple scenarios
3. **Contextual Precision** needs improvement for better retrieval ranking

**Priority Focus Areas:**
1. Fix hallucination by ensuring responses are grounded in retrieved contexts
2. Improve retrieval to find all relevant information (increase recall)
3. Enhance query understanding to retrieve the right regulation sections

**Next Steps:**
1. Implement immediate actions to improve retrieval quality
2. Add stricter citation verification
3. Re-evaluate after improvements
4. Target Stage 2 thresholds (medium-term improvements)
