# RAG Quality Evaluation Report

**Evaluation ID:** eval_20260218_104503
**Timestamp:** 2026-02-18T10:45:03.769696
**Duration:** 41 seconds

## Executive Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Overall Pass Rate** | 0.0% | 80% | FAIL |
| **Total Queries** | 150 | 150+ | PASS |
| **Queries Passed** | 0 | - | - |

## Metric Scores

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Accuracy (Faithfulness) | 0.918 | 0.85 | PASS |
| Completeness (Recall) | 0.870 | 0.75 | PASS |
| Citations (Precision) | 0.500 | 0.7 | FAIL |
| Context Relevance | 0.500 | 0.75 | FAIL |
| **Overall Score** | **0.697** | **0.75** | **FAIL** |

## Per-Persona Results

| Persona | Total | Passed | Pass Rate | Avg Score |
|---------|-------|--------|-----------|-----------|
| student-undergraduate | 25 | 0 | 0.0% | 0.698 |
| student-graduate | 25 | 0 | 0.0% | 0.692 |
| professor | 25 | 0 | 0.0% | 0.700 |
| staff-admin | 25 | 0 | 0.0% | 0.700 |
| parent | 25 | 0 | 0.0% | 0.698 |
| student-international | 25 | 0 | 0.0% | 0.694 |

## Per-Category Results

| Category | Total | Passed | Pass Rate | Avg Score |
|----------|-------|--------|-----------|-----------|
| academic | 29 | 0 | 0.0% | 0.698 |
| admin | 15 | 0 | 0.0% | 0.697 |
| campus_life | 7 | 0 | 0.0% | 0.691 |
| financial | 27 | 0 | 0.0% | 0.694 |
| leave | 3 | 0 | 0.0% | 0.703 |
| personnel | 15 | 0 | 0.0% | 0.701 |
| procedural | 7 | 0 | 0.0% | 0.700 |
| research | 24 | 0 | 0.0% | 0.695 |
| salary | 7 | 0 | 0.0% | 0.702 |
| support | 1 | 0 | 0.0% | 0.693 |
| teaching | 2 | 0 | 0.0% | 0.702 |
| training | 2 | 0 | 0.0% | 0.703 |
| visa | 11 | 0 | 0.0% | 0.699 |

## Per-Difficulty Results

| Difficulty | Total | Passed | Pass Rate | Avg Score |
|------------|-------|--------|-----------|-----------|
| easy | 60 | 0 | 0.0% | 0.697 |
| hard | 30 | 0 | 0.0% | 0.697 |
| medium | 60 | 0 | 0.0% | 0.698 |

## Sample Failures (First 10)

### Failure 1: student-undergraduate - procedural
- **Query:** 휴학 어떻게 해?
- **Difficulty:** easy
- **Score:** 0.705
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 2: student-undergraduate - academic
- **Query:** 성적 평균 어떻게 계산돼?
- **Difficulty:** easy
- **Score:** 0.701
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 3: student-undergraduate - academic
- **Query:** 졸업 요건 뭐야?
- **Difficulty:** easy
- **Score:** 0.705
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 4: student-undergraduate - financial
- **Query:** 장학금 종류 알려줘
- **Difficulty:** easy
- **Score:** 0.700
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 5: student-undergraduate - campus_life
- **Query:** 기숙사 신청 방법 알려줘
- **Difficulty:** easy
- **Score:** 0.673
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 6: student-undergraduate - procedural
- **Query:** 복학하려면 뭐 해야돼?
- **Difficulty:** easy
- **Score:** 0.696
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 7: student-undergraduate - academic
- **Query:** 수강 신청 언제부터야?
- **Difficulty:** easy
- **Score:** 0.704
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 8: student-undergraduate - campus_life
- **Query:** 도서관 이용 시간이 어떻게 돼?
- **Difficulty:** easy
- **Score:** 0.698
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 9: student-undergraduate - admin
- **Query:** 학생증 재발급 어떻게 해?
- **Difficulty:** easy
- **Score:** 0.693
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65

### Failure 10: student-undergraduate - admin
- **Query:** 성적증명서 발급받고 싶어
- **Difficulty:** easy
- **Score:** 0.693
- **Reasons:** Answer Relevancy below threshold: 0.500 < 0.7; Contextual Precision below threshold: 0.500 < 0.65
