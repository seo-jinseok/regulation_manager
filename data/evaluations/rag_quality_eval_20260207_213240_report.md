# RAG Quality Evaluation Report
**Evaluation ID:** rag_quality_20260207_213240
**Generated:** 2026-02-07 21:32:40

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Queries** | 30 |
| **Passed** | 0 |
| **Failed** | 30 |
| **Pass Rate** | 0.0% |

## Average Scores

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Overall** | 0.274 | 0.800 | ✗ |
| **Accuracy** | 0.500 | 0.850 | ✗ |
| **Completeness** | 0.337 | 0.750 | ✗ |
| **Citations** | 0.077 | 0.700 | ✗ |
| **Context Relevance** | 0.000 | 0.750 | ✗ |

## Results by Persona

### student-undergraduate

- **Queries Tested:** 5
- **Average Score:** 0.311
- **Pass Rate:** 0.0%

**Issues:**

- 일부 부정확한 정보: 5x
- 검색된 문서 관련성 낮음: 5x
- 핵심 정보 누락: 4x
- 규정 인용 부족 또는 부정확: 4x

### professor

- **Queries Tested:** 5
- **Average Score:** 0.311
- **Pass Rate:** 0.0%

**Issues:**

- 일부 부정확한 정보: 5x
- 검색된 문서 관련성 낮음: 5x
- 핵심 정보 누락: 4x
- 규정 인용 부족 또는 부정확: 4x

### student-international

- **Queries Tested:** 5
- **Average Score:** 0.274
- **Pass Rate:** 0.0%

**Issues:**

- 일부 부정확한 정보: 5x
- 핵심 정보 누락: 5x
- 규정 인용 부족 또는 부정확: 5x
- 검색된 문서 관련성 낮음: 5x

### student-graduate

- **Queries Tested:** 5
- **Average Score:** 0.250
- **Pass Rate:** 0.0%

**Issues:**

- 일부 부정확한 정보: 5x
- 핵심 정보 누락: 5x
- 규정 인용 부족 또는 부정확: 5x
- 검색된 문서 관련성 낮음: 5x

### staff-admin

- **Queries Tested:** 5
- **Average Score:** 0.250
- **Pass Rate:** 0.0%

**Issues:**

- 일부 부정확한 정보: 5x
- 핵심 정보 누락: 5x
- 규정 인용 부족 또는 부정확: 5x
- 검색된 문서 관련성 낮음: 5x

### parent

- **Queries Tested:** 5
- **Average Score:** 0.250
- **Pass Rate:** 0.0%

**Issues:**

- 일부 부정확한 정보: 5x
- 핵심 정보 누락: 5x
- 규정 인용 부족 또는 부정확: 5x
- 검색된 문서 관련성 낮음: 5x

## Top Failure Patterns

- 일부 부정확한 정보: 30x
- 검색된 문서 관련성 낮음: 30x
- 핵심 정보 누락: 28x
- 규정 인용 부족 또는 부정확: 28x

## Detailed Query Results

### 1. 휴학 방법 알려줘
**Score:** 0.557 (FAIL)
**Metrics:** Acc=0.500, Comp=0.850, Cit=0.850, Ctx=0.000
**Issues:** 일부 부정확한 정보, 검색된 문서 관련성 낮음
**Strengths:** 적절한 규정 인용, 상세하고 구조화된 답변
**Answer:** **유학 휴학 방법은 다음과 같습니다.**

### 1. 휴학 신청 방법
- 휴学 applying는 교내시스템을 통해 신청할 수 있습니다.

### 2. 휴학 신청 절차
1.  휴학원서를 작성하여 제출해야 합니다.
2.  관련 증빙서류를 준비하여첨부해야 합니다.
3.  창업 education 학사제도 운영위원회의 사전 심의를 받아야 합니다.

### 3. ...

### 2. 등록금 언제까지 납부해요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 3. 성적 조회 어떻게 해요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 4. 장학금 신청하는 법
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 5. 수강신청 기간이 언제인가요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 6. 연구년 신청 관련 규정 확인 부탁드립니다
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 7. 연구비 지원 요건이 어떻게 되나요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 8. 논문 심사 절차 상세히 알려주세요
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 9. 조교 신청 자격 설명해주세요
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 10. 등록금 납부 유예 관련 문의
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 11. 휴직 관련 조항 확인 필요
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 12. 연구년 적용 기준 상세히
**Score:** 0.557 (FAIL)
**Metrics:** Acc=0.500, Comp=0.850, Cit=0.850, Ctx=0.000
**Issues:** 일부 부정확한 정보, 검색된 문서 관련성 낮음
**Strengths:** 적절한 규정 인용, 상세하고 구조화된 답변
**Answer:** *   연구년 적용 기준은 교내시스템을 통해 진행합니다. 

*   3-1-24 제9조 선정기준 및 절차에 따르면, 소속 대학장은 소속 교원에게 연구년제 허가 신청이 있을 경우, 당해 교원의 연구년제 계획이 본인이 학교에 유익한 공헌을 할 수 있는 가를 확인한 후 총장에게 추천하고, 총장은 추천사항을 심사하여 연구년제 연구자를 선정합니다. 

*   이와 ...

### 13. 교원 승진 관련 편/장/조 구체적 근거
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** 질문 대상을 선택해주세요: 교수, 학생, 직원...

### 14. 연구비 집행 규정 해석 부탁드립니다
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 15. Sabbatical 관련 예외 사항 확인
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 16. 휴가 업무 처리 절차 확인
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 17. 급여 관련 서식 양식 알려주세요
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 18. 연수 승인 권한자가 누구인가요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 19. 사무용품 사용 규정 안내
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 20. 복무 처리 기한이 언제까지인가요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 21. 자녀 등록금 관련해서 알고 싶어요
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 22. 장학금 부모님도 알아야 하나요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 23. 기숙사 비용이 어떻게 되나요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 24. 휴학 신청은 부모가 해야 하나요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 25. 졸업 관련 서류 뭐 필요한가요?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 26. How do I apply for leave of absence?
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 27. Visa related tuition payment procedure
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 28. Dormitory requirements for international students
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 29. Tell me about scholarship in English if possible
**Score:** 0.250 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.000, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** ...

### 30. Course registration English version available?
**Score:** 0.370 (FAIL)
**Metrics:** Acc=0.500, Comp=0.300, Cit=0.600, Ctx=0.000
**Issues:** 일부 부정확한 정보, 핵심 정보 누락, 규정 인용 부족 또는 부정확, 검색된 문서 관련성 낮음
**Answer:** 이 질문은 규정 검색이 필요하지 않습니다. 구체적인 규정 관련 질문을 해주세요....
