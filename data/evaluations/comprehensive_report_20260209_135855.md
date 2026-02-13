# RAG Quality Evaluation Report
Generated: 2026-02-09 13:58:55

## Executive Summary

**Total Queries Evaluated:** 30
**Passed:** 23
**Failed:** 7
**Overall Pass Rate:** 76.7%

### Average Scores
- **Overall Score:** 0.781
- **Accuracy:** 0.812
- **Completeness:** 0.736
- **Citations:** 0.743
- **Context Relevance:** 0.833

## Per-Persona Breakdown

### Parent
- **Queries Tested:** 5
- **Average Score:** 0.754
- **Pass Rate:** 60.0%

**Common Issues:**
- 정보 불충분: 5 occurrences
- 문서 관련성 낮음: 2 occurrences
- 일부 정보 부정확: 2 occurrences
- 규정 인용 부족: 1 occurrences

### Professor
- **Queries Tested:** 5
- **Average Score:** 0.805
- **Pass Rate:** 60.0%

**Common Issues:**
- 일부 정보 부정확: 3 occurrences
- 문서 관련성 낮음: 3 occurrences
- 규정 인용 부족: 1 occurrences

### Administrative Staff
- **Queries Tested:** 5
- **Average Score:** 0.785
- **Pass Rate:** 100.0%

**Common Issues:**
- 일부 정보 부정확: 4 occurrences
- 정보 불충분: 4 occurrences
- 문서 관련성 낮음: 2 occurrences
- 규정 인용 부족: 2 occurrences

### Graduate Student
- **Queries Tested:** 5
- **Average Score:** 0.772
- **Pass Rate:** 80.0%

**Common Issues:**
- 정보 불충분: 5 occurrences
- 일부 정보 부정확: 2 occurrences
- 규정 인용 부족: 1 occurrences
- 문서 관련성 낮음: 1 occurrences

### International Student
- **Queries Tested:** 5
- **Average Score:** 0.767
- **Pass Rate:** 60.0%

**Common Issues:**
- 규정 인용 부족: 5 occurrences
- 정보 불충분: 4 occurrences
- 문서 관련성 낮음: 1 occurrences
- 일부 정보 부정확: 1 occurrences

### Undergraduate Student
- **Queries Tested:** 5
- **Average Score:** 0.804
- **Pass Rate:** 100.0%

**Common Issues:**
- 일부 정보 부정확: 2 occurrences
- 문서 관련성 낮음: 2 occurrences
- 규정 인용 부족: 1 occurrences

## Failure Pattern Analysis

### Top Issues Across All Personas
- **일부 정보 부정확**: 6 occurrences
- **문서 관련성 낮음**: 6 occurrences
- **규정 인용 부족**: 6 occurrences
- **정보 불충분**: 4 occurrences

## Detailed Query Results

### Parent

#### 1. 자녀 등록금 관련해서 알고 싶어요 ❌ FAIL
**Score:** 0.708 | Acc: 0.82 | Comp: 0.70 | Cit: 0.60 | Ctx: 0.72

**Strengths:** 정확한 정보 제공
**Issues:** 정보 불충분, 규정 인용 부족, 문서 관련성 낮음

#### 2. 장학금 부모님도 알아야 하나요? ✅ PASS
**Score:** 0.762 | Acc: 0.71 | Comp: 0.72 | Cit: 0.82 | Ctx: 0.79

**Strengths:** 적절한 규정 인용
**Issues:** 일부 정보 부정확, 정보 불충분, 문서 관련성 낮음

#### 3. 기숙사 신청 방법 알려주세요 ✅ PASS
**Score:** 0.753 | Acc: 0.81 | Comp: 0.68 | Cit: 0.72 | Ctx: 0.80

**Strengths:** 정확한 정보 제공, 적절한 규정 인용
**Issues:** 정보 불충분

#### 4. 성적 확인은 부모가 할 수 있나요? ❌ FAIL
**Score:** 0.740 | Acc: 0.69 | Comp: 0.63 | Cit: 0.78 | Ctx: 0.86

**Strengths:** 적절한 규정 인용
**Issues:** 일부 정보 부정확, 정보 불충분

#### 5. 휴학 비용이 어떻게 되나요? ✅ PASS
**Score:** 0.806 | Acc: 0.85 | Comp: 0.65 | Cit: 0.84 | Ctx: 0.89

**Strengths:** 정확한 정보 제공, 적절한 규정 인용
**Issues:** 정보 불충분

### Professor

#### 1. 연구년 관련 조항 확인 필요 ✅ PASS
**Score:** 0.874 | Acc: 0.93 | Comp: 0.83 | Cit: 0.82 | Ctx: 0.92

**Strengths:** 정확한 정보 제공, 포괄적인 답변, 적절한 규정 인용

#### 2. 승진 심사 기준 상세히 ✅ PASS
**Score:** 0.788 | Acc: 0.80 | Comp: 0.81 | Cit: 0.84 | Ctx: 0.70

**Strengths:** 포괄적인 답변, 적절한 규정 인용
**Issues:** 일부 정보 부정확, 문서 관련성 낮음

#### 3. 연구비 집행 관련 규정 해석 부탁드립니다 ❌ FAIL
**Score:** 0.739 | Acc: 0.75 | Comp: 0.86 | Cit: 0.57 | Ctx: 0.78

**Strengths:** 포괄적인 답변
**Issues:** 일부 정보 부정확, 규정 인용 부족, 문서 관련성 낮음

#### 4. 교원 인사 규정 예외 사항 확인 필요 ✅ PASS
**Score:** 0.886 | Acc: 0.82 | Comp: 0.90 | Cit: 0.88 | Ctx: 0.95

**Strengths:** 정확한 정보 제공, 포괄적인 답변, 적절한 규정 인용

#### 5. Sabbatical leave 관련 규정 안내 ❌ FAIL
**Score:** 0.738 | Acc: 0.72 | Comp: 0.76 | Cit: 0.74 | Ctx: 0.73

**Strengths:** 포괄적인 답변, 적절한 규정 인용
**Issues:** 일부 정보 부정확, 문서 관련성 낮음

### Administrative Staff

#### 1. 휴가 신청 업무 처리 절차 확인 ✅ PASS
**Score:** 0.756 | Acc: 0.79 | Comp: 0.69 | Cit: 0.82 | Ctx: 0.72

**Strengths:** 적절한 규정 인용
**Issues:** 일부 정보 부정확, 정보 불충분, 문서 관련성 낮음

#### 2. 급여 지급일이 언제인가요? ✅ PASS
**Score:** 0.843 | Acc: 0.93 | Comp: 0.75 | Cit: 0.83 | Ctx: 0.86

**Strengths:** 정확한 정보 제공, 적절한 규정 인용
**Issues:** 정보 불충분

#### 3. 연수 참여 절차 안내 ✅ PASS
**Score:** 0.782 | Acc: 0.79 | Comp: 0.68 | Cit: 0.89 | Ctx: 0.77

**Strengths:** 적절한 규정 인용
**Issues:** 일부 정보 부정확, 정보 불충분, 문서 관련성 낮음

#### 4. 사무용품 신청 서식 양식 알려주세요 ✅ PASS
**Score:** 0.776 | Acc: 0.73 | Comp: 0.80 | Cit: 0.63 | Ctx: 0.94

**Strengths:** 포괄적인 답변
**Issues:** 일부 정보 부정확, 규정 인용 부족

#### 5. 시설 사용 승인 권한자가 누구인가요? ✅ PASS
**Score:** 0.767 | Acc: 0.79 | Comp: 0.67 | Cit: 0.67 | Ctx: 0.94

**Issues:** 일부 정보 부정확, 정보 불충분, 규정 인용 부족

### Graduate Student

#### 1. 연구년 신청 자격 요건이 어떻게 되나요? ✅ PASS
**Score:** 0.788 | Acc: 0.95 | Comp: 0.62 | Cit: 0.67 | Ctx: 0.91

**Strengths:** 정확한 정보 제공
**Issues:** 정보 불충분, 규정 인용 부족

#### 2. 연구비 지원 관련 규정 확인 부탁드립니다 ❌ FAIL
**Score:** 0.749 | Acc: 0.66 | Comp: 0.64 | Cit: 0.83 | Ctx: 0.87

**Strengths:** 적절한 규정 인용
**Issues:** 일부 정보 부정확, 정보 불충분

#### 3. 논문 심사 절차 상세히 알려주세요 ✅ PASS
**Score:** 0.795 | Acc: 0.73 | Comp: 0.71 | Cit: 0.81 | Ctx: 0.93

**Strengths:** 적절한 규정 인용
**Issues:** 일부 정보 부정확, 정보 불충분

#### 4. 조교 신청 방법이 있나요? ✅ PASS
**Score:** 0.750 | Acc: 0.86 | Comp: 0.60 | Cit: 0.77 | Ctx: 0.77

**Strengths:** 정확한 정보 제공, 적절한 규정 인용
**Issues:** 정보 불충분, 문서 관련성 낮음

#### 5. 대학원 등록금 납부 기한이 언제까지인가요? ✅ PASS
**Score:** 0.776 | Acc: 0.92 | Comp: 0.62 | Cit: 0.73 | Ctx: 0.83

**Strengths:** 정확한 정보 제공, 적절한 규정 인용
**Issues:** 정보 불충분

### International Student

#### 1. How do I apply for leave of absence? ✅ PASS
**Score:** 0.775 | Acc: 0.83 | Comp: 0.72 | Cit: 0.63 | Ctx: 0.92

**Strengths:** 정확한 정보 제공
**Issues:** 정보 불충분, 규정 인용 부족

#### 2. Tuition payment procedure for international students ❌ FAIL
**Score:** 0.726 | Acc: 0.92 | Comp: 0.60 | Cit: 0.59 | Ctx: 0.79

**Strengths:** 정확한 정보 제공
**Issues:** 정보 불충분, 규정 인용 부족, 문서 관련성 낮음

#### 3. Scholarship requirements ❌ FAIL
**Score:** 0.748 | Acc: 0.79 | Comp: 0.72 | Cit: 0.60 | Ctx: 0.89

**Issues:** 일부 정보 부정확, 정보 불충분, 규정 인용 부족

#### 4. Dormitory application related to visa status ✅ PASS
**Score:** 0.776 | Acc: 0.90 | Comp: 0.69 | Cit: 0.62 | Ctx: 0.89

**Strengths:** 정확한 정보 제공
**Issues:** 정보 불충분, 규정 인용 부족

#### 5. English version of course registration available? ✅ PASS
**Score:** 0.812 | Acc: 0.88 | Comp: 0.88 | Cit: 0.67 | Ctx: 0.82

**Strengths:** 정확한 정보 제공, 포괄적인 답변
**Issues:** 규정 인용 부족

### Undergraduate Student

#### 1. 휴학 방법 알려줘 ✅ PASS
**Score:** 0.803 | Acc: 0.77 | Comp: 0.85 | Cit: 0.89 | Ctx: 0.70

**Strengths:** 포괄적인 답변, 적절한 규정 인용
**Issues:** 일부 정보 부정확, 문서 관련성 낮음

#### 2. 성적 조회 어떻게 해요? ✅ PASS
**Score:** 0.753 | Acc: 0.87 | Comp: 0.75 | Cit: 0.58 | Ctx: 0.81

**Strengths:** 정확한 정보 제공, 포괄적인 답변
**Issues:** 규정 인용 부족

#### 3. 장학금 신청 절차가 궁금해요 ✅ PASS
**Score:** 0.750 | Acc: 0.70 | Comp: 0.84 | Cit: 0.77 | Ctx: 0.70

**Strengths:** 포괄적인 답변, 적절한 규정 인용
**Issues:** 일부 정보 부정확, 문서 관련성 낮음

#### 4. 수강신청 기간이 언제인가요? ✅ PASS
**Score:** 0.851 | Acc: 0.85 | Comp: 0.88 | Cit: 0.80 | Ctx: 0.87

**Strengths:** 정확한 정보 제공, 포괄적인 답변, 적절한 규정 인용

#### 5. 등록금 납부 방법 알려주세요 ✅ PASS
**Score:** 0.860 | Acc: 0.82 | Comp: 0.83 | Cit: 0.88 | Ctx: 0.91

**Strengths:** 정확한 정보 제공, 포괄적인 답변, 적절한 규정 인용

## Recommendations

- **Improve Citations:** Enhance regulation reference accuracy and completeness
- **Enhance Completeness:** Include more comprehensive information in responses
- **Boost Accuracy:** Focus on factual correctness and reduce hallucinations
