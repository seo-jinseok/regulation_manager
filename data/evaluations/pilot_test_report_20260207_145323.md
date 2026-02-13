# RAG Quality Pilot Test Report

**Generated:** 2026-02-07T14:53:23.557264
**Test Duration:** 643.27 seconds

## Summary Statistics

- **Total Scenarios:** 30
- **Passed:** 0
- **Failed:** 30
- **Pass Rate:** 0.0%

## Average Scores

| Metric | Score | Threshold |
|--------|-------|-----------|
| Faithfulness | 0.502 | 0.80 |
| Answer Relevancy | 0.557 | 0.80 |
| Contextual Precision | 0.500 | 0.80 |
| Contextual Recall | 0.500 | 0.80 |
| Overall | 0.515 | 0.80 |

## Per-Persona Breakdown

| Persona | Total | Passed | Failed | Pass Rate | Avg Score |
|---------|-------|--------|--------|-----------|-----------|
| freshman | 5 | 0 | 5 | 0.0% | 0.517 |
| graduate | 5 | 0 | 5 | 0.0% | 0.530 |
| professor | 5 | 0 | 5 | 0.0% | 0.528 |
| staff | 5 | 0 | 5 | 0.0% | 0.508 |
| parent | 5 | 0 | 5 | 0.0% | 0.505 |
| international | 5 | 0 | 5 | 0.0% | 0.500 |

## Per-Category Breakdown

| Category | Total | Passed | Failed | Pass Rate | Avg Score |
|----------|-------|--------|--------|-----------|-----------|
| Simple | 14 | 0 | 14 | 0.0% | 0.507 |
| Complex | 12 | 0 | 12 | 0.0% | 0.518 |
| Edge | 4 | 0 | 4 | 0.0% | 0.531 |

## Failure Analysis

### Persona: freshman
**Query:** 휴학 어떻게 해요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: freshman
**Query:** 장학금 신청 방법 알려주실까요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: freshman
**Query:** 성적이 나쁘면 휴학해야 하나요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.750 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: freshman
**Query:** 처음이라 수강 신청 절차를 잘 몰라요
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: freshman
**Query:** 복학 신청도 따로 해야 하나요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.600 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: graduate
**Query:** 연구년 자격 요건이 어떻게 되나요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: graduate
**Query:** 연구비 지급 규정 확인 부탁드립니다
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: graduate
**Query:** 논문 제출 기한 연장 가능한가요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.800 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: graduate
**Query:** 조교 근무 시간과 장학금 혜택 관련하여
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: graduate
**Query:** 등록금 면제 기준이 대학원마다 달라요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.800 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: professor
**Query:** 교원인사규정 제8조 확인 필요
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.750 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: professor
**Query:** 연구년 적용 기준 상세히
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.750 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: professor
**Query:** 승진 심의 기준과 편장조 구체적 근거
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: professor
**Query:** 휴직 시 급여 지급 규정 해석 부탁드립니다
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.571 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: professor
**Query:** Sabbatical leave 규정과 국내 연구년 차이점
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: staff
**Query:** 복무 규정 확인 부탁드립니다
**Failure Reasons:**
- Faithfulness below threshold: 0.561 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: staff
**Query:** 휴가 신청 서식 양식 알려주세요
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.600 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: staff
**Query:** 급여 지급일과 처리 기한이 언제까지인가요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: staff
**Query:** 사무용품 사용 규정과 승인 권한자 확인
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: staff
**Query:** 연수 참가 절차와 경비 처리 방법
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: parent
**Query:** 자녀 장학금 관련해서 알고 싶어요
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: parent
**Query:** 기숙사 비용이 어떻게 되나요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: parent
**Query:** 휴학 비용도 내야 하나요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: parent
**Query:** 성적 저하 시 학교에서 알려주나요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.600 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: parent
**Query:** 자녀 졸업 요건이 무엇인가요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: international
**Query:** How do I apply for student visa?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: international
**Query:** Tell me about dormitory procedure for international students
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: international
**Query:** Korean language program requirements
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: international
**Query:**  tuition fee payment in English version available?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

### Persona: international
**Query:** 비자 연장 절차와 관련 서류 뭐 필요한가요?
**Failure Reasons:**
- Faithfulness below threshold: 0.500 < 0.9
- Answer Relevancy below threshold: 0.500 < 0.85
- Contextual Precision below threshold: 0.500 < 0.8
- Contextual Recall below threshold: 0.500 < 0.8
- CRITICAL: Faithfulness below critical threshold - high hallucination risk

## Recommendations

- Poor quality. Major system improvements required.
