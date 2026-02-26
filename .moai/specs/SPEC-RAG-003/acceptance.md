# SPEC-RAG-003 Acceptance Criteria

## 핵심 수치 목표

| Metric | Baseline (현재) | Target (목표) | Stretch |
|--------|---------------|--------------|---------|
| Pass Rate | 16.7% (5/30) | 60%+ (18/30) | 70%+ (21/30) |
| Average Score | 0.534 | 0.700+ | 0.750+ |
| International Persona | 0.122 | 0.500+ | 0.600+ |
| Professor Persona | 0.195 | 0.500+ | 0.600+ |
| No Regression | 5 passing queries | 5 still passing | 5 still passing |

---

## Scenario 1: 영어 쿼리 Self-RAG 통과

**Given** 영어로 작성된 대학 규정 관련 쿼리가 있을 때  
**When** Self-RAG 평가기가 검색 필요 여부를 판단하면  
**Then** [RETRIEVE_YES]를 반환하고 검색 파이프라인으로 진행해야 한다

### Test Cases:
- `"What are the tuition fees for international students?"` → RETRIEVE_YES
- `"How do I apply for a leave of absence?"` → RETRIEVE_YES
- `"What are the dormitory rules?"` → RETRIEVE_YES
- `"How to apply for scholarship?"` → RETRIEVE_YES
- `"What is the graduation requirement?"` → RETRIEVE_YES

### Acceptance: 5/5 영어 쿼리 모두 검색 파이프라인 도달

---

## Scenario 2: CoT 추론 과정 미노출

**Given** 임의의 사용자 쿼리가 있을 때  
**When** RAG 시스템이 답변을 생성하면  
**Then** 응답에 내부 추론 과정(Chain-of-Thought)이 포함되지 않아야 한다

### Negative Patterns (응답에 없어야 함):
- `"Analyze the User's Request"`
- `"User Persona:"`
- `"Constraint"`
- `"Step 1: Analyze"`
- `"Let me think"`
- 번호 매기기 분석 단계 (`1. **Analyze**`, `2. **Check**`)

### Acceptance: 30개 쿼리 중 CoT 패턴 0건 검출

---

## Scenario 3: 기존 통과 쿼리 회귀 방지

**Given** 현재 통과하는 5개 쿼리가 있을 때  
**When** SPEC-RAG-003 변경 사항이 적용된 후 동일 쿼리를 실행하면  
**Then** 5개 쿼리 모두 여전히 통과해야 한다

### Protected Queries:
1. Q2: 장학금 종류와 신청방법
2. Q19: 도서관 운영시간
3. Q21: 캡스톤디자인 수업
4. Q24: 기숙사 입사 조건
5. Q26: 학과 변경(전과)

### Acceptance: 이 5개 쿼리의 overall score ≥ 0.80 유지

---

## Scenario 4: 환각 방지

**Given** 검색 결과에 쿼리와 무관한 문서만 있을 때  
**When** 답변이 생성되면  
**Then** 검색되지 않은 내용을 지어내지 않아야 한다

### Test Cases:
- 교수 휴직 → 비품 규정 검색됨 → 비품 내용이 답변에 없어야 함
- 등록금 납부 → 비품 규정 검색됨 → 비품 내용이 답변에 없어야 함

### Acceptance: Accuracy ≥ 0.70 for queries with irrelevant search results

---

## Scenario 5: 답변 완전성

**Given** 검색 결과에 절차, 요건, 기한 등이 포함된 경우  
**When** 답변이 생성되면  
**Then** 해당 정보가 모두 답변에 포함되어야 한다

### Metrics:
- Completeness score: 0.413 → 0.650+ (average)
- 절차 쿼리: 단계별 나열 포함
- 요건 쿼리: 자격 조건 목록 포함
- 기한 쿼리: 날짜/기간 정보 포함

### Acceptance: Average completeness ≥ 0.65 across all 30 queries

---

## Scenario 6: 페르소나별 최소 품질

**Given** 6개 사용자 페르소나별 5개 쿼리가 있을 때  
**When** 전체 평가를 실행하면  
**Then** 모든 페르소나가 최소 품질 기준을 달성해야 한다

### Per-Persona Targets:

| Persona | Current Avg | Target |
|---------|------------|--------|
| 학부생 | 0.744 | 0.750+ |
| 대학원생 | 0.601 | 0.650+ |
| 교수 | 0.195 | 0.500+ |
| 교직원 | 0.551 | 0.600+ |
| 학부모 | 0.593 | 0.650+ |
| 유학생 | 0.122 | 0.500+ |

### Acceptance: 모든 페르소나 최소 0.500+ 달성

---

## 검증 방법

### 자동 검증
```bash
# 전체 30-쿼리 평가 실행
uv run python run_rag_quality_eval.py --quick --summary

# 결과 확인
# 1. pass_rate >= 0.60
# 2. avg_score >= 0.70
# 3. 기존 5개 통과 쿼리 유지 확인
```

### 단위 테스트
```bash
# Self-RAG 이중언어 테스트
uv run pytest tests/test_self_rag_bilingual.py -v

# CoT 제거 테스트
uv run pytest tests/test_cot_stripping.py -v

# 환각 방지 테스트
uv run pytest tests/test_hallucination_guard.py -v
```

---

## Phase별 Acceptance Gate

### Phase 1 완료 기준
- [ ] 영어 쿼리 5개 모두 Self-RAG 통과 (RETRIEVE_YES)
- [ ] 영어 쿼리 5개 모두 검색 결과 반환 (거부 메시지 없음)
- [ ] 기존 5개 통과 쿼리 회귀 없음

### Phase 2 완료 기준
- [ ] 30개 쿼리 응답에서 CoT 패턴 0건
- [ ] Average completeness ≥ 0.55
- [ ] 기존 5개 통과 쿼리 회귀 없음

### Phase 3 완료 기준
- [ ] 환각성 답변 감소 (관련 없는 규정 인용 0건)
- [ ] Overall pass rate ≥ 60%
- [ ] Average score ≥ 0.70
- [ ] 모든 페르소나 ≥ 0.50
