---
description: 다양한 사용자 유형별 RAG 응답 품질 테스트 및 개선사항 도출
---

# 포괄적 RAG 테스트 및 개선 워크플로우

이 워크플로우는 다양한 유형의 사용자 입장에서 RAG 시스템을 체계적으로 테스트하고, 응답 품질을 분석하여 개선사항을 도출합니다.

## 사전 준비

// turbo
1. **환경 확인**
   RAG 시스템이 정상적으로 동작하는지 확인합니다.
   
   ```bash
   uv run regulation status
   ```
   
   ChromaDB에 데이터가 로드되어 있는지 확인합니다.

---

## Phase 1: 페르소나별 테스트 케이스 수집

### 1.1 테스트 케이스 파일 분석

기존 테스트 케이스를 분석하여 커버리지 현황을 파악합니다.

// turbo
```bash
cat data/config/evaluation_dataset.json | head -100
```

### 1.2 사용자 페르소나 정의

다음 네 가지 페르소나를 기준으로 테스트합니다:

| 페르소나 | 설명 | 주요 관심사 |
|----------|------|-------------|
| **학생** | 학부생, 대학원생 | 휴학, 졸업, 장학금, 전과, 학사경고 |
| **교원** | 교수, 강사 | 연구년, 승진, 강의, 연구윤리, 해외파견 |
| **직원** | 일반 행정직원 | 인사, 복무, 휴가, 퇴직 |
| **공통** | 모든 구성원 | 성희롱 신고, 시설 이용, 기숙사, 주차 |

---

## Phase 2: 체계적 테스트 수행

### 2.1 학생 페르소나 테스트

학생 관점에서 다양한 쿼리를 테스트합니다. 각 쿼리의 응답을 분석하고 기대 응답과 비교합니다.

**자연어 질문**:
// turbo
```bash
uv run regulation search "휴학하고 싶어" -a
```

// turbo
```bash
uv run regulation search "장학금 받고 싶어요" -a
```

// turbo  
```bash
uv run regulation search "졸업 요건이 뭐야?" -a
```

// turbo
```bash
uv run regulation search "학사경고 받으면 어떻게 돼?" -a
```

// turbo
```bash
uv run regulation search "전과 신청 방법" -a
```

**간접 의도 표현 테스트**:
// turbo
```bash
uv run regulation search "공부하기 싫어" -a
```

// turbo
```bash
uv run regulation search "학교 그만두고 싶어" -a
```

// turbo
```bash
uv run regulation search "성적이 너무 낮아" -a
```

---

### 2.2 교원 페르소나 테스트

교원 관점에서 테스트합니다.

**자연어 질문**:
// turbo
```bash
uv run regulation search "연구년 신청하고 싶어" -a
```

// turbo
```bash
uv run regulation search "승진 요건이 뭐야?" -a
```

// turbo
```bash
uv run regulation search "해외 학회 참석하고 싶어" -a
```

// turbo
```bash
uv run regulation search "강의 면제 받으려면?" -a
```

**간접 의도 표현 테스트**:
// turbo
```bash
uv run regulation search "학교 가기 싫어" -a
```

// turbo
```bash
uv run regulation search "그만두고 싶어" -a
```

// turbo
```bash
uv run regulation search "수업 안 하고 싶어" -a
```

---

### 2.3 직원 페르소나 테스트

직원 관점에서 테스트합니다.

// turbo
```bash
uv run regulation search "휴가 규정 알려줘" -a
```

// turbo
```bash
uv run regulation search "퇴직금 어떻게 계산해?" -a
```

// turbo
```bash
uv run regulation search "육아휴직 쓰고 싶어" -a
```

---

### 2.4 공통 테스트

모든 구성원에게 해당하는 테스트입니다.

// turbo
```bash
uv run regulation search "성희롱 당했어" -a
```

// turbo
```bash
uv run regulation search "기숙사 살고 싶어" -a
```

// turbo
```bash
uv run regulation search "주차 등록 방법" -a
```

// turbo
```bash
uv run regulation search "연구윤리 위반 신고" -a
```

---

### 2.5 조문 검색 테스트

정확한 조문 검색 기능을 테스트합니다.

// turbo
```bash
uv run regulation search "제15조" -n 5
```

// turbo
```bash
uv run regulation search "교원인사규정 제10조" -n 5
```

---

### 2.6 에지 케이스 테스트

시스템의 한계를 테스트합니다.

// turbo
```bash
uv run regulation search "맛있는 점심 추천해줘" -a
```

// turbo
```bash
uv run regulation search "날씨 어때?" -a
```

// turbo
```bash
uv run regulation search "" -a
```

---

## Phase 3: 결과 분석 및 개선사항 도출

### 3.1 자동 평가 실행

// turbo
```bash
uv run python scripts/auto_evaluate.py --run
```

### 3.2 개선 제안 확인

```bash
cat data/output/improvement_plan.json
```

### 3.3 분석 항목 체크리스트

각 테스트 결과에 대해 다음 항목을 면밀히 분석합니다:

- [ ] **검색 결과 관련성**: 반환된 청크가 질문과 관련 있는가?
- [ ] **LLM 응답 정확도**: 답변이 규정에 근거한 정확한 정보인가?
- [ ] **의도 인식**: 간접적 표현이 올바른 의도로 해석되었는가?
- [ ] **대상 구분**: 학생/교원/직원 대상이 올바르게 구분되었는가?
- [ ] **동의어 확장**: 유사 용어가 적절히 확장되었는가?
- [ ] **에지 케이스 처리**: 관련 없는 질문에 대해 적절히 안내하는가?

---

## Phase 4: 개선 적용

### 4.1 인텐트 규칙 추가

새로운 의도 패턴이 발견되면 `intents.json`에 추가합니다:

```bash
# 인텐트 파일 위치
vim data/config/intents.json
```

### 4.2 동의어 추가

새로운 동의어가 필요하면 `synonyms.json`에 추가합니다:

```bash
# 동의어 파일 위치
vim data/config/synonyms.json
```

### 4.3 자동 생성 스크립트 활용

LLM을 활용하여 인텐트/동의어를 자동 생성할 수 있습니다:

```bash
# 인텐트 자동 생성
uv run python scripts/generate_intents.py

# 동의어 자동 생성
uv run python scripts/generate_synonyms.py
```

---

## Phase 5: 재검증

### 5.1 변경 후 테스트

개선사항 적용 후 동일한 테스트를 재수행하여 개선 효과를 확인합니다.

// turbo
```bash
uv run pytest tests/rag/ -v --tb=short
```

### 5.2 평가 재실행

// turbo
```bash
uv run python scripts/auto_evaluate.py --run
```

### 5.3 결과 비교

이전 평가 결과와 비교하여 개선 여부를 판단합니다:

- 통과율 변화 확인
- 실패 케이스 수 감소 여부
- 새로운 실패 케이스 발생 여부

---

## 부록: 테스트 쿼리 참고 목록

### 학생 쿼리 예시

| 카테고리 | 직접 표현 | 간접 표현 |
|----------|-----------|-----------|
| 휴학 | 휴학 신청 방법 | 학교 안 다니고 싶어 |
| 졸업 | 졸업 요건 뭐야 | 빨리 끝내고 싶어 |
| 장학금 | 장학금 받고 싶어 | 돈 없어 |
| 성적 | 학사경고 기준 | 성적 망했어 |
| 전과 | 전과 신청 | 다른 과 가고 싶어 |

### 교원 쿼리 예시

| 카테고리 | 직접 표현 | 간접 표현 |
|----------|-----------|-----------|
| 연구년 | 연구년 신청하고 싶어 | 강의 안 하고 싶어 |
| 승진 | 교수 승진 요건 | 정교수 되고 싶어 |
| 휴직 | 병가 쓰고 싶어 | 아파서 못 나가 |
| 퇴직 | 명예퇴직 신청 | 그만두고 싶어 |

### 직원 쿼리 예시

| 카테고리 | 직접 표현 | 간접 표현 |
|----------|-----------|-----------|
| 휴가 | 연가 규정 | 쉬고 싶어 |
| 퇴직 | 퇴직금 규정 | 때려치고 싶어 |
| 육아 | 육아휴직 신청 | 아이 때문에 쉬고 싶어 |

---

## 완료 조건

- [ ] 모든 페르소나별 테스트 완료
- [ ] 검색 결과 분석 완료
- [ ] 개선 필요 항목 문서화
- [ ] 인텐트/동의어 추가 (필요시)
- [ ] 재검증 통과

---

*이 워크플로우는 `/comprehensive_rag_testing` 명령으로 실행할 수 있습니다.*
