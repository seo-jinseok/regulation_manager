# Plan: RAG 테스트 100점 달성 - 근본적 개선

## 문제 정의

현재 동적 쿼리 성공률 67점(8/12)을 100점으로 올려야 합니다.

**4개 실패 쿼리:**
1. "장학금 받으려면 성적이 몇 점이어야 해?"
2. "졸업학점이 몇 학점이야?"
3. "졸업하려면 영어 점수도 필요해?"
4. "교수 승진 기준이 어떻게 됩니까?"

---

## 근본 원인 분석

### 현재 접근법의 한계

```
실패 쿼리 발생 → 해당 트리거/키워드 추가 → 테스트 통과
```

**문제점:**
- 확장성 없음: 새로운 표현마다 수동 추가 필요
- intents.json 비대화: 이미 2000줄+ → 유지보수 부담
- 패턴 폭발: 사용자의 무한한 표현을 예측 불가
- 근본 원인 미해결: "검색 반복 오류"는 인텐트 추가로 해결 안 됨

### 5-Why 분석

```
Why 1: 왜 "장학금 성적 기준" 쿼리가 실패했는가?
→ intents.json에 해당 트리거가 없어서

Why 2: 왜 트리거가 없으면 실패하는가?
→ 패턴 매칭 기반이라 사전 정의된 패턴만 인식

Why 3: 왜 패턴 매칭에 의존하는가?
→ LLM 호출 비용/지연 절약을 위해

Why 4: 왜 패턴 미매칭 시 폴백이 없는가?
→ 현재 아키텍처가 단일 경로만 지원

Why 5: 근본 원인은?
→ **의도 분류 시스템이 단일 계층(패턴 매칭)에만 의존**
```

---

## 해결 전략: 3단계 접근

### Phase 1: 단기 (설정 파일 수정) - 즉시 테스트 통과용

**목표**: 4개 실패 쿼리 해결하여 동적 쿼리 100점 달성

**작업:**
1. intents.json에 3개 신규 인텐트 추가
2. synonyms.json에 8개 동의어 그룹 추가
3. graduation 인텐트 keywords 강화

**한계**: 임시방편, 새로운 표현에는 다시 실패

---

### Phase 2: 중기 (코드 개선) - 근본 해결

**목표**: 패턴 미매칭 쿼리도 처리할 수 있는 폴백 시스템 구축

#### 2.1 2단계 의도 분류 시스템

**파일**: `src/rag/infrastructure/query_analyzer.py`

```python
def classify_intent(self, query: str) -> IntentResult:
    # 1단계: 패턴 매칭 (빠름, 비용 무료)
    pattern_result = self._match_patterns(query)
    if pattern_result.confidence >= 0.8:
        return pattern_result
    
    # 2단계: LLM 폴백 (느림, 비용 있음)
    return self._llm_classify_intent(query)

def _llm_classify_intent(self, query: str) -> IntentResult:
    """패턴 매칭 실패 시 LLM으로 의도 분류"""
    prompt = f"""
    대학 규정 시스템입니다. 사용자 질문의 의도를 분석하세요.
    
    질문: {query}
    
    다음 중 가장 적합한 의도를 선택하고 검색 키워드를 추출하세요:
    - scholarship: 장학금 관련
    - graduation: 졸업 요건 관련
    - faculty: 교원/교수 관련
    - student_status: 학적 관련
    - other: 기타
    
    JSON 형식: {{"intent": "...", "keywords": ["...", "..."], "confidence": 0.9}}
    """
    return self.llm.generate_json(prompt)
```

#### 2.2 검색 전략 분기

**파일**: `src/rag/application/search_usecase.py`

```python
def _determine_search_strategy(self, query: str) -> SearchStrategy:
    """쿼리 특성에 따라 검색 전략 결정"""
    # 단순 사실 질문 (3단어 이하, 명확한 의도)
    if self._is_simple_factual(query):
        return SearchStrategy.DIRECT  # Tool Calling 우회
    
    # 복합/모호 쿼리
    return SearchStrategy.TOOL_CALLING

def _is_simple_factual(self, query: str) -> bool:
    """단순 사실 질문 감지"""
    simple_patterns = [
        r"^.{2,10}(이|가)\s*(몇|얼마|언제)",  # "졸업학점이 몇"
        r"^.{2,10}\s*어떻게\s*(돼|됩니까|되나요)",
    ]
    return any(re.match(p, query) for p in simple_patterns)
```

#### 2.3 검색 반복 제한 개선

**파일**: `src/rag/infrastructure/function_gemma_adapter.py`

```python
MAX_SEARCH_RETRIES = 2  # 기존 3에서 축소

def _handle_insufficient_results(self, query: str, results: list) -> str:
    """검색 결과 불충분 시 현재 결과로 답변 생성"""
    if len(results) == 0:
        return "관련 규정을 찾지 못했습니다. 질문을 더 구체적으로 해주세요."
    
    # 결과가 있으면 현재 결과로 답변 생성 (재검색 대신)
    return self._generate_answer_from_partial_results(query, results)
```

---

### Phase 3: 장기 (아키텍처 개선) - 확장성 확보

#### 3.1 동적 쿼리 확장

**파일**: `src/rag/infrastructure/query_expander.py` (신규)

```python
class DynamicQueryExpander:
    """LLM 기반 동적 쿼리 확장"""
    
    def expand(self, query: str) -> list[str]:
        prompt = f"""
        대학 규정 검색을 위한 키워드를 추출하세요.
        
        질문: {query}
        
        5개의 검색 키워드를 JSON 배열로 반환:
        ["키워드1", "키워드2", ...]
        """
        return self.llm.generate_json(prompt)
```

#### 3.2 실패 쿼리 자동 학습

**파일**: `scripts/learn_from_failures.py` (신규)

```python
def auto_generate_intent(failed_query: str, expected_result: str) -> dict:
    """실패 쿼리에서 자동으로 인텐트 생성"""
    prompt = f"""
    다음 쿼리가 실패했습니다. 새로운 인텐트를 생성하세요.
    
    쿼리: {failed_query}
    기대 결과: {expected_result}
    
    intents.json 형식으로 반환:
    {{
        "id": "...",
        "triggers": ["...", "..."],
        "keywords": ["...", "..."]
    }}
    """
    return llm.generate_json(prompt)
```

---

## 실행 계획

### Step 1: Phase 1 실행 (즉시)

```bash
# 1. intents.json 수정
# 2. synonyms.json 수정
# 3. 회귀 테스트
uv run pytest

# 4. 4개 실패 쿼리 재테스트
uv run regulation search "장학금 받으려면 성적이 몇 점이어야 해?" -a
uv run regulation search "졸업학점이 몇 학점이야?" -a
uv run regulation search "졸업하려면 영어 점수도 필요해?" -a
uv run regulation search "교수 승진 기준이 어떻게 됩니까?" -a

# 5. 전체 정적 테스트
uv run python scripts/auto_evaluate.py --run
```

### Step 2: Phase 2 구현 (1-2일)

1. `query_analyzer.py`에 LLM 폴백 추가
2. `search_usecase.py`에 검색 전략 분기 추가
3. `function_gemma_adapter.py` 검색 반복 제한 개선
4. 단위 테스트 작성 및 통과 확인

### Step 3: Phase 3 구현 (3-5일)

1. `DynamicQueryExpander` 클래스 구현
2. `learn_from_failures.py` 스크립트 구현
3. 통합 테스트 및 성능 평가

---

## Phase 1 상세: 설정 파일 변경사항

### intents.json 추가 (3개 인텐트)

<details>
<summary>scholarship_grade_requirement</summary>

```json
{
    "id": "scholarship_grade_requirement",
    "label": "장학금 성적 기준",
    "triggers": [
        "장학금 받으려면 성적이 몇 점",
        "장학금 성적 기준",
        "장학금 받으려면 몇 점",
        "장학금 성적 조건",
        "장학금 평점"
    ],
    "keywords": [
        "장학금", "성적기준", "장학금지급규정", "성적우수장학금"
    ],
    "audience": "student",
    "weight": 1.5
}
```
</details>

<details>
<summary>graduation_language_requirement</summary>

```json
{
    "id": "graduation_language_requirement",
    "label": "졸업 외국어 요건",
    "triggers": [
        "졸업하려면 영어 점수",
        "졸업 영어 요건",
        "토익 점수 필요",
        "영어 점수도 필요해"
    ],
    "keywords": [
        "어학인증", "졸업요건", "토익", "TOEIC", "동의대학교학칙"
    ],
    "audience": "student",
    "weight": 1.5
}
```
</details>

<details>
<summary>faculty_promotion</summary>

```json
{
    "id": "faculty_promotion",
    "label": "교원 승진 요건",
    "triggers": [
        "교수 승진 기준",
        "교수 승진 기준이 어떻게",
        "승진 기준이 어떻게 됩니까"
    ],
    "keywords": [
        "승진", "교원인사규정", "업적평가", "승진임용"
    ],
    "audience": "faculty",
    "weight": 1.5
}
```
</details>

### synonyms.json 추가 (8개 그룹)

```json
{
    "졸업학점": ["이수학점", "총이수학점", "최소이수학점"],
    "영어점수": ["토익", "TOEIC", "어학성적", "어학인증"],
    "승진기준": ["승진 요건", "진급 기준", "승급 요건"],
    "교원승진": ["교수 승진", "교수 진급"]
}
```

---

## 성공 기준

| Phase | 메트릭 | 목표 |
|-------|--------|------|
| Phase 1 | 동적 쿼리 성공률 | 100점 (12/12) |
| Phase 2 | 신규 쿼리 대응률 | ≥90% (패턴 미정의 쿼리) |
| Phase 3 | intents.json 증가율 | ≤10% (자동 학습으로 수동 추가 최소화) |

---

## 의사결정 필요

1. **Phase 1만 먼저** 진행하여 100점 달성 후 Phase 2로?
2. **Phase 2를 병행**하여 근본 해결과 함께?
3. **Phase 2 우선**하여 근본 해결 후 테스트?

권장: **Phase 1 → Phase 2 순차 진행** (즉시 성과 + 근본 해결)
