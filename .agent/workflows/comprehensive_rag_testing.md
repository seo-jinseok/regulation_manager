---
description: 다양한 사용자 유형별 RAG 응답 품질 테스트 및 개선사항 도출
---

# 포괄적 RAG 테스트 및 개선 워크플로우

이 워크플로우는 다양한 유형의 사용자 입장에서 RAG 시스템을 체계적으로 테스트하고, 응답 품질을 분석하여 **근본적인 시스템 개선**을 도출합니다.

> **핵심 특징**:
> - **동적 쿼리 생성**: 매 실행마다 LLM이 새로운 테스트 쿼리를 생성
> - **근본적 개선 유도**: 데이터 패치(intents/synonyms) 외에 **코드 레벨 개선** 제안

---

## 사전 준비

// turbo
1. **환경 확인**
   RAG 시스템이 정상적으로 동작하는지 확인합니다.
   
   ```bash
   uv run regulation status
   ```
   
   ChromaDB에 데이터가 로드되어 있는지 확인합니다.

---

## Phase 1: 동적 테스트 쿼리 생성

기존의 고정된 테스트 케이스 대신, LLM을 활용해 **새로운 테스트 쿼리를 동적으로 생성**합니다.

### 1.1 모든 페르소나 쿼리 생성

// turbo
```bash
uv run python scripts/generate_test_queries.py --all --count 5
```

이 명령은 4가지 페르소나(학생, 교원, 직원, 공통)별로 각 5개의 새로운 쿼리를 생성합니다.

### 1.2 생성된 쿼리 확인

```bash
ls -la data/output/generated_queries_*.json | tail -1
```

생성된 쿼리 파일을 확인하고 내용을 검토합니다:

```bash
cat $(ls data/output/generated_queries_*.json | tail -1) | head -50
```

### 1.3 특정 페르소나 추가 생성 (선택적)

필요시 특정 페르소나에 대해 추가 쿼리를 생성합니다:

// turbo
```bash
uv run python scripts/generate_test_queries.py --persona student --count 3
```

---

## Phase 2: 테스트 수행

### 2.1 생성된 쿼리로 수동 테스트

생성된 쿼리 파일에서 샘플을 추출하여 테스트합니다:

// turbo
```bash
# 생성된 쿼리 중 첫 번째 학생 쿼리 테스트 (예시)
uv run regulation search "휴학 신청 절차" -a
```

### 2.2 자동 평가 실행

기존 평가 데이터셋과 함께 자동 평가를 수행합니다:

// turbo
```bash
uv run python scripts/auto_evaluate.py --run
```


---

### 2.3 플랫폼 간 일관성 검증 (CLI vs Web)

사용자가 어떤 인터페이스를 사용하든 동일한 품질의 답변을 받아야 합니다.

1. **테스트 쿼리 선정**: 생성된 쿼리 중 3~5개를 무작위 선정합니다.
2. **CLI 실행**:
   ```bash
   uv run regulation search "선정된 쿼리"
   ```
3. **Web UI 실행**:
   ```bash
   uv run regulation serve --web
   ```
   웹 브라우저에서 동일한 쿼리를 입력하여 실행합니다.
4. **결과 비교**:
   - 검색된 규정 조항(청크)이 동일한가?
   - LLM 답변의 내용과 뉘앙스가 유사한가?
   - 제공되는 메타데이터(참조 링크 등)가 일관적인가?

---

## Phase 3: 결과 분석 및 개선사항 도출

### 3.1 개선 제안 확인

```bash
cat data/output/improvement_plan.json
```

### 3.2 제안 유형 확인

개선 제안은 다음 유형으로 분류됩니다:

| 유형 | 설명 | 개선 방향 |
|------|------|-----------|
| `intent` | intents.json 트리거 추가 | 데이터 패치 |
| `synonym` | synonyms.json 동의어 추가 | 데이터 패치 |
| `code_pattern` | QueryAnalyzer 패턴 로직 | **코드 개선** |
| `code_weight` | 가중치 프리셋 조정 | **코드 개선** |
| `code_audience` | 대상 감지 로직 | **코드 개선** |
| `architecture` | 파이프라인 구조 | **코드 개선** |

### 3.3 분석 체크리스트

각 테스트 결과에 대해 다음 항목을 분석합니다:

- [ ] **검색 결과 관련성**: 반환된 청크가 질문과 관련 있는가?
- [ ] **LLM 응답 정확도**: 답변이 규정에 근거한 정확한 정보인가?
- [ ] **의도 인식**: 간접적 표현이 올바른 의도로 해석되었는가?
- [ ] **대상 구분**: 학생/교원/직원 대상이 올바르게 구분되었는가?
- [ ] **동의어 확장**: 유사 용어가 적절히 확장되었는가?
- [ ] **플랫폼 일관성**: CLI와 Web UI의 결과가 실질적으로 동일한가?

---

## Phase 4: 개선 적용

### 4.1 데이터 패치 (1회성 개선)

단순 트리거/동의어 누락인 경우:

```bash
# 인텐트 파일 수정
vim data/config/intents.json

# 동의어 파일 수정  
vim data/config/synonyms.json
```

### 4.2 코드 레벨 개선 (근본적 개선)

`code_*` 또는 `architecture` 유형 제안인 경우, 해당 파일의 로직을 수정합니다:

| 제안 유형 | 수정 대상 파일 |
|-----------|---------------|
| `code_pattern` | `src/rag/infrastructure/query_analyzer.py` → `INTENT_PATTERNS` |
| `code_weight` | `src/rag/infrastructure/query_analyzer.py` → `WEIGHT_PRESETS` |
| `code_audience` | `src/rag/infrastructure/query_analyzer.py` → `*_KEYWORDS` |
| `architecture` | `src/rag/infrastructure/hybrid_search.py`, `reranker.py` |

### 4.3 개선 우선순위

1. **코드 레벨 개선 우선**: 반복 발생하는 패턴은 코드로 해결
2. **데이터 패치 최소화**: 개별 케이스만 데이터로 해결
3. **테스트 케이스 추가**: 수정 후 `evaluation_dataset.json`에 케이스 추가

---

## Phase 5: 재검증

### 5.1 단위 테스트 실행

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

## 완료 조건

- [ ] 동적 쿼리 생성 완료 (Phase 1)
- [ ] 테스트 수행 완료 (Phase 2)
- [ ] 개선 제안 분석 완료 (Phase 3)
- [ ] 코드 레벨 / 데이터 패치 적용 (Phase 4)
- [ ] 재검증 통과 (Phase 5)

---

*이 워크플로우는 `/comprehensive_rag_testing` 명령으로 실행할 수 있습니다.*
