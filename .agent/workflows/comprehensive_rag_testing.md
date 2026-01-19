---
description: AI 에이전트 자율 실행을 위한 RAG 시스템 품질 심층 테스트 및 개선 워크플로우 (Strict Mode v2.2)
---

# RAG 시스템 품질 심층 테스트 및 개선 (Strict Mode v2.2)

AI 에이전트가 다양한 사용자 페르소나를 시뮬레이션하여 RAG 시스템의 품질을 **엄격하게** 테스트하고, 답변 품질을 **비판적으로** 검토하는 워크플로우입니다.

**v2.2 주요 기능:**
- 🆕 **Dynamic Query Expansion** 컴포넌트 검증
- 🆕 **Fact Check 기본 활성화** (`ENABLE_FACT_CHECK=true`)
- 🆕 Tool Calling CLI 옵션 (`--no-tools`, `--tool-mode`)
- 🆕 synonym CLI 서브커맨드 (suggest --auto-add, --context)
- 🆕 BM25 인덱스 캐시 기능
- 고급 RAG 컴포넌트 단위 테스트 (Self-RAG, HyDE, Corrective RAG 등)
- 의도 추론 정확도 검증 (3단계 의도 분석)
- RAG 컴포넌트 기여도 분석
- 입력 검증 및 보안 테스트

## ⚠️ 평가자 마인드셋 (중요!)

> **당신은 까다로운 품질 검수자입니다. 사용자 입장에서 "이게 정말 도움이 되는가?"를 냉정하게 판단하세요.**
>
> - ❌ "대체로 맞는 것 같다" → 팩트체크 없이 통과시키지 마세요
> - ❌ "규정 조항이 언급되었다" → 그 조항이 실제로 존재하는지 확인했나요?
> - ❌ "답변이 길고 상세하다" → 핵심 정보가 정확한지가 중요합니다
> - ❌ "일반적으로 맞는 내용이다" → **이 학교**의 규정과 일치하는지 확인하세요
> - ❌ "절차를 설명했다" → 구체적인 기한, 서류, 담당부서가 명시되었나요?
> - ❌ "검색 결과가 나왔다" → **어떤 RAG 컴포넌트**가 결과에 기여했는지 분석했나요?

---

## 고급 RAG 컴포넌트 개요

| 컴포넌트 | 역할 | 활성화 조건 | 검증 포인트 |
|----------|------|-------------|-------------|
| **Self-RAG** | 검색 필요성 판단 + 결과 관련성 평가 | `ENABLE_SELF_RAG=true` | 인사말에서 검색 스킵 |
| **HyDE** | 모호한 쿼리 → 가상 문서 생성 | `ENABLE_HYDE=true` | 모호 쿼리 처리 |
| **Corrective RAG** | 검색 결과 품질 평가 → 재검색 | 동적 임계값 (0.3~0.5) | 낮은 점수 재검색 |
| **Hybrid Search** | BM25 + Dense 검색 융합 (RRF) | `use_hybrid=True` | 키워드+의미 균형 |
| **BGE Reranker** | 검색 결과 재정렬 | `use_reranker=True` | 최종 순위 품질 |
| **Tool Calling** | Agentic RAG | 기본 활성화 | 도구 선택 정확성 |
| **Query Analyzer** | 인텐트 분석 + 쿼리 확장 | 항상 활성 | 의도 파악 |
| **Dynamic Query Expansion** | LLM 기반 동적 쿼리 확장 | `ENABLE_QUERY_EXPANSION=true` | 키워드 확장 품질 |
| **Fact Check** | 답변 팩트체크 및 재생성 | `ENABLE_FACT_CHECK=true` | 오류 시 재생성 |

## 성공 기준

| 메트릭 | 목표 | 엄격 모드 적용 |
|--------|------|---------------|
| 정적 테스트 통과율 | ≥ 85% | - |
| 동적 쿼리 성공률 | ≥ 80% | **"부분 성공" = 실패** |
| 멀티턴 대화 성공률 | ≥ 80% | **최소 3턴 후속 질문 필수** |
| 후속 질문 성공률 | ≥ 85% | 각 Turn별 성공률 |
| 답변 품질 점수 | ≥ 4.0/5.0 | 팩트체크 실패 시 **최대 2.0점** |
| 팩트체크 수행률 | **100%** | 모든 답변 필수 검증 |
| 의도 추론 정확도 | ≥ 85% | AI 파악 의도 = 실제 의도 |
| RAG 컴포넌트 기여도 | **분석 필수** | 각 기능 영향 분석 |

---

## Phase 0: 사전 검증

### 0.1 시스템 상태 확인
// turbo
```bash
uv run regulation status
```

### 0.2 데이터 동기화 (필요시)
```bash
uv run regulation sync data/output/규정집.json
```

### 0.3 LLM 연결 확인
// turbo
```bash
# .env의 LLM_PROVIDER에 따라
curl -s $LLM_BASE_URL/v1/models | head -5
# 또는 Ollama:
curl http://localhost:11434/api/tags 2>/dev/null || echo "Ollama 미실행"
```

### 0.4 고급 RAG 설정 확인
// turbo
```bash
cat .env | grep -E "(ENABLE_SELF_RAG|ENABLE_HYDE|BM25_TOKENIZE_MODE|HYDE_CACHE|ENABLE_FACT_CHECK|ENABLE_QUERY_EXPANSION|BM25_INDEX_CACHE_PATH)"
```

---

## Phase 1: 정적 평가 실행

### 1.1 자동 평가 실행
// turbo
```bash
uv run python scripts/auto_evaluate.py --run
```

### 1.2 결과 판정
- 통과율 ≥ 85% → Phase 1.5로 진행
- 통과율 < 85% → Phase 5 (개선)로 이동

---

## Phase 1.5: 고급 RAG 컴포넌트 단위 테스트

> ⚠️ **각 RAG 컴포넌트가 제대로 작동하는지 개별 검증합니다.**

### 1.5.1 Self-RAG 검증

| 쿼리 유형 | 예시 | 기대 동작 |
|----------|------|----------|
| 인사말 | "안녕하세요" | `RETRIEVE_NO` → 검색 스킵 |
| 단순 정보 | "휴학 신청 기간" | `RETRIEVE_YES` → 검색 수행 |

```bash
uv run regulation --debug search "안녕하세요" -n 1
# 로그에서 "Self-RAG: needs_retrieval=False" 확인

uv run regulation --debug search "휴학 신청 기간" -n 3
# 로그에서 "Self-RAG: needs_retrieval=True" 확인
```

### 1.5.2 HyDE 검증

| 쿼리 유형 | 예시 | HyDE 발동 |
|----------|------|----------|
| 명확한 쿼리 | "휴학 신청 기간" | ❌ |
| 모호한 쿼리 | "학교 가기 싫어" | ✅ |
| 감정 표현 | "교수님이 너무 힘들어요" | ✅ |

```bash
# HyDE 비교 테스트
uv run regulation search "학교 가기 싫어" -n 5
ENABLE_HYDE=false uv run regulation search "학교 가기 싫어" -n 5
```

### 1.5.3 Corrective RAG 검증

```bash
uv run regulation --debug search "희귀한키워드조합" -n 5
# 로그에서 "Corrective RAG: triggered" 확인
```

### 1.5.4 Hybrid Search (BM25 + Dense) 검증

| 쿼리 특성 | 예시 | 우세 |
|----------|------|------|
| 정확한 키워드 | "교원인사규정 제8조" | BM25 |
| 의미적 유사 | "교수 승진 조건" | Dense |

```bash
uv run regulation --debug search "교원인사규정 제8조" -n 5
```

### 1.5.5 BGE Reranker 검증

```bash
uv run regulation search "휴학 신청" -n 5
uv run regulation search "휴학 신청" -n 5 --no-rerank
# 순위 비교
```

### 1.5.6 Query Analyzer (인텐트 분석) 검증

```bash
uv run regulation --debug search "학교 안 가고 싶어" -a -n 5
# 로그에서 "[의도 분석]" 확인
cat data/config/intents.json | python -m json.tool | head -30
```

### 1.5.7 Fact Check 검증

```bash
uv run regulation --debug search "휴학 신청 기간" -a -n 5
# 로그에서 "Fact check passed" 확인

ENABLE_FACT_CHECK=false uv run regulation search "휴학 신청 기간" -a -n 5
```

### 1.5.8 Dynamic Query Expansion 검증

```bash
uv run regulation --debug search "학교 가기 싫어" -a -n 5
# 로그에서 "Query expansion:" 확인

ENABLE_QUERY_EXPANSION=false uv run regulation search "학교 가기 싫어" -a -n 5
```

### 1.5.9 입력 검증 및 보안 테스트

```bash
# 길이 제한 테스트 (500자 초과)
uv run regulation search "$(python -c 'print("a"*501)')" -n 1

# XSS 패턴 차단
uv run regulation search "<script>alert(1)</script>" -n 1

# SQL Injection 차단
uv run regulation search "'; DROP TABLE regulations; --" -n 1
```

### 1.5.10 컴포넌트 통합 테스트

```bash
uv run regulation search "돈 없어서 학교 다니기 힘들어요" -a -n 5
ENABLE_HYDE=false uv run regulation search "돈 없어서 학교 다니기 힘들어요" -a -n 5
```

**컴포넌트 기여도 분석 기록:**
```
[통합 테스트] 쿼리: "돈 없어서 학교 다니기 힘들어요"

| 컴포넌트 | 활성화 | 동작 여부 | 기여도 |
|----------|--------|----------|--------|
| Self-RAG | ✅ | 검색 필요 판단 | 정상 |
| HyDE | ✅ | 가상 문서 생성 | 장학금/분납 키워드 추가 |
| Query Analyzer | ✅ | 인텐트: 경제적 어려움 | 키워드 확장 |
| Dynamic Query Expansion | ✅ | LLM 확장 | 지원금/학자금 추가 |
| Hybrid Search | ✅ | BM25+Dense 융합 | 정상 |
| Corrective RAG | ❌ | 미트리거 | - |
| Reranker | ✅ | 재정렬 수행 | 장학금 규정 상위 |
| Fact Check | ✅ | 검증 통과 | 정상 |
```

---

## Phase 2: 동적 쿼리 테스트 - 의도 추론 중심

### 2.1 페르소나 정의

**3~5개 무작위 선택:**

| 페르소나 | 특성 | 예상 관심사 |
|----------|------|-------------|
| 🎓 신입생 | 비공식적 표현 | 수강신청, 장학금, 휴학 |
| 📚 재학생 (3학년) | 졸업 준비 | 졸업요건, 전과, 복수전공 |
| 🎓 대학원생 | 연구/논문 중심 | 논문심사, 연구비, 학위 |
| 👨‍🏫 교수 | 제도 파악 | 연구년, 승진, 업적평가 |
| 👔 직원 | 복무규정 | 휴가, 복리후생, 겸직 |
| 🤕 어려운 상황 학생 | 급한 상황 | 제적위기, 학사경고 |
| 😡 불만있는 구성원 | 권리 주장 | 인권침해, 고충처리 |
| 👪 학부모 | 외부 시선 | 등록금, 장학금 |

### 2.2 쿼리 난이도 분포

| 난이도 | 비율 | 특성 | 예시 |
|--------|------|------|------|
| **쉬움** | 30% | 단일 규정, 명확 키워드 | "휴학 신청 기간" |
| **중간** | 40% | 여러 규정 연계, 조건부 | "장학금 받다가 휴학하면?" |
| **어려움** | 30% | 모호, 감정적, 복합 | "돈이 없어서 학교 다니기 힘들어요" |

### 2.3 의도 추론 3단계

각 쿼리에 대해:
1. **표면적 의도**: 직접 표현된 것
2. **숨겨진 의도**: 실제 니즈
3. **행동 의도**: 궁극적 행동

**의도 추론 정확도:**
| 등급 | 조건 | 점수 |
|------|------|------|
| **정확** | 3단계 모두 정확 | 100% |
| **부분 정확** | 숨겨진 의도 일부 누락 | 70% |
| **표면적 파악** | 표면적 의도만 파악 | 40% |
| **오해** | 의도를 잘못 파악 | 0% |

### 2.4 쿼리 실행

```bash
uv run regulation search "<생성된_쿼리>" -a -n 5
```

### 2.5 의도 충족 체크리스트

모든 답변에 대해 확인:
- [ ] 구체적 **행동**이 명확히 안내되었는가?
- [ ] **다음 단계**가 무엇인지 알 수 있는가?
- [ ] 필요한 **준비물/조건**이 명시되었는가?
- [ ] **기한**이 명시되었는가?
- [ ] **담당부서/연락처**가 안내되었는가?

**판정:**
- ✅ **충족**: 5개 이상 통과 → 성공
- ⚠️ **부분 충족**: 3~4개 통과 → **실패로 카운트**
- ❌ **미충족**: 2개 이하 → 실패

---

## Phase 3: 답변 품질 심층 검토

> ⚠️ **모든 답변에 대해 팩트체크 필수. 생략 금지!**

### 3.1 팩트체크 절차

각 답변에서 **핵심 주장 3개** 추출 후 검증:

```bash
uv run regulation search "<규정명> 제X조" -n 3
uv run regulation search "<키워드> 기간|학점|요건" -n 5
```

**기록 형식:**
```
[팩트체크 #1]
- 답변 주장: "휴학은 수업일수 2/3까지 가능"
- 검증 쿼리: uv run regulation search "휴학 수업일수" -n 3
- 검증 결과: ✅ 학칙 제XX조에서 확인됨 / ❌ 해당 내용 없음
```

### 3.2 품질 평가 매트릭스

| 항목 | 배점 | 자동 감점 조건 |
|------|------|---------------|
| **정확성** | 1.0 | 팩트체크 실패 시 **0점** |
| **완전성** | 1.0 | 복합질문 1개 누락 시 **-0.5** |
| **관련성** | 1.0 | 50%↑ 무관 내용 시 **0점** |
| **출처 명시** | 1.0 | 출처 없는 단정 시 **0점** |
| **실용성** | 0.5 | 기한 없으면 **-0.25** |
| **행동 가능성** | 0.5 | 기한/서류/부서 2개↑ 누락 시 **0점** |
| **합계** | 5.0 | |

### 3.3 성공/실패 판정

| 판정 | 조건 |
|------|------|
| ✅ **성공** | 점수 ≥ 4.0 AND 팩트체크 모두 통과 |
| ⚠️ **부분성공** | 점수 3.0~3.9 OR 팩트체크 1개 실패 → **실패로 카운트** |
| ❌ **실패** | 점수 < 3.0 OR 팩트체크 2개↑ 실패 |

### 3.4 일반론 답변 감지 패턴 (자동 실패)

다음 패턴 포함 시 **GENERIC_ANSWER**로 즉시 실패:
- `대학마다 다를 수 있습니다`
- `확인이 필요합니다`
- `일반적으로` (규정 인용 없이)
- `담당 부서에 문의` (구체 부서명 없이)

### 3.5 실패 유형

| 실패 유형 | 심각도 | 판정 |
|----------|--------|------|
| `WRONG_FACT` | Critical | 즉시 실패 |
| `HALLUCINATION` | Critical | 즉시 실패 |
| `GENERIC_ANSWER` | High | 즉시 실패 |
| `IRRELEVANT` | High | 즉시 실패 |
| `INCOMPLETE` | Medium | 부분성공 |
| `NO_SOURCE` | Medium | 부분성공 |

### 3.6 RAG 컴포넌트 기여도 분석 (필수)

각 쿼리에 대해 기록:
```
[RAG 컴포넌트 분석]
| 컴포넌트 | 동작 | 기여도 |
|----------|------|--------|
| Self-RAG | RETRIEVE_YES | ✅ 긍정적 |
| HyDE | 발동 | ✅ 관련 결과 증가 |
| Query Analyzer | 인텐트 감지 | ⚠️ 부분적 |
| Corrective RAG | 미트리거 | - |
| Reranker | 재정렬 | ✅ 상위 배치 |
```

---

## Phase 4: 멀티턴 대화 테스트

> ⚠️ **모든 시나리오는 최소 3턴 이상 후속 질문 필수**

### 4.1 후속 질문 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **구체화** | 더 자세한 정보 | "정확히 몇 학점?" |
| **관련 확장** | 연관 주제 | "휴학하면 장학금은?" |
| **예외 확인** | 특수 상황 | "군대 가는 경우도?" |
| **절차 심화** | 구체적 절차 | "신청서 어디서?" |
| **조건 변경** | 다른 조건 | "대학원생도 마찬가지?" |

### 4.2 멀티턴 시나리오 최소 요건

> 각 페르소나에 대해 **2개 이상의 5턴 대화** 실행

**예시 시나리오:**
```
[Turn 1] "휴학하고 싶어요"
[Turn 2] "일반휴학이요. 신청 기간은?"
[Turn 3] "장학금 받고 있는데 어떻게 돼요?"
[Turn 4] "그럼 복학할 때는요?"
[Turn 5] "휴학 기간은 얼마까지?"
```

### 4.3 Turn별 의도 추론 검증

각 후속 질문(Turn 2+)에 대해:
1. **맥락 연결**: 이전 Turn 맥락 올바르게 연결?
2. **의도 진화**: 의도 발전 추적?
3. **암묵적 정보**: 명시 안 한 정보 기억?
4. **핵심 니즈**: 이번 Turn 핵심 니즈 파악?

### 4.4 멀티턴 평가 기준

| 항목 | 배점 |
|------|------|
| 맥락 유지 | 1.0 |
| 정보 일관성 | 1.0 |
| 점진적 심화 | 1.0 |
| 중복 회피 | 0.5 |
| 자연스러운 전환 | 0.5 |
| 후속 질문 예측 | 0.5 |
| 의도 추론 정확도 | 0.5 |
| **합계** | 5.0 |

### 4.5 CLI로 멀티턴 테스트

```bash
uv run regulation  # 인터랙티브 모드
uv run regulation --debug  # 디버그 모드
```

---

## Phase 4.5: 실패 케이스 심층 분석

### 5-Why 분석 템플릿

```
[실패 쿼리] "장학금 받고 있는데 휴학하면?"
[실패 유형] INCOMPLETE

Why 1: 왜 실패? → 장학금 유형별 처리 누락
Why 2: 왜 누락? → 검색 결과에 연계 정보 없음
Why 3: 왜 검색 안됨? → 키워드 조합이 인텐트에 없음
Why 4: 왜 인텐트에 없음? → 복합 시나리오 미반영
Why 5: 근본 원인? → 인텐트 설계가 단일 주제 중심

[RAG 컴포넌트별 분석]
| 컴포넌트 | 동작 | 문제점 |
|----------|------|--------|
| Query Analyzer | 단일 인텐트만 | 복합 인텐트 미지원 |
| HyDE | 미발동 | 발동 조건 개선 필요 |

[조치 방안]
1. intents.json에 복합 인텐트 추가
2. HyDE 발동 조건 완화
```

### 실패 유형별 RAG 원인 매핑

| 실패 유형 | 관련 컴포넌트 | 개선 방향 |
|----------|--------------|----------|
| `WRONG_FACT` | 생성 단계 | prompts.json |
| `HALLUCINATION` | Self-RAG | Self-RAG 강화 |
| `GENERIC_ANSWER` | Query Analyzer | 인텐트 추가 |
| `IRRELEVANT` | Query Analyzer | 인텐트 추가 |
| `INCOMPLETE` | Hybrid Search | 동의어 추가 |
| `NO_SOURCE` | 생성 단계 | prompts.json |

---

## Phase 5: 개선 적용

### 5.1 실패 분석

```bash
cat data/output/improvement_plan.json | python -m json.tool
```

### 5.2 제안 유형별 처리

| 유형 | 처리 방법 |
|------|----------|
| `intent` | `data/config/intents.json` 패치 |
| `synonym` | `data/config/synonyms.json` 패치 (CLI 권장) |
| `code_pattern` | `src/rag/infrastructure/query_analyzer.py` |
| `hyde_condition` | `src/rag/infrastructure/hyde.py` |
| `self_rag_prompt` | `src/rag/infrastructure/self_rag.py` |
| `query_expansion` | `src/rag/infrastructure/query_expander.py` |
| `fact_check` | `src/rag/infrastructure/fact_checker.py` |
| `prompt` | `data/config/prompts.json` |

### 5.3 동의어 CLI (권장)

```bash
uv run regulation synonym suggest "휴학"                   # 제안 확인
uv run regulation synonym suggest "휴학" --auto-add       # 자동 추가
uv run regulation synonym suggest "휴학" --context "대학" # 맥락 지정
uv run regulation synonym add 휴학 학업중단               # 수동 추가
uv run regulation synonym list                            # 전체 목록
uv run regulation synonym remove 휴학 학업중단            # 삭제
```

### 5.4 패치 검증

```bash
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer.py -v
uv run pytest tests/rag/unit/infrastructure/test_self_rag.py -v
uv run pytest tests/rag/unit/infrastructure/test_hyde.py -v
uv run pytest tests/rag/ -v --tb=short
```

---

## Phase 6: 재평가 및 반복 판단

### 6.1 단위 테스트 확인
```bash
uv run pytest tests/rag/ -v --tb=short
```

### 6.2 정적 평가 재실행
```bash
uv run python scripts/auto_evaluate.py --run
```

### 6.3 종료 조건

다음 중 하나 해당 시 Phase 7으로:
1. 모든 목표 달성
2. 2회 연속 동일 결과
3. 3회 사이클 완료
4. `architecture` 유형만 남음

종료 조건 미해당 시 **Phase 5로 반복**

---

## Phase 7: 완료 보고

### 7.1 세션 요약 생성

`data/output/test_report_strict_<날짜>.md` 생성:

1. **시작 상태**: 정적 통과율, 규정 수, RAG 설정
2. **RAG 컴포넌트 단위 테스트**: 각 컴포넌트별 결과
3. **동적 테스트 결과**: 페르소나, 난이도, 팩트체크, 성공률, 의도 추론 정확도
4. **멀티턴 테스트 결과**: 시나리오 수, 맥락 유지, Turn별 의도 추론
5. **답변 품질 점수**: 항목별 평균, 실패 유형 분포
6. **RAG 컴포넌트 기여도 종합**: 발동 횟수, 기여도, 개선 필요 컴포넌트
7. **적용된 개선**: 인텐트/동의어 추가, 코드 수정, 설정 변경
8. **최종 상태**: 최종 통과율, 남은 문제 및 원인
9. **다음 단계 권장사항**: 우선순위별 개선 항목

### 7.2 실패 쿼리 저장
```bash
# data/output/failed_queries_<날짜>.json에 저장
```

---

## 트러블슈팅

```bash
# LLM 연결 확인
curl -s $LLM_BASE_URL/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin))"

# 특정 테스트 디버깅
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer.py::test_specific -v -s

# RAG 컴포넌트별 테스트
uv run pytest tests/rag/unit/infrastructure/test_self_rag.py -v -s
uv run pytest tests/rag/unit/infrastructure/test_hyde.py -v -s
uv run pytest tests/rag/unit/infrastructure/test_query_expander.py -v -s
uv run pytest tests/rag/unit/infrastructure/test_fact_checker.py -v -s

# 변경사항 확인/되돌리기
git diff data/config/
git checkout -- data/config/intents.json

# 디버그 모드로 RAG 파이프라인 확인
uv run regulation --debug search "<쿼리>" -a -n 5

# 캐시 확인/초기화
cat data/cache/hyde/hyde_cache.json | python -m json.tool
rm data/cache/hyde/hyde_cache.json
rm -rf data/cache/query_expansion/

# RAG 설정 임시 변경 테스트
ENABLE_HYDE=false uv run regulation search "<쿼리>" -n 5
ENABLE_SELF_RAG=false uv run regulation search "<쿼리>" -n 5
ENABLE_QUERY_EXPANSION=false uv run regulation search "<쿼리>" -n 5
ENABLE_FACT_CHECK=false uv run regulation search "<쿼리>" -a -n 5

# Tool Calling 비활성화
uv run regulation search "<쿼리>" -a --no-tools
```

---

## 체크리스트

### 기본 검증
- [ ] Phase 0: 시스템 상태 확인
- [ ] Phase 0: 고급 RAG 설정 확인
- [ ] Phase 1: 정적 평가 완료

### RAG 컴포넌트 검증
- [ ] Phase 1.5: Self-RAG 검증
- [ ] Phase 1.5: HyDE 검증 (ON/OFF 비교)
- [ ] Phase 1.5: Corrective RAG 검증
- [ ] Phase 1.5: Hybrid Search 검증
- [ ] Phase 1.5: Reranker 검증 (ON/OFF 비교)
- [ ] Phase 1.5: Query Analyzer 검증
- [ ] Phase 1.5: Fact Check 검증
- [ ] Phase 1.5: Dynamic Query Expansion 검증
- [ ] Phase 1.5: 입력 검증/보안 테스트
- [ ] Phase 1.5: 컴포넌트 통합 테스트

### 동적 테스트
- [ ] Phase 2: 페르소나 선택 (3~5개)
- [ ] Phase 2: 난이도 분포 반영 (쉬움 30%, 중간 40%, 어려움 30%)
- [ ] Phase 2: 의도 추론 검증 (3단계 분석)
- [ ] Phase 2: 의도 충족 검증
- [ ] Phase 3: **모든 답변 팩트체크 완료**
- [ ] Phase 3: 일반론 답변 패턴 체크
- [ ] Phase 3: RAG 컴포넌트 기여도 분석
- [ ] Phase 3: 답변 품질 검토 (평균 ≥ 4.0)

### 멀티턴 테스트 (필수)
- [ ] Phase 4: 멀티턴 시나리오 (페르소나당 2개 × 5턴 이상)
- [ ] Phase 4: 각 Turn별 의도 추론 검증
- [ ] Phase 4: 맥락 유지 확인
- [ ] Phase 4: 멀티턴 점수 (평균 ≥ 4.0)
- [ ] Phase 4: 후속 질문 성공률 (≥ 85%)
- [ ] Phase 4: 의도 추론 정확도 (≥ 80%)

### 실패 분석
- [ ] Phase 4.5: 실패 케이스 5-Why 분석 (해당 시)
- [ ] Phase 4.5: RAG 컴포넌트별 원인 분석

### 개선 및 완료
- [ ] Phase 5: 개선 적용 (필요시)
- [ ] Phase 5: RAG 컴포넌트별 개선 적용
- [ ] Phase 6: 재평가 완료
- [ ] Phase 7: 보고서 생성 (RAG 기여도 포함)
