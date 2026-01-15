# 쿼리 처리 파이프라인 상세 문서

이 문서는 사용자 쿼리가 입력되어 최종 답변이 출력되기까지의 모든 과정을 상세히 설명합니다.

---

## 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Query Processing Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Input → CLI/Web → QueryHandler → [Routing] → [Processing] → Response │
│                                            │                                │
│                    ┌───────────────────────┼───────────────────────┐        │
│                    │                       │                       │        │
│              Structural             Tool Calling           Traditional      │
│              (Overview,             (Agentic RAG)          (Search/Ask)     │
│               Article,                   │                       │         │
│               Chapter)                   │                       │         │
│                    │                     │                       │         │
│                    └───────────> Final Response <────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1단계: 초기화

### 1.1 CLI 초기화 (cli.py)

```python
# 1. 벡터 저장소 로드
store = ChromaVectorStore(persist_directory=args.db_path)

# 2. LLM 클라이언트 초기화
llm_client = LLMClientAdapter(
    provider=args.provider,
    model=args.model,
    base_url=args.base_url,
)

# 3. Tool Calling 사용 시 추가 초기화
if use_tool_calling:
    search_uc = SearchUseCase(store, use_reranker=use_reranker)
    analyzer = QueryAnalyzer()
    executor = ToolExecutor(search_usecase=search_uc, query_analyzer=analyzer)
    function_gemma_client = FunctionGemmaAdapter(
        tool_executor=executor,
        query_analyzer=analyzer,  # 인텐트 힌트 생성용
        api_mode=tool_mode,
        llm_client=llm_client,
    )

# 4. QueryHandler 생성
handler = QueryHandler(
    store=store,
    llm_client=llm_client,
    function_gemma_client=function_gemma_client,
    use_reranker=use_reranker,
)
```

### 1.2 Web UI 초기화 (gradio_app.py)

Web UI는 동일한 초기화 로직을 사용하며, Gradio 인터페이스와 연결됩니다.

---

## 2단계: 쿼리 라우팅 (QueryHandler)

### 2.1 전처리

```python
def process_query_stream(self, query, context, options):
    # NFC 정규화 (한글 NFD→NFC 변환)
    query = self._normalize_query(query)
```

**왜 NFC 정규화가 필요한가?**
- macOS 브라우저는 한글을 NFD(분해형)로 전송
- 정규식 패턴과 문자열 비교는 NFC(조합형) 기준
- 불일치 시 "전문" 같은 키워드 매칭 실패

### 2.2 쿼리 유형 판별

```python
# 1. 규정명만 (교원인사규정)
if self._is_overview_query(query):
    return self.get_regulation_overview(query)

# 2. 조항 요청 (교원인사규정 제8조)
if target_regulation and article_match:
    return self.get_article_view(target_regulation, article_no, context)

# 3. 장 요청 (학칙 제2장)
if target_regulation and chapter_match:
    return self.get_chapter_view(target_regulation, chapter_no, context)

# 4. 첨부 요청 (별표 1, 서식 2)
if attachment_request:
    return self.get_attachment_view(...)

# 5. 전문 요청 (학칙 전문)
if mode == "full_view":
    return self.get_full_view(query, context)

# 6. 일반 질문 → Tool Calling 또는 Traditional
```

### 2.3 패턴 정의 (patterns.py)

| 패턴 | 정규식 | 예시 |
|------|--------|------|
| `REGULATION_ONLY_PATTERN` | `^([\w가-힣]+(?:규정|규칙|세칙|지침|요령|학칙))$` | "교원인사규정" |
| `RULE_CODE_PATTERN` | `^\d+-\d+-\d+$` | "3-1-24" |
| `REGULATION_ARTICLE_PATTERN` | `([\w가-힣]+(?:규정|규칙|세칙))\s*제?\s*(\d+)\s*조` | "학칙 제15조" |

---

## 3단계: Tool Calling 경로 (Agentic RAG)

### 3.1 인텐트 분석 힌트 생성

```python
# function_gemma_adapter.py
def _build_analysis_context(self, query: str) -> str:
    hints = []
    
    # 인텐트 매칭
    intent_matches = self._query_analyzer._match_intents(query)
    if intent_matches:
        keywords = [kw for m in intent_matches[:2] for kw in m.keywords[:3]]
        hints.append(f"[의도 분석] 사용자의 진짜 의도는 '{', '.join(keywords)}' 관련일 수 있습니다.")
    
    # 청중 감지
    audience = self._query_analyzer.detect_audience(query)
    if audience != Audience.ALL:
        hints.append(f"[대상] {audience_map.get(audience)}")
    
    # 쿼리 확장
    expanded = self._query_analyzer.expand_query(query)
    if expanded != query:
        hints.append(f"[검색 키워드] {expanded}")
    
    return "\n".join(hints)
```

### 3.2 LLM 시스템 프롬프트

```
당신은 대학 규정 전문가입니다. 사용자의 질문에 답하기 위해 제공된 도구를 사용하세요.

[의도 분석] 사용자의 진짜 의도는 '휴직, 휴가, 연구년' 관련일 수 있습니다.
[대상] 교수/교원
[검색 키워드] 나는 교수인데 학교에 가기 싫어 휴직 휴가 연구년 안식년

작업 순서:
1. 위 분석 결과를 참고하여 search_regulations 도구로 관련 규정을 검색합니다.
   - [검색 키워드]가 제공된 경우, 해당 키워드를 query에 포함하세요.
2. 검색 결과를 바탕으로 generate_answer 도구를 호출하여 최종 답변을 생성합니다.
```

### 3.3 도구 정의 (tool_definitions.py)

| 도구 | 설명 | 파라미터 |
|------|------|----------|
| `search_regulations` | 규정 검색 | `query`, `top_k`, `audience` |
| `get_regulation_detail` | 특정 규정 상세 조회 | `regulation_name`, `article_number` |
| `generate_answer` | 최종 답변 생성 | `question`, `context` |

### 3.4 도구 실행 (tool_executor.py)

```python
def _handle_search_regulations(self, query, ...):
    # 인텐트 확장 적용 (핵심!)
    expanded_query = self._query_analyzer.expand_query(query)
    
    # 확장된 쿼리로 검색
    results = self._search_usecase.search(expanded_query, ...)
    
    return ToolResult(success=True, result=formatted_results)
```

---

## 4단계: Traditional 검색 경로

### 4.1 쿼리 재작성 (search_usecase.py)

```python
def _perform_query_rewriting(self, query_text, include_abolished):
    rewrite_info = self.hybrid_searcher._query_analyzer.rewrite_query_with_info(query_text)
    rewritten_query_text = rewrite_info.rewritten
    return query, rewritten_query_text
```

### 4.2 하이브리드 검색 (hybrid_search.py)

```python
def hybrid_search(self, query, ...):
    # 1. Dense 검색 (BGE-M3 임베딩)
    dense_results = self.store.search(query, ...)
    
    # 2. Sparse 검색 (BM25)
    sparse_results = self._bm25_search(query, ...)
    
    # 3. RRF 융합
    merged = self._reciprocal_rank_fusion(dense_results, sparse_results)
    
    return merged
```

### 4.3 Corrective RAG (retrieval_evaluator.py)

```python
def _apply_corrective_rag(self, query_text, results, ...):
    # 관련성 평가
    if not self._retrieval_evaluator.needs_correction(query_text, results):
        return results  # 충분히 관련있음
    
    # 쿼리 확장
    expanded_query = analyzer.expand_query(query_text)
    
    # 재검색
    corrected_results = self._search_general(expanded_query, ...)
    
    # 결과 병합
    return self._merge_results(results, corrected_results)
```

**RetrievalEvaluator 평가 기준:**
- Top 결과 점수 (50%): `results[0].score`
- 키워드 오버랩 (30%): 쿼리 키워드가 결과에 포함된 비율
- 결과 다양성 (20%): 서로 다른 규정에서 온 결과 수

### 4.4 재정렬 (reranker.py)

```python
def _apply_reranking(self, results, scoring_query_text, top_k):
    reranker = BGEReranker()
    reranked = reranker.rerank(scoring_query_text, documents, top_k)
    return reranked
```

### 4.5 LLM 답변 생성

```python
def ask_stream(self, question, options, context):
    # 검색 결과를 컨텍스트로 구성
    context_text = self._build_context(results)
    
    # 스트리밍 답변 생성
    for token in self.llm.generate_stream(
        system_prompt=REGULATION_QA_PROMPT,
        user_message=f"질문: {question}\n\n컨텍스트:\n{context_text}",
    ):
        yield {"type": "token", "content": token}
```

---

## 5단계: 후처리

### 5.1 후속 질문 제안

```python
def _enrich_with_suggestions(self, result, query):
    suggestions = get_followup_suggestions(
        query=query,
        regulation_title=result.data.get("regulation_title"),
        answer_text=result.content,
    )
    result.suggestions = suggestions
    return result
```

### 5.2 상태 업데이트

```python
# 다음 대화를 위해 상태 저장
state_update = {
    "last_regulation": regulation_title,
    "last_rule_code": rule_code,
    "last_query": query,
}
```

---

## 인텐트 확장 상세

### intents.json 구조

```json
{
  "학교에 가기 싫어": {
    "triggers": ["가기 싫어", "출근하기 싫어", "가고 싶지 않아"],
    "keywords": ["휴직", "휴가", "연구년", "안식년"],
    "audience": "faculty"
  },
  "그만두고 싶어": {
    "triggers": ["그만두", "퇴직", "사직"],
    "keywords": ["퇴직", "사직", "명예퇴직", "정년"],
    "audience": "all"
  }
}
```

### 매칭 로직 (query_analyzer.py)

```python
def _match_intents(self, query: str) -> List[IntentMatch]:
    matches = []
    for intent_name, rule in self._intent_rules.items():
        for trigger in rule.triggers:
            if trigger in query:
                matches.append(IntentMatch(
                    intent=intent_name,
                    trigger=trigger,
                    keywords=rule.keywords,
                    confidence=0.9,
                ))
    return sorted(matches, key=lambda m: -m.confidence)
```

---

## Self-RAG

Self-RAG는 LLM이 검색 필요성과 결과 품질을 자체 평가하는 고급 RAG 기법입니다.

### 활성화 설정

Self-RAG는 **기본적으로 활성화**되어 있습니다. 비활성화하려면:

```bash
# 환경 변수로 비활성화
ENABLE_SELF_RAG=false uv run regulation

# 또는 .env 파일에 설정
ENABLE_SELF_RAG=false
```

### 파이프라인 통합

```python
# search_usecase.py
def ask(self, question, ...):
    # 1. Self-RAG: 검색 필요성 평가
    if self._enable_self_rag:
        self._ensure_self_rag()
        if not self._self_rag_evaluator.needs_retrieval(question):
            # 검색 없이 직접 답변 (간단한 인사말 등)
            return self._generate_direct_answer(question)
    
    # 2. 검색 수행
    results = self.search(search_query or question, ...)
    
    # 3. Self-RAG: 결과 관련성 필터링
    if self._enable_self_rag and results:
        results = self._apply_self_rag_relevance_filter(question, results)
    
    # 4. 답변 생성
    return self._generate_answer(question, results)
```

### 평가 프롬프트

**검색 필요성 평가**:
```
다음 질문에 답하기 위해 외부 문서 검색이 필요한지 판단하세요.

질문: {query}

답변 형식: [RETRIEVE_YES] 또는 [RETRIEVE_NO]
```

**결과 관련성 평가**:
```
다음 문서가 질문에 답변하는 데 관련이 있는지 평가하세요.

질문: {query}
문서: {context}

답변 형식: [RELEVANT] 또는 [IRRELEVANT]
```

---

## HyDE (Hypothetical Document Embeddings)

HyDE는 모호한 쿼리에 대해 가상의 규정 문서를 생성한 후, 그 문서의 임베딩으로 검색하는 기법입니다.

### 활성화 설정

HyDE는 **기본적으로 활성화**되어 있습니다. 비활성화하려면:

```bash
ENABLE_HYDE=false uv run regulation
```

### 자동 감지 로직

```python
# search_usecase.py
VAGUE_PATTERNS = [
    r"싶어|싫어",      # 의도 표현
    r"뭐가|어떤|어떻게",  # 불명확 질문
    r"알려|설명",      # 정보 요청
]

def _should_use_hyde(self, query: str) -> bool:
    # 모호한 표현이 포함된 경우 HyDE 적용
    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, query):
            return True
    return False
```

### 파이프라인 통합

```python
def _search_general(self, query, ...):
    # 1. HyDE 적용 여부 확인
    if self._enable_hyde and self._should_use_hyde(query.text):
        self._ensure_hyde()
        hyde_result = self._apply_hyde(query.text)
        if hyde_result:
            # 가상 문서로 검색
            query = Query(text=hyde_result.hypothetical_doc, ...)
    
    # 2. 하이브리드 검색 수행
    results = self.hybrid_searcher.hybrid_search(query, ...)
    
    return results
```

### 가상 문서 생성 프롬프트

```
당신은 대학 규정 전문가입니다. 
사용자의 질문에 답하는 대학 규정 조문을 작성하세요.

작성 규칙:
1. 실제 대학 규정처럼 형식적인 문체로 작성
2. 관련 키워드와 용어를 포함
3. 100-200자 내외로 간결하게 작성

예시:
질문: "학교 안 가고 싶어"
답변: 교직원의 휴직은 다음 각 호의 사유에 해당하는 경우 신청할 수 있다...
```

### 캐시 전략

HyDE 결과는 영구 캐시에 저장됩니다:

```python
# 캐시 위치: data/cache/hyde/hyde_cache.json
{
    "학교에 가기 싫어": {
        "hypothetical_doc": "교직원의 휴직은 다음 각 호의...",
        "created_at": "2024-01-15T10:30:00"
    }
}
```

환경 변수로 캐시 제어:
```bash
HYDE_CACHE_DIR=data/cache/hyde  # 캐시 디렉토리
HYDE_CACHE_ENABLED=true         # 캐시 활성화 (기본: true)
```

---

## Corrective RAG 동적 임계값

쿼리 복잡도에 따라 관련성 평가 임계값이 자동 조정됩니다.

### 임계값 설정

```python
# config.py
corrective_rag_thresholds = {
    "simple": 0.3,   # 단순 키워드 검색
    "medium": 0.4,   # 일반 질문 (기본값)
    "complex": 0.5,  # 비교, 다단계 질문
}
```

### 복잡도 판단 로직

```python
def _determine_complexity(self, query: str) -> str:
    # 비교 표현: "차이점", "vs", "비교"
    if re.search(r"차이|비교|vs|versus", query):
        return "complex"
    
    # 다단계 질문: "그리고", "또한", "추가로"
    if re.search(r"그리고|또한|추가로|동시에", query):
        return "complex"
    
    # 단순 키워드 (3단어 이하)
    if len(query.split()) <= 3:
        return "simple"
    
    return "medium"
```

---

## 디버깅 가이드

### CLI 디버그 모드

```bash
uv run regulation --debug
```

### 주요 로그 포인트

| 위치 | 로그 내용 |
|------|----------|
| `FunctionGemmaAdapter.__init__` | `tool_executor`, `api_mode` 확인 |
| `_build_analysis_context` | 인텐트 힌트 생성 결과 |
| `ToolExecutor.execute` | 도구 호출 및 결과 |
| `RetrievalEvaluator.evaluate` | 관련성 점수 |
| `_apply_corrective_rag` | 재검색 트리거 여부 |
| `SelfRAGEvaluator.needs_retrieval` | 검색 필요성 판단 |
| `HyDEGenerator.generate` | 가상 문서 생성 |

### 일반적인 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| 인텐트 확장 미적용 | `QueryAnalyzer`가 `FunctionGemmaAdapter`에 전달되지 않음 | CLI 초기화 확인 |
| LLM이 엉뚱한 검색 | 시스템 프롬프트에 힌트 미포함 | `_build_analysis_context` 확인 |
| "전문" 쿼리 실패 | NFD/NFC 정규화 누락 | `_normalize_query` 확인 |
| Corrective RAG 미작동 | `_corrective_rag_enabled = False` | 설정 확인 |
| Self-RAG 비활성화됨 | `ENABLE_SELF_RAG=false` 환경 변수 | `.env` 확인 |
| HyDE 캐시 미사용 | `HYDE_CACHE_ENABLED=false` | 설정 확인 |
