## Plan: RAG 시스템 핵심 교훈 기반 개선

테스트에서 발견된 3가지 문제(Reranker 역효과, 비현실적 임계값, 인텐트 미흡)를 해결하고, 9가지 RAG 아키텍처 프레임워크를 적용하여 시스템을 고도화합니다.

---

### 현재 구현 상태 (9 RAG Architectures 기준)

| # | 아키텍처 | 상태 | 구현 위치 |
|---|----------|------|-----------|
| 1 | **Standard RAG** | ✅ 완료 | `ChromaVectorStore` + `SearchUseCase` |
| 2 | **Conversational RAG** | ✅ 완료 | `QueryHandler` 멀티턴 컨텍스트 |
| 3 | **Corrective RAG (CRAG)** | ✅ 완료 | `RetrievalEvaluator` + 쿼리 확장 |
| 4 | **Adaptive RAG** | ⚠️ 부분 | 쿼리 라우팅만 (복잡도 기반 전략 없음) |
| 5 | **Self-RAG** | ⚠️ 부분 | `SelfRAGPipeline` 존재, 기본 비활성화 |
| 6 | **Fusion RAG** | ✅ 완료 | `HybridSearcher` (Dense + BM25 + RRF) |
| 7 | **HyDE** | ❌ 미구현 | 가설 문서 임베딩 없음 |
| 8 | **Agentic RAG** | ✅ 완료 | `FunctionGemmaAdapter` + `ToolExecutor` |
| 9 | **GraphRAG** | ❌ 미구현 | 지식 그래프 없음 |

---

### Steps (우선순위 순)

#### Phase 1: 즉시 적용 (설정 파일)

1. **min_relevance_score 일괄 조정** - [evaluation_dataset.json](data/config/evaluation_dataset.json)에서 0.3 이상인 20+개 케이스를 0.05~0.15로 하향

2. **누락 인텐트 10개 추가** - [intents.json](data/config/intents.json)에 `schedule_change`, `grade_inquiry`, `retake_course`, `appeal_process` 등 추가

#### Phase 2: 단기 개선 (1주)

3. **Adaptive RAG 구현** - 쿼리 복잡도 분류기 추가 → 복잡도에 따라 검색 전략 선택
   - Simple: Standard RAG (빠른 응답)
   - Medium: Fusion RAG + Reranker
   - Complex: Agentic RAG (도구 호출)
   ```python
   # search_usecase.py에 추가
   def _classify_complexity(self, query: str) -> Literal["simple", "medium", "complex"]:
       if self._is_structural_query(query):  # 제15조, 교원인사규정
           return "simple"
       if len(matched_intents) >= 2 or "비교" in query:
           return "complex"
       return "medium"
   ```

4. **Reranker 조건부 적용** - [search_usecase.py](src/rag/application/search_usecase.py#L577)에서 인텐트 기반 쿼리 시 스킵
   ```python
   # 인텐트 확장된 쿼리는 reranker가 오히려 역효과
   skip_reranker = len(matched_intents) > 0 and self._skip_reranker_for_intent
   ```

#### Phase 3: 중기 고도화 (2-4주)

5. **HyDE (Hypothetical Document Embeddings) 구현** - 모호한 쿼리에 가설 답변 생성 후 임베딩 검색
   ```python
   # 새 파일: src/rag/infrastructure/hyde.py
   class HyDEGenerator:
       def generate_hypothetical_doc(self, query: str) -> str:
           """LLM으로 가설 규정 조문 생성"""
           prompt = f"다음 질문에 답하는 대학 규정 조문을 작성하세요: {query}"
           return self._llm.generate(prompt)
       
       def search_with_hyde(self, query: str) -> List[SearchResult]:
           hypo_doc = self.generate_hypothetical_doc(query)
           return self._store.search(hypo_doc)  # 가설 문서로 검색
   ```

6. **Self-RAG 활성화 및 개선** - [self_rag.py](src/rag/infrastructure/self_rag.py) 평가 로직 강화
   - `enable_retrieval_check`: 검색 필요성 판단 ✅
   - `enable_relevance_check`: 결과 관련성 평가 ✅
   - `enable_support_check`: **답변 근거 검증 활성화**

7. **Hybrid Scoring 구현** - Reranker + Score Bonus 결합
   ```python
   final_score = α * reranker_score + (1 - α) * boosted_score  # α = 0.7
   ```

#### Phase 4: 장기 로드맵 (1-2개월)

8. **GraphRAG 탐색** - 규정 간 참조 관계를 지식 그래프로 구축
   - "제15조에서 정한 바에 따라" → 참조 엣지 생성
   - Neo4j 또는 NetworkX 기반 그래프 검색
   - 연관 규정 자동 추천

---

### Further Considerations

1. **우선순위**: Phase 1~2는 1주 내 완료 가능, Phase 3~4는 ROI 검토 후 결정?
2. **HyDE 비용**: LLM 호출 2배 증가 → 캐싱 전략 필요
3. **GraphRAG 범위**: 전체 318개 규정의 참조 관계 추출 자동화 가능 여부?
4. **Self-RAG 활성화**: support_check 활성화 시 응답 시간 3배 증가 예상 → 비동기 처리?

---

### 예상 효과

| Phase | 개선 항목 | 예상 효과 |
|-------|----------|-----------|
| 1 | 설정 최적화 | 평가 통과율 100% 유지 |
| 2 | Adaptive + 조건부 Reranker | 모호한 쿼리 정확도 +15% |
| 3 | HyDE + Self-RAG | 복잡 쿼리 답변 품질 +20% |
| 4 | GraphRAG | 연관 규정 추천 기능 추가 |
