## Plan: 고급 RAG 기능 전면 활성화

4가지 개선점(Self-RAG, HyDE, Corrective RAG 동적 임계값, BM25 형태소 분석)을 TDD 방식으로 구현하여 RAG 시스템의 검색 품질을 향상시킵니다.

### Steps

1. **설정 인프라 확장**: [config.py](src/rag/config.py)에 `enable_self_rag`, `enable_hyde`, `corrective_rag_thresholds`, `bm25_tokenize_mode` 설정 추가

2. **Corrective RAG 동적 임계값**: [retrieval_evaluator.py](src/rag/infrastructure/retrieval_evaluator.py)에 쿼리 유형별 임계값 딕셔너리 지원 추가, [search_usecase.py](src/rag/application/search_usecase.py)에서 쿼리 유형 전달

3. **HyDE 통합**: [search_usecase.py](src/rag/application/search_usecase.py)에서 `HyDERetriever` 호출 로직 추가, 모호한 쿼리(`싶어`, `싫어` 등) 자동 감지 및 가상 문서 생성

4. **Self-RAG 선택적 활성화**: [search_usecase.py](src/rag/application/search_usecase.py)에서 `SelfRAGPipeline` 통합, 복잡 쿼리(comparison, multi-hop)에만 적용

5. **BM25 형태소 분석 강화**: [hybrid_search.py](src/rag/infrastructure/hybrid_search.py)에 KoNLPy 기반 `_tokenize_konlpy()` 메서드 추가 (optional dependency)

6. **테스트 작성 및 검증**: `test_self_rag.py`, `test_hyde.py` 신규 생성, 기존 테스트 파일에 새 기능 테스트 추가

### Further Considerations

1. **KoNLPy 의존성**: pyproject.toml에 optional dependency로 추가할지, 완전 필수로 할지? (Option A: optional / Option B: required / Option C: 규칙 기반만 유지) : B

2. **Self-RAG LLM 비용**: 쿼리당 추가 LLM 호출 1-3회 발생. 모든 쿼리에 적용 vs 복잡 쿼리만 적용? (Option A: 전체 / Option B: complex만 / Option C: 사용자 opt-in) : A

3. **HyDE 캐시 전략**: 가상 문서 캐시를 영구 저장 vs 세션 단위 vs 비활성화? (Option A: 영구 / Option B: 세션 / Option C: 비활성화) : 영구
