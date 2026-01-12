# Plan: RAG 검색 파이프라인 품질 개선

통과율 84% → 90%+ 달성을 목표로, HybridSearcher/Reranker 파이프라인 개선 및 평가 체계 강화를 TDD 기반으로 진행합니다. Clean Architecture 원칙에 따라 도메인 인터페이스 정의 → 테스트 작성 → 구현 순서로 진행합니다.

## Steps

1. **[테스트 기반 구축]** [tests/rag/unit/infrastructure/](tests/rag/unit/infrastructure/)에 `test_hybrid_search.py`, `test_reranker.py`, `test_retrieval_evaluator.py` 신규 생성 → RRF 융합, Reranker 점수 계산, Corrective RAG 트리거 로직의 단위 테스트 작성 (RED)

2. **[평가 데이터셋 검토]** [evaluation_dataset.json](data/config/evaluation_dataset.json)의 8개 실패 케이스 정답 기준 분석 → `expected_rule_codes` 채우고 `min_relevance_score` 임계값 조정 (일부 0.05는 너무 관대함)

3. **[HybridSearcher 개선]** [hybrid_search.py](src/rag/infrastructure/hybrid_search.py)의 `_tokenize()` 메서드에 형태소 분석 옵션 추가 + RRF k값 동적 조정 파라미터 도입 → 테스트 통과 확인 (GREEN)

4. **[Reranker 메타데이터 활용]** [reranker.py](src/rag/infrastructure/reranker.py)에서 규정명/rule_code 메타데이터를 쿼리 컨텍스트로 포함하는 `rerank_with_context()` 메서드 추가 → [search_usecase.py](src/rag/application/search_usecase.py) 호출부 수정

5. **[Integration 테스트]** [tests/rag/integration/test_search_pipeline.py](tests/rag/integration/test_search_pipeline.py) 신규 생성 → 실제 ChromaDB 연동 E2E 파이프라인 테스트로 8개 실패 케이스 검증

6. **[최종 평가]** `uv run python scripts/auto_evaluate.py --run` 재실행하여 통과율 85%+ 달성 확인 → REFACTOR

## Further Considerations

1. **형태소 분석기 선택**: Kiwi (순수 Python, 설치 용이) vs Mecab (속도 빠름, 설치 복잡) → 선택 필요
2. **Reranker 후보군 확대**: 현재 `top_k*2` (20개) → `top_k*3` (30개)로 확대 시 성능 vs 속도 트레이드오프 검토
3. **음성 테스트 추가 여부**: "존재하지 않는 규정" 검색 시 빈 결과 반환 테스트 추가 권장
