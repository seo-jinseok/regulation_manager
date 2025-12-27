import pytest

from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.entities import Chunk, ChunkLevel, Keyword, SearchResult


class FakeStore:
    def __init__(self, results):
        self._results = results

    def search(self, query, filter=None, top_k: int = 10):
        return self._results
    
    def get_all_documents(self):
        """Return empty list - hybrid searcher will be skipped."""
        return []


def make_chunk(text: str, keywords=None) -> Chunk:
    return Chunk(
        id="c1",
        rule_code="1-1-1",
        level=ChunkLevel.ARTICLE,
        title="",
        text=text,
        embedding_text=text,
        full_text="",
        parent_path=[],
        token_count=1,
        keywords=keywords or [],
        is_searchable=True,
    )


def test_keyword_bonus_applied():
    """키워드 보너스가 점수에 적용되는지 테스트 (reranker/hybrid 비활성화)"""
    chunk = make_chunk("내용", keywords=[Keyword(term="교원", weight=1.0)])
    store = FakeStore([SearchResult(chunk=chunk, score=0.4, rank=1)])
    # Reranker와 Hybrid Search를 비활성화하여 키워드 보너스만 테스트
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

    results = usecase.search("교원", top_k=1)

    # 키워드 보너스: 0.05 (keyword_bonus = min(0.3, 1.0 * 0.05))
    assert results[0].score == pytest.approx(0.45)

