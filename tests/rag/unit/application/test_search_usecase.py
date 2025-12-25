import pytest

from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.entities import Chunk, ChunkLevel, Keyword, SearchResult


class FakeStore:
    def __init__(self, results):
        self._results = results

    def search(self, query, filter=None, top_k: int = 10):
        return self._results


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
    chunk = make_chunk("내용", keywords=[Keyword(term="교원", weight=1.0)])
    store = FakeStore([SearchResult(chunk=chunk, score=0.4, rank=1)])
    usecase = SearchUseCase(store)

    results = usecase.search("교원", top_k=1)

    assert results[0].score == pytest.approx(0.45)
