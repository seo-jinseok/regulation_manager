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


class FakeStoreWithDocs:
    def __init__(self, results, documents):
        self._results = results
        self._documents = documents

    def search(self, query, filter=None, top_k: int = 10):
        return self._results

    def get_all_documents(self):
        return self._documents


class FakeLLM:
    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        return "휴직 휴가 연구년"


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


def test_rerank_uses_rewritten_query(monkeypatch):
    """Reranker는 리라이팅된 쿼리를 사용해야 함."""
    from src.rag.infrastructure.reranker import RerankedResult

    chunk = make_chunk("내용")
    results = [SearchResult(chunk=chunk, score=0.4, rank=1)]
    documents = [(chunk.id, chunk.text, chunk.to_metadata())]
    store = FakeStoreWithDocs(results, documents)
    llm = FakeLLM()

    captured = {}

    def fake_rerank(query, documents, top_k=10):
        captured["query"] = query
        return [
            RerankedResult(
                doc_id=documents[0][0],
                content=documents[0][1],
                score=0.9,
                original_rank=1,
                metadata={},
            )
        ]

    monkeypatch.setattr("src.rag.infrastructure.reranker.rerank", fake_rerank)

    usecase = SearchUseCase(store, llm_client=llm, use_reranker=True, use_hybrid=True)
    usecase.search("나는 교수인데 학교에 가기 싫어", top_k=1)

    assert "휴직" in captured["query"]
    assert captured["query"] != "나는 교수인데 학교에 가기 싫어"
