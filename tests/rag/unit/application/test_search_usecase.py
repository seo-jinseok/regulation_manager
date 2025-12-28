import pytest

from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.value_objects import SearchFilter
from src.rag.domain.entities import (
    Chunk,
    ChunkLevel,
    Keyword,
    RegulationStatus,
    SearchResult,
)


class FakeStore:
    def __init__(self, results):
        self._results = results

    def search(self, query, filter=None, top_k: int = 10):
        return self._results
    
    def get_all_documents(self):
        """Return empty list - hybrid searcher will be skipped."""
        return []


class FakeStoreCapture:
    def __init__(self):
        self.last_query = None
        self.last_filter = None
        self.last_top_k = None

    def search(self, query, filter=None, top_k: int = 10):
        self.last_query = query
        self.last_filter = filter
        self.last_top_k = top_k
        return []

    def get_all_documents(self):
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


class FakeLLMCapture:
    def __init__(self):
        self.last_user_message = None

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        self.last_user_message = user_message
        return "ok"


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


def test_search_coerces_non_string_query():
    store = FakeStoreCapture()
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

    usecase.search(["교원인사규정", "전문"], top_k=1)

    assert store.last_query is not None
    assert store.last_query.text == "교원인사규정 전문"


def test_search_rule_code_filters_by_rule_code():
    store = FakeStoreCapture()
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

    usecase.search("3-1-5", top_k=7)

    assert store.last_query is not None
    assert store.last_query.text == "규정"
    assert store.last_filter is not None
    assert store.last_filter.rule_codes == ["3-1-5"]
    assert store.last_top_k == 7


def test_ask_includes_history_and_uses_search_query(monkeypatch):
    chunk = make_chunk("본문")
    results = [SearchResult(chunk=chunk, score=0.4, rank=1)]
    store = FakeStore(results)
    llm = FakeLLMCapture()
    usecase = SearchUseCase(store, llm_client=llm, use_reranker=False, use_hybrid=False)

    captured = {}

    def fake_search(self, query_text, filter=None, top_k: int = 10, include_abolished: bool = False, audience_override=None):
        captured["query_text"] = query_text
        return results

    monkeypatch.setattr(SearchUseCase, "search", fake_search)

    usecase.ask(
        question="다른 부칙은?",
        search_query="교원인사규정 다른 부칙은?",
        history_text="사용자: 교원인사규정\n어시스턴트: 부칙을 확인했습니다.",
        top_k=1,
    )

    assert captured["query_text"] == "교원인사규정 다른 부칙은?"
    assert llm.last_user_message is not None
    assert "대화 기록" in llm.last_user_message
    assert "현재 질문: 다른 부칙은?" in llm.last_user_message


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


def test_hybrid_filters_abolished_sparse_results():
    """Hybrid 검색에서 BM25 결과도 폐지 규정 필터링됨."""
    documents = [
        (
            "doc_active",
            "휴학 절차 안내",
            {
                "rule_code": "1-1-1",
                "level": "article",
                "title": "휴학",
                "parent_path": "학생규정",
                "status": "active",
            },
        ),
        (
            "doc_abolished",
            "휴학 절차 안내",
            {
                "rule_code": "1-1-2",
                "level": "article",
                "title": "휴학",
                "parent_path": "학생규정",
                "status": "abolished",
            },
        ),
    ]
    store = FakeStoreWithDocs(results=[], documents=documents)
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=True)

    results = usecase.search("휴학", top_k=5, include_abolished=False)

    assert {r.chunk.id for r in results} == {"doc_active"}
    assert all(r.chunk.status == RegulationStatus.ACTIVE for r in results)


def test_confidence_uses_normalized_scores():
    """Reranker 점수(0~1)를 과도하게 포화시키지 않음."""
    store = FakeStore([])
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

    chunk = make_chunk("내용")
    results = [
        SearchResult(chunk=chunk, score=0.8, rank=1),
        SearchResult(chunk=chunk, score=0.6, rank=2),
        SearchResult(chunk=chunk, score=0.4, rank=3),
    ]

    confidence = usecase._compute_confidence(results)

    assert confidence == pytest.approx(0.72)


def test_hybrid_filters_by_rule_code_and_level():
    """Hybrid 검색에서 BM25 결과가 rule_code/level 필터를 만족해야 함."""
    documents = [
        (
            "doc_allowed",
            "휴학 절차 안내",
            {
                "rule_code": "A-1-1",
                "level": "article",
                "title": "휴학",
                "parent_path": "학생규정",
                "status": "active",
            },
        ),
        (
            "doc_wrong_level",
            "휴학 절차 안내",
            {
                "rule_code": "A-1-1",
                "level": "section",
                "title": "휴학",
                "parent_path": "학생규정",
                "status": "active",
            },
        ),
        (
            "doc_wrong_rule",
            "휴학 절차 안내",
            {
                "rule_code": "B-1-1",
                "level": "article",
                "title": "휴학",
                "parent_path": "학생규정",
                "status": "active",
            },
        ),
    ]
    store = FakeStoreWithDocs(results=[], documents=documents)
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=True)

    filter = SearchFilter(rule_codes=["A-1-1"], levels=[ChunkLevel.ARTICLE])
    results = usecase.search("휴학", filter=filter, top_k=10, include_abolished=True)

    assert {r.chunk.id for r in results} == {"doc_allowed"}
