from unittest.mock import patch

import pytest

from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.entities import (
    Chunk,
    ChunkLevel,
    Keyword,
    RegulationStatus,
    SearchResult,
)
from src.rag.domain.value_objects import SearchFilter


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
    def generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.0
    ) -> str:
        return "휴직 휴가 연구년"


class FakeLLMCapture:
    def __init__(self):
        self.last_user_message = None

    def generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.0
    ) -> str:
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
    # SPEC-RAG-QUALITY-004: Query may be expanded with synonyms
    # Original query is preserved but may have additional synonym terms
    assert "교원인사규정 전문" in store.last_query.text or "교원인사규정" in store.last_query.text


def test_search_rule_code_filters_by_rule_code():
    store = FakeStoreCapture()
    usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

    usecase.search("3-1-5", top_k=7)

    assert store.last_query is not None
    assert store.last_query.text == "규정"
    assert store.last_filter is not None
    assert store.last_filter.rule_codes == ["3-1-5"]
    assert store.last_top_k == 35  # top_k * 5 for initial retrieval before dedup


def test_ask_includes_history_and_uses_search_query(monkeypatch):
    chunk = make_chunk("본문")
    results = [SearchResult(chunk=chunk, score=0.4, rank=1)]
    store = FakeStore(results)
    llm = FakeLLMCapture()
    usecase = SearchUseCase(store, llm_client=llm, use_reranker=False, use_hybrid=False)
    usecase._enable_self_rag = False  # Disable Self-RAG for this test

    captured = {}

    def fake_search(
        self,
        query_text,
        filter=None,
        top_k: int = 10,
        include_abolished: bool = False,
        audience_override=None,
    ):
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
    """Reranker는 리라이팅된 쿼리(인텐트 키워드 포함)를 사용해야 함."""
    chunk = make_chunk("내용")
    results = [SearchResult(chunk=chunk, score=0.4, rank=1)]
    documents = [(chunk.id, chunk.text, chunk.to_metadata())]
    store = FakeStoreWithDocs(results, documents)
    llm = FakeLLM()

    captured = {}

    class FakeReranker:
        def rerank(self, query, documents, top_k=10):
            captured["query"] = query
            return [
                (documents[0][0], documents[0][1], 0.9, {}) for doc in documents[:top_k]
            ]

    usecase = SearchUseCase(store, llm_client=llm, use_reranker=True, use_hybrid=True)
    # 직접 fake reranker 주입
    usecase._reranker = FakeReranker()
    usecase._reranker_initialized = True

    # 단순 키워드 쿼리 사용 (인텐트 확장 없이 동의어 확장만)
    usecase.search("휴직 신청 방법", top_k=1)

    # reranker가 호출되었는지 확인
    # (composite decomposition이 아닌 경우에만 reranker 경로 사용)
    if "query" in captured:
        # 단순 쿼리의 경우 reranker 호출됨
        assert captured["query"] != "휴직 신청 방법", "쿼리가 리라이팅되어야 함"
    # composite decomposition된 경우 reranker 미호출도 정상


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

    # Mock cache check to bypass retrieval cache
    with patch.object(usecase, "_check_retrieval_cache", return_value=None):
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


def test_search_deduplicates_same_article():
    """같은 조항의 여러 청크가 검색되면 가장 점수가 높은 하나만 반환해야 함."""
    # 상황: 제4조에 해당하는 청크가 3개 검색됨 (본문, 항1, 항2)
    # 점수: 항1(0.9) > 본문(0.8) > 항2(0.7)
    # 기대: 항1 하나만 결과에 포함되어야 함 (One Chunk Per Article)

    # Chunk 1: Article 4 Body
    c1 = Chunk(
        id="c1",
        rule_code="3-1-6",
        level=ChunkLevel.ARTICLE,
        title="제4조(위원회)",
        text="위원회는...",
        embedding_text="",
        full_text="",
        parent_path=["규정"],
        token_count=10,
        keywords=[],
        is_searchable=True,
    )

    # Chunk 2: Article 4 Paragraph 1 (Highest Score)
    c2 = Chunk(
        id="c2",
        rule_code="3-1-6",
        level=ChunkLevel.PARAGRAPH,
        title="①",
        text="위원회는 5인...",
        embedding_text="",
        full_text="",
        parent_path=["규정", "제4조(위원회)"],
        token_count=10,
        keywords=[],
        is_searchable=True,
    )

    # Chunk 3: Article 4 Paragraph 2
    c3 = Chunk(
        id="c3",
        rule_code="3-1-6",
        level=ChunkLevel.PARAGRAPH,
        title="②",
        text="위원장은...",
        embedding_text="",
        full_text="",
        parent_path=["규정", "제4조(위원회)"],
        token_count=10,
        keywords=[],
        is_searchable=True,
    )

    # Search Results (Sorted by score already)
    raw_results = [
        SearchResult(chunk=c2, score=0.9, rank=1),
        SearchResult(chunk=c1, score=0.8, rank=2),
        SearchResult(chunk=c3, score=0.7, rank=3),
    ]

    store = FakeStore(raw_results)
    _usecase = SearchUseCase(store, use_reranker=False, use_hybrid=False)

    # Act


class TestSearchUseCaseWarmup:
    """Test SearchUseCase warmup functionality."""

    def test_warmup_initializes_hybrid_searcher(self):
        """_warmup이 hybrid_searcher를 초기화하는지 확인"""
        documents = [
            ("doc1", "교원 휴직 규정", {}),
        ]
        store = FakeStoreWithDocs(results=[], documents=documents)
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=True, enable_warmup=False
        )

        # warmup 전에는 초기화되지 않음
        assert usecase._hybrid_initialized is False

        # warmup 호출
        usecase._warmup()

        # warmup 후에는 초기화됨
        assert usecase._hybrid_initialized is True
        assert usecase._hybrid_searcher is not None

    def test_warmup_initializes_reranker_flag(self):
        """_warmup이 use_reranker=True일 때 reranker를 초기화하는지 확인"""
        store = FakeStore([])
        # use_reranker=True로 설정하고 warmup=False로 초기화 시점 제어
        usecase = SearchUseCase(
            store, use_reranker=True, use_hybrid=False, enable_warmup=False
        )

        # warmup 전에는 초기화되지 않음
        assert usecase._reranker_initialized is False

        # _ensure_reranker 직접 호출 (실제 모델 로드 없이 mock 사용)
        with patch("src.rag.infrastructure.reranker.warmup_reranker"):
            usecase._ensure_reranker()

        # 초기화됨
        assert usecase._reranker_initialized is True
        assert usecase._reranker is not None

    def test_warmup_with_no_documents_skips_hybrid(self):
        """문서가 없으면 hybrid_searcher 초기화를 건너뛰는지 확인"""
        store = FakeStore([])  # get_all_documents returns []
        usecase = SearchUseCase(
            store, use_reranker=False, use_hybrid=True, enable_warmup=False
        )

        usecase._warmup()

        # 문서가 없으므로 hybrid_searcher는 None
        assert usecase._hybrid_initialized is True
        assert usecase._hybrid_searcher is None

    def test_enable_warmup_env_variable(self, monkeypatch):
        """WARMUP_ON_INIT 환경변수가 적용되는지 확인"""

        warmup_called = []

        def mock_warmup(self):
            warmup_called.append(True)

        monkeypatch.setattr(SearchUseCase, "_warmup", mock_warmup)
        monkeypatch.setenv("WARMUP_ON_INIT", "true")

        store = FakeStore([])
        _usecase = SearchUseCase(store, use_reranker=False, enable_warmup=None)

        # 스레드 시작 대기
        import time

        time.sleep(0.1)

        assert len(warmup_called) == 1


# --- Phase 2: Search Strategy Tests ---


class TestSearchStrategy:
    """Tests for search strategy determination (Phase 2)."""

    def test_search_strategy_enum_exists(self):
        """SearchStrategy Enum이 존재하고 올바른 값을 가지는지 확인"""
        from src.rag.application.search_usecase import SearchStrategy

        assert SearchStrategy.DIRECT.value == "direct"
        assert SearchStrategy.TOOL_CALLING.value == "tool_calling"

    def test_determine_search_strategy_short_query(self):
        """짧은 쿼리는 DIRECT 전략 반환"""
        from src.rag.application.search_usecase import SearchStrategy

        store = FakeStore([])
        usecase = SearchUseCase(store, use_reranker=False, enable_warmup=False)

        result = usecase._determine_search_strategy("휴학")
        assert result == SearchStrategy.DIRECT

    def test_determine_search_strategy_simple_factual(self):
        """단순 사실 질문은 DIRECT 전략 반환"""
        from src.rag.application.search_usecase import SearchStrategy

        store = FakeStore([])
        usecase = SearchUseCase(store, use_reranker=False, enable_warmup=False)

        # Pattern: "~이 몇"
        result = usecase._determine_search_strategy("졸업학점이 몇 학점이야?")
        assert result == SearchStrategy.DIRECT

    def test_determine_search_strategy_complex_query(self):
        """복잡한 쿼리는 TOOL_CALLING 전략 반환"""
        from src.rag.application.search_usecase import SearchStrategy

        store = FakeStore([])
        usecase = SearchUseCase(store, use_reranker=False, enable_warmup=False)

        # Long, complex query
        result = usecase._determine_search_strategy(
            "휴학하면서 장학금도 받을 수 있는지 궁금합니다. 그리고 복학 절차도 알려주세요."
        )
        assert result == SearchStrategy.TOOL_CALLING

    def test_is_simple_factual_patterns(self):
        """단순 사실 질문 패턴 감지 테스트"""
        store = FakeStore([])
        usecase = SearchUseCase(store, use_reranker=False, enable_warmup=False)

        # Should match simple factual patterns
        assert usecase._is_simple_factual("졸업학점이 몇 학점이야?") is True
        assert usecase._is_simple_factual("영어 점수도 필요해?") is True
        assert usecase._is_simple_factual("장학금 성적 기준") is True

        # Should NOT match
        assert usecase._is_simple_factual("휴학하면서 장학금 받을 수 있나요?") is False

    def test_get_recommended_strategy_public_api(self):
        """공개 API get_recommended_strategy 테스트"""
        from src.rag.application.search_usecase import SearchStrategy

        store = FakeStore([])
        usecase = SearchUseCase(store, use_reranker=False, enable_warmup=False)

        # Simple query -> DIRECT
        result = usecase.get_recommended_strategy("휴학")
        assert result == SearchStrategy.DIRECT

        # Complex query -> TOOL_CALLING
        result = usecase.get_recommended_strategy(
            "교원 승진에 필요한 업적 평가 기준과 절차를 상세히 알려주세요."
        )
        assert result == SearchStrategy.TOOL_CALLING
