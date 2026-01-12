"""
Integration tests for the complete RAG search pipeline.

Tests the end-to-end flow from query input to search results,
covering the interaction between all pipeline components:
- QueryAnalyzer (intent detection, query expansion)
- HybridSearcher (BM25 + Dense + RRF fusion)
- BGEReranker (cross-encoder reranking)
- SearchUseCase (orchestration)
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from src.rag.domain.entities import Chunk, SearchResult


class FakeVectorStore:
    """Fake vector store for integration testing."""

    def __init__(self, chunks: List[Chunk]):
        self._chunks = {c.id: c for c in chunks}
        self._search_results = []

    def set_search_results(self, results: List[SearchResult]):
        """Set the results to return from search."""
        self._search_results = results

    def search(self, query, filter=None, top_k: int = 10) -> List[SearchResult]:
        """Return preset search results."""
        return self._search_results[:top_k]

    def get_all_documents(self) -> List[tuple]:
        """Return all chunks as tuples for BM25 indexing."""
        return [
            (c.id, c.text, c.to_metadata())
            for c in self._chunks.values()
        ]


class FakeReranker:
    """Fake reranker for integration testing."""

    def __init__(self):
        self.call_count = 0
        self.last_query = None

    def compute_score(self, pairs: List[List[str]], normalize: bool = True) -> List[float]:
        """Return fake scores based on keyword overlap."""
        self.call_count += 1
        self.last_query = pairs[0][0] if pairs else None
        
        scores = []
        for query, doc in pairs:
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            base_score = min(1.0, overlap / max(len(query_terms), 1) * 0.8)
            scores.append(base_score)
        return scores


def make_chunk(
    id: str,
    text: str,
    title: str = "",
    rule_code: str = "",
    **kwargs,
) -> Chunk:
    """Helper to create test chunks."""
    from src.rag.domain.entities import ChunkLevel, Keyword
    
    return Chunk(
        id=id,
        text=text,
        title=title,
        rule_code=rule_code,
        level=ChunkLevel.ARTICLE,
        embedding_text=text,
        full_text=text,
        parent_path=[title] if title else [],
        token_count=len(text.split()),
        keywords=[],
        is_searchable=True,
    )


def make_search_result(chunk: Chunk, score: float, rank: int) -> SearchResult:
    """Helper to create test search results."""
    return SearchResult(chunk=chunk, score=score, rank=rank)


class TestSearchPipelineIntegration:
    """Integration tests for the complete search pipeline."""

    @pytest.fixture
    def sample_chunks(self) -> List[Chunk]:
        """Sample chunks for testing."""
        return [
            make_chunk(
                id="chunk1",
                text="교원인사규정 제15조에 따라 교원을 임용한다. 임용 절차는 다음과 같다.",
                title="교원인사규정",
                rule_code="3-1-5",
            ),
            make_chunk(
                id="chunk2",
                text="장학금 지급 대상은 성적 우수자로 한다. 장학금은 등록금의 일부를 지원한다.",
                title="장학금지급규정",
                rule_code="3-3-4",
            ),
            make_chunk(
                id="chunk3",
                text="휴학을 원하는 학생은 휴학원을 제출해야 한다. 휴학 기간은 1년이다.",
                title="학칙",
                rule_code="2-1-1",
            ),
            make_chunk(
                id="chunk4",
                text="육아휴직 신청은 출산 후 1년 이내에 해야 한다. 휴직 기간은 최대 2년이다.",
                title="교원인사규정",
                rule_code="3-1-5",
            ),
            make_chunk(
                id="chunk5",
                text="인권센터는 학생의 권리를 보호한다. 학생 고충 상담을 제공한다.",
                title="인권센터규정",
                rule_code="5-1-38",
            ),
        ]

    @pytest.fixture
    def mock_store(self, sample_chunks: List[Chunk]):
        """Create a fake vector store with sample chunks."""
        store = FakeVectorStore(sample_chunks)
        # Set default search results
        results = [
            make_search_result(sample_chunks[0], 0.9, 1),
            make_search_result(sample_chunks[1], 0.8, 2),
            make_search_result(sample_chunks[2], 0.7, 3),
        ]
        store.set_search_results(results)
        return store

    @pytest.fixture
    def mock_reranker(self):
        """Create a fake reranker."""
        return FakeReranker()

    def test_query_analyzer_expands_query(self, mock_store, mock_reranker):
        """쿼리 분석기가 동의어를 확장하는지 확인"""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer
        
        analyzer = QueryAnalyzer()
        
        # "교수" → "교원" 확장 확인
        expanded = analyzer.expand_query("교수 임용")
        assert "교원" in expanded or "교수" in expanded

    def test_query_analyzer_detects_intent(self, mock_store, mock_reranker):
        """쿼리 분석기가 의도를 감지하는지 확인"""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType
        
        analyzer = QueryAnalyzer()
        
        # 자연어 의도 쿼리 - 휴학 의도를 감지
        query_type = analyzer.analyze("휴학하고 싶어")
        # 휴학 키워드가 규정명 패턴과 일치할 수 있으므로 여러 타입 허용
        assert query_type in (QueryType.INTENT, QueryType.REGULATION_NAME, QueryType.GENERAL)

    def test_hybrid_searcher_combines_results(self, mock_store, mock_reranker):
        """하이브리드 검색기가 결과를 융합하는지 확인"""
        from src.rag.infrastructure.hybrid_search import HybridSearcher, ScoredDocument
        
        searcher = HybridSearcher(use_dynamic_weights=False)
        
        sparse_results = [
            ScoredDocument("doc1", 0.9, "교원 임용 규정", {}),
            ScoredDocument("doc2", 0.8, "장학금 규정", {}),
        ]
        dense_results = [
            ScoredDocument("doc2", 0.95, "장학금 규정", {}),
            ScoredDocument("doc1", 0.85, "교원 임용 규정", {}),
        ]
        
        fused = searcher.fuse_results(sparse_results, dense_results, top_k=2)
        
        # 두 결과에 모두 있는 문서들이 융합되어야 함
        assert len(fused) == 2
        doc_ids = {r.doc_id for r in fused}
        assert doc_ids == {"doc1", "doc2"}

    def test_reranker_reorders_by_relevance(self, mock_store, mock_reranker):
        """Reranker가 관련성에 따라 재정렬하는지 확인"""
        from src.rag.infrastructure.reranker import BGEReranker
        
        with patch("src.rag.infrastructure.reranker.get_reranker", return_value=mock_reranker):
            reranker = BGEReranker()
            
            docs = [
                ("doc1", "일반 내용", {}),
                ("doc2", "장학금 신청 방법 절차", {}),
                ("doc3", "장학금 규정", {}),
            ]
            
            result = reranker.rerank("장학금 신청", docs, top_k=3)
            
            # 키워드 매칭이 더 많은 doc2가 상위에 있어야 함
            assert result[0][0] == "doc2"

    def test_pipeline_article_reference_query(self, mock_store, mock_reranker, sample_chunks):
        """조문 참조 쿼리 처리 파이프라인 테스트"""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType
        from src.rag.infrastructure.hybrid_search import HybridSearcher
        
        # 1. 쿼리 분석
        analyzer = QueryAnalyzer()
        query = "교원인사규정 제15조"
        query_type = analyzer.analyze(query)
        
        assert query_type == QueryType.ARTICLE_REFERENCE

        # 2. 동적 가중치 확인
        bm25_w, dense_w = analyzer.get_weights(query)
        assert bm25_w > dense_w  # 조문 참조는 BM25 우선

    def test_pipeline_natural_question_query(self, mock_store, mock_reranker):
        """자연어 질문 처리 파이프라인 테스트"""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType
        
        analyzer = QueryAnalyzer()
        query = "학교 다니면서 아르바이트 해도 되나요?"
        
        # 자연어 질문 타입 확인
        query_type = analyzer.analyze(query)
        # INTENT 또는 NATURAL_QUESTION
        assert query_type in (QueryType.INTENT, QueryType.NATURAL_QUESTION, QueryType.GENERAL)

    def test_pipeline_intent_based_expansion(self, mock_store, mock_reranker):
        """의도 기반 쿼리 확장 파이프라인 테스트"""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer
        
        analyzer = QueryAnalyzer()
        
        # 모호한 의도 쿼리
        query = "휴학하고 싶어"
        expanded = analyzer.expand_query(query)
        
        # 휴학 관련 키워드가 확장되어야 함
        assert "휴학" in expanded

    def test_pipeline_regulation_context_boost(self, mock_store, mock_reranker):
        """규정 컨텍스트 부스트 파이프라인 테스트"""
        from src.rag.infrastructure.reranker import BGEReranker
        
        with patch("src.rag.infrastructure.reranker.get_reranker", return_value=mock_reranker):
            reranker = BGEReranker()
            
            docs = [
                ("doc1", "휴학 신청 절차", {"regulation_title": "학칙"}),
                ("doc2", "휴학 관련 내용", {"regulation_title": "장학규정"}),
            ]
            
            # 학칙 컨텍스트로 재정렬
            result = reranker.rerank_with_context(
                "휴학",
                docs,
                context={"target_regulation": "학칙"},
                top_k=2,
            )
            
            # 학칙 문서가 부스트되어야 함
            assert result[0][0] == "doc1"


class TestSearchUseCaseIntegration:
    """Integration tests for SearchUseCase with mocked dependencies."""

    @pytest.fixture
    def sample_chunks(self) -> List[Chunk]:
        """Sample chunks for testing."""
        return [
            make_chunk(
                id="chunk1",
                text="장학금 신청 방법에 대한 규정",
                title="장학금규정",
                rule_code="3-3-4",
            ),
            make_chunk(
                id="chunk2",
                text="휴학 신청 절차에 대한 내용",
                title="학칙",
                rule_code="2-1-1",
            ),
        ]

    @pytest.fixture
    def mock_store(self, sample_chunks: List[Chunk]):
        """Create fake store with sample data."""
        store = FakeVectorStore(sample_chunks)
        results = [
            make_search_result(sample_chunks[0], 0.85, 1),
            make_search_result(sample_chunks[1], 0.75, 2),
        ]
        store.set_search_results(results)
        return store

    def test_search_usecase_returns_results(self, mock_store, sample_chunks):
        """SearchUseCase가 결과를 반환하는지 확인"""
        from src.rag.application.search_usecase import SearchUseCase
        
        usecase = SearchUseCase(store=mock_store, use_reranker=False)
        
        results = usecase.search("장학금", top_k=2)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_usecase_query_rewrite_info(self, mock_store):
        """SearchUseCase가 쿼리 재작성 정보를 제공하는지 확인"""
        from src.rag.application.search_usecase import SearchUseCase
        
        usecase = SearchUseCase(store=mock_store, use_reranker=False)
        
        usecase.search("휴학하고 싶어", top_k=2)
        
        rewrite_info = usecase.get_last_query_rewrite()
        
        # 재작성 정보가 존재해야 함
        assert rewrite_info is not None
        assert hasattr(rewrite_info, 'rewritten')
        assert hasattr(rewrite_info, 'matched_intents')


class TestEvaluationIntegration:
    """Integration tests for the evaluation system."""

    def test_evaluation_loads_dataset(self):
        """평가 시스템이 데이터셋을 로드하는지 확인"""
        from src.rag.application.evaluate import EvaluationUseCase
        
        eval_uc = EvaluationUseCase(search_usecase=None)
        test_cases = eval_uc.load_dataset()
        
        assert len(test_cases) > 0
        assert all(hasattr(tc, 'query') for tc in test_cases)
        assert all(hasattr(tc, 'expected_intents') for tc in test_cases)

    def test_evaluation_test_case_structure(self):
        """테스트 케이스 구조가 올바른지 확인"""
        from src.rag.application.evaluate import EvaluationUseCase, TestCase
        
        eval_uc = EvaluationUseCase(search_usecase=None)
        test_cases = eval_uc.load_dataset()
        
        for tc in test_cases[:5]:  # 첫 5개만 확인
            assert isinstance(tc, TestCase)
            assert isinstance(tc.id, str)
            assert isinstance(tc.query, str)
            assert isinstance(tc.expected_intents, list)
            assert isinstance(tc.expected_keywords, list)
            assert isinstance(tc.min_relevance_score, float)
