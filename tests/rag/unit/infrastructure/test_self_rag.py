"""
Unit tests for Self-RAG integration in SearchUseCase.

Tests cover:
- Self-RAG activation based on config
- Retrieval necessity check
- Relevance filtering
- Integration with search pipeline
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional

from src.rag.config import reset_config


class FakeChunk:
    """Fake Chunk for testing."""
    
    def __init__(
        self,
        id: str,
        text: str,
        title: str = "",
        rule_code: str = "",
    ):
        self.id = id
        self.text = text
        self.title = title
        self.rule_code = rule_code
        self.keywords = []


class FakeSearchResult:
    """Fake SearchResult for testing."""
    
    def __init__(self, chunk: FakeChunk, score: float, rank: int = 1):
        self.chunk = chunk
        self.score = score
        self.rank = rank


def make_result(
    doc_id: str,
    text: str,
    score: float,
    title: str = "",
    rule_code: str = "",
    rank: int = 1,
) -> FakeSearchResult:
    """Helper to create fake search results."""
    chunk = FakeChunk(id=doc_id, text=text, title=title, rule_code=rule_code)
    return FakeSearchResult(chunk=chunk, score=score, rank=rank)


class FakeStore:
    """Fake Vector Store for testing."""
    
    def __init__(self, results: List[FakeSearchResult] = None):
        self._results = results or []
        self.search_calls = []
    
    def search(self, query, filter=None, top_k: int = 10):
        self.search_calls.append({"query": query, "filter": filter, "top_k": top_k})
        return self._results[:top_k]


class TestSelfRAGIntegration:
    """Test Self-RAG integration in SearchUseCase."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_search_usecase_has_self_rag_attributes(self):
        """SearchUseCase가 Self-RAG 관련 속성을 가져야 함."""
        from src.rag.application.search_usecase import SearchUseCase
        
        store = FakeStore([
            make_result("doc1", "휴직 신청", 0.8, rule_code="1-1-1"),
        ])
        
        usecase = SearchUseCase(
            store,
            use_reranker=False,
            use_hybrid=False,
        )
        
        # Self-RAG 관련 속성들이 있어야 함
        assert hasattr(usecase, "_self_rag_pipeline")
        assert hasattr(usecase, "_enable_self_rag")

    def test_self_rag_enabled_by_default(self):
        """Self-RAG는 기본적으로 활성화되어야 함."""
        from src.rag.config import RAGConfig
        
        config = RAGConfig()
        assert config.enable_self_rag is True


class TestSelfRAGPipeline:
    """Test SelfRAGPipeline functionality."""

    def test_should_retrieve_returns_true_for_factual_queries(self):
        """사실적 질문에는 검색이 필요함."""
        from src.rag.infrastructure.self_rag import SelfRAGPipeline
        
        # LLM 없이도 기본적으로 True 반환
        pipeline = SelfRAGPipeline(
            enable_retrieval_check=False,
        )
        
        assert pipeline.should_retrieve("교원 휴직 신청 절차는?") is True

    def test_filter_relevant_results_returns_all_when_disabled(self):
        """enable_relevance_check=False면 모든 결과 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGPipeline
        
        pipeline = SelfRAGPipeline(
            enable_relevance_check=False,
        )
        
        results = [
            make_result("doc1", "관련 내용", 0.9),
            make_result("doc2", "무관한 내용", 0.3),
        ]
        
        filtered = pipeline.filter_relevant_results("query", results)
        
        # 필터링 비활성화 시 모든 결과 반환
        assert len(filtered) == len(results)

    def test_evaluate_results_batch_skips_llm_for_high_scores(self):
        """점수가 높으면 LLM 호출 없이 결과 반환."""
        from src.rag.infrastructure.self_rag import SelfRAGPipeline
        
        pipeline = SelfRAGPipeline()
        
        results = [
            make_result("doc1", "관련 내용", 0.95),
            make_result("doc2", "또 다른 관련 내용", 0.85),
        ]
        
        is_relevant, filtered, confidence = pipeline.evaluate_results_batch(
            "query", results
        )
        
        # 높은 점수면 바로 관련성 있다고 판단
        assert is_relevant is True
        assert confidence > 0.8


class TestSelfRAGEvaluator:
    """Test SelfRAGEvaluator functionality."""

    def test_evaluator_needs_llm_for_retrieval_check(self):
        """검색 필요성 판단은 LLM이 필요함."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator
        
        evaluator = SelfRAGEvaluator(llm_client=None)
        
        # LLM 없으면 기본적으로 검색 필요하다고 판단
        result = evaluator.needs_retrieval("오늘 날씨 어때?")
        assert result is True  # Fallback to True when no LLM

    def test_evaluator_evaluate_relevance_without_llm(self):
        """LLM 없이도 관련성 평가 가능해야 함 (fallback)."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator
        
        evaluator = SelfRAGEvaluator(llm_client=None)
        
        results = [
            make_result("doc1", "휴직 신청 방법", 0.8),
        ]
        
        # LLM 없으면 모든 결과를 관련있다고 판단 (fallback)
        is_relevant, filtered = evaluator.evaluate_relevance("휴직", results)
        
        assert is_relevant is True
        assert len(filtered) == len(results)


class TestSearchUseCaseSelfRAGIntegration:
    """Test complete Self-RAG integration in SearchUseCase."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_ask_method_can_use_self_rag(self):
        """ask 메서드가 Self-RAG를 사용할 수 있어야 함."""
        from src.rag.application.search_usecase import SearchUseCase
        
        store = FakeStore([
            make_result("doc1", "휴직 신청은 인사과에 서류를 제출하면 됩니다.", 0.9, 
                       rule_code="1-1-1", title="휴직규정"),
        ])
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "휴직 신청은 인사과에 서류를 제출하면 됩니다."
        
        usecase = SearchUseCase(
            store,
            llm_client=mock_llm,
            use_reranker=False,
            use_hybrid=False,
        )
        usecase._corrective_rag_enabled = False
        
        # Self-RAG pipeline이 초기화되어 있어야 함
        assert hasattr(usecase, "_self_rag_pipeline")
