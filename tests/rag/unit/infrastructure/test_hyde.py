"""
Unit tests for HyDE integration in SearchUseCase.

Tests cover:
- HyDE activation based on config
- Automatic detection of vague queries
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


class FakeLLMClient:
    """Fake LLM client for HyDE generation."""
    
    def __init__(self, response: str = "가상 문서 내용"):
        self._response = response
    
    def generate(self, **kwargs):
        return self._response


class TestHyDEIntegration:
    """Test HyDE integration in SearchUseCase."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_hyde_enabled_for_vague_queries(self):
        """모호한 쿼리에서 HyDE가 활성화되어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator
        
        generator = HyDEGenerator(enable_cache=False)
        
        # 모호한 쿼리 패턴들
        vague_queries = [
            "학교 가기 싫어",
            "쉬고 싶어",
            "어떻게 해야 하나요",
            "가능한가요?",
        ]
        
        for query in vague_queries:
            assert generator.should_use_hyde(query, complexity="medium"), \
                f"'{query}'에서 HyDE가 활성화되어야 함"

    def test_hyde_skipped_for_structural_queries(self):
        """구조적 쿼리(조문 번호 등)에서는 HyDE를 건너뛰어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator
        
        generator = HyDEGenerator(enable_cache=False)
        
        # 구조적 쿼리 패턴들
        structural_queries = [
            "교원인사규정 제8조",
            "학칙",
            "3-1-24",
        ]
        
        for query in structural_queries:
            # simple 복잡도에서는 HyDE 비활성화
            assert not generator.should_use_hyde(query, complexity="simple"), \
                f"'{query}'에서 HyDE가 비활성화되어야 함"

    def test_hyde_cache_uses_config_settings(self):
        """HyDE 캐시가 config 설정을 사용해야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator
        
        generator = HyDEGenerator(
            cache_dir="custom/cache/dir",
            enable_cache=True,
        )
        
        assert generator._enable_cache is True


class TestHyDESearchFlow:
    """Test the complete HyDE search flow."""

    def test_search_with_hyde_merges_results(self):
        """HyDE 검색이 원본 쿼리 결과와 병합되어야 함."""
        from src.rag.infrastructure.hyde import HyDEGenerator, HyDESearcher, HyDEResult
        from src.rag.domain.value_objects import Query
        
        # Mock HyDE generator
        mock_generator = MagicMock(spec=HyDEGenerator)
        mock_generator.generate_hypothetical_doc.return_value = HyDEResult(
            original_query="휴직",
            hypothetical_doc="교원이 휴직을 신청하려면 인사규정에 따라...",
            from_cache=False,
        )
        
        # Fake store that returns SearchResult-like objects
        mock_store = MagicMock()
        mock_store.search.return_value = [
            make_result("doc1", "휴직 규정", 0.9, rule_code="1-1-1"),
            make_result("doc2", "휴직 절차", 0.8, rule_code="1-1-2"),
        ]
        
        searcher = HyDESearcher(mock_generator, mock_store)
        
        # Execute HyDE search
        merged_results = searcher.search_with_hyde("휴직", top_k=5)
        
        # HyDE generator가 호출되었어야 함
        mock_generator.generate_hypothetical_doc.assert_called_once_with("휴직")
        
        # Store가 두 번 검색되었어야 함 (HyDE + original)
        assert mock_store.search.call_count == 2


class TestSearchUseCaseHyDEIntegration:
    """Test SearchUseCase with HyDE integrated."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_search_usecase_has_hyde_attributes(self):
        """SearchUseCase가 HyDE 관련 속성을 가져야 함."""
        from src.rag.application.search_usecase import SearchUseCase
        
        store = FakeStore([
            make_result("doc1", "휴직 신청", 0.8, rule_code="1-1-1"),
        ])
        
        usecase = SearchUseCase(
            store,
            use_reranker=False,
            use_hybrid=False,
        )
        
        # HyDE 관련 속성들이 있어야 함
        assert hasattr(usecase, "_hyde_generator")
        assert hasattr(usecase, "_enable_hyde")
