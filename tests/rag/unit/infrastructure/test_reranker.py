"""
Unit tests for BGEReranker implementation.

Tests cover:
- Basic reranking functionality
- Score normalization
- Metadata preservation
- Edge cases (empty input, single document)
- Warmup functionality
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.rag.infrastructure.reranker import (
    BGEReranker,
    RerankedResult,
    clear_reranker,
    rerank,
    warmup_reranker,
)


class FakeReranker:
    """Fake reranker for testing without loading the actual model."""

    def __init__(self):
        self.call_count = 0

    def compute_score(
        self, pairs: List[List[str]], normalize: bool = True
    ) -> List[float]:
        """Return fake scores based on keyword matching."""
        self.call_count += 1
        scores = []
        for query, doc in pairs:
            # Simple keyword matching score
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = min(1.0, overlap / max(len(query_terms), 1) * 0.5 + 0.3)
            scores.append(score)
        return scores


@pytest.fixture
def mock_reranker():
    """Provide a mock reranker to avoid loading the actual model."""
    fake = FakeReranker()
    with patch("src.rag.infrastructure.reranker.get_reranker", return_value=fake):
        yield fake


class TestRerankerBasicFunctionality:
    """Test basic reranking functionality."""

    def test_rerank_empty_documents(self, mock_reranker):
        """빈 문서 목록 재정렬"""
        result = rerank("query", [])
        assert result == []

    def test_rerank_single_document(self, mock_reranker):
        """단일 문서 재정렬"""
        docs = [("doc1", "장학금 신청 방법", {"title": "장학규정"})]
        result = rerank("장학금", docs, top_k=1)

        assert len(result) == 1
        assert isinstance(result[0], RerankedResult)
        assert result[0].doc_id == "doc1"
        assert result[0].original_rank == 1

    def test_rerank_multiple_documents(self, mock_reranker):
        """여러 문서 재정렬"""
        docs = [
            ("doc1", "일반적인 내용", {}),
            ("doc2", "장학금 신청 절차와 방법", {}),
            ("doc3", "장학금 지급 규정", {}),
        ]
        result = rerank("장학금 신청", docs, top_k=3)

        assert len(result) == 3
        # 키워드 매칭이 더 많은 doc2가 상위에 있어야 함
        assert result[0].doc_id == "doc2"

    def test_rerank_respects_top_k(self, mock_reranker):
        """top_k 제한이 적용되는지 확인"""
        docs = [(f"doc{i}", f"content {i}", {}) for i in range(10)]
        result = rerank("query", docs, top_k=3)

        assert len(result) == 3

    def test_rerank_preserves_metadata(self, mock_reranker):
        """메타데이터가 보존되는지 확인"""
        metadata = {"title": "장학규정", "rule_code": "3-1-24"}
        docs = [("doc1", "장학금 규정 내용", metadata)]
        result = rerank("장학금", docs, top_k=1)

        assert result[0].metadata == metadata

    def test_rerank_scores_normalized(self, mock_reranker):
        """점수가 0-1 범위로 정규화되는지 확인"""
        docs = [
            ("doc1", "장학금 신청", {}),
            ("doc2", "휴학 규정", {}),
        ]
        result = rerank("장학금", docs, top_k=2)

        for r in result:
            assert 0.0 <= r.score <= 1.0


class TestBGERerankerClass:
    """Test BGEReranker class implementation."""

    def test_bge_reranker_interface(self, mock_reranker):
        """BGEReranker가 IReranker 인터페이스를 구현하는지 확인"""
        reranker = BGEReranker()

        # rerank 메서드가 tuple 리스트를 반환해야 함
        docs = [("doc1", "content", {"key": "value"})]
        result = reranker.rerank("query", docs, top_k=1)

        assert len(result) == 1
        assert len(result[0]) == 4  # (doc_id, content, score, metadata)
        doc_id, content, score, metadata = result[0]
        assert doc_id == "doc1"
        assert content == "content"
        assert isinstance(score, float)
        assert metadata == {"key": "value"}

    def test_bge_reranker_empty_input(self, mock_reranker):
        """빈 입력 처리"""
        reranker = BGEReranker()
        result = reranker.rerank("query", [], top_k=10)
        assert result == []


class TestRerankerScoreOrdering:
    """Test that reranking properly orders by relevance."""

    def test_rerank_orders_by_score_descending(self, mock_reranker):
        """점수가 높은 순으로 정렬되는지 확인"""
        docs = [
            ("doc1", "무관한 내용", {}),
            ("doc2", "장학금 신청 방법 절차", {}),
            ("doc3", "장학금", {}),
        ]
        result = rerank("장학금 신청 방법", docs, top_k=3)

        # 점수가 내림차순으로 정렬되어야 함
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_original_rank_preserved(self, mock_reranker):
        """원본 순위가 보존되는지 확인"""
        docs = [
            ("doc1", "첫번째", {}),
            ("doc2", "두번째", {}),
            ("doc3", "세번째", {}),
        ]
        result = rerank("query", docs, top_k=3)

        # original_rank가 1부터 시작
        original_ranks = {r.doc_id: r.original_rank for r in result}
        assert original_ranks["doc1"] == 1
        assert original_ranks["doc2"] == 2
        assert original_ranks["doc3"] == 3


class TestRerankerWithContext:
    """Test reranking with metadata context."""

    def test_rerank_with_context_boosts_matching_regulation(self, mock_reranker):
        """규정명 컨텍스트가 일치하면 점수가 부스트되는지 확인"""
        reranker = BGEReranker()
        docs = [
            ("doc1", "휴학 신청 절차", {"regulation_title": "학적규정"}),
            ("doc2", "휴학 관련 내용", {"regulation_title": "장학규정"}),
        ]

        # 컨텍스트 없이 재정렬
        result_no_context = reranker.rerank("학적규정 휴학", docs, top_k=2)

        # 컨텍스트와 함께 재정렬 - 학적규정 관련 문서 부스트
        result_with_context = reranker.rerank_with_context(
            "휴학 신청",
            docs,
            context={"target_regulation": "학적규정"},
            top_k=2
        )

        # 컨텍스트 사용 시 학적규정 문서가 상위에 있어야 함
        assert result_with_context[0][0] == "doc1"

    def test_rerank_with_context_empty_context(self, mock_reranker):
        """빈 컨텍스트는 일반 rerank와 동일해야 함"""
        reranker = BGEReranker()
        docs = [
            ("doc1", "장학금 신청", {"regulation_title": "장학규정"}),
        ]

        result_normal = reranker.rerank("장학금", docs, top_k=1)
        result_context = reranker.rerank_with_context("장학금", docs, context={}, top_k=1)

        assert result_normal[0][2] == result_context[0][2]  # 동일 점수

    def test_rerank_with_audience_context(self, mock_reranker):
        """대상(audience) 컨텍스트가 점수에 반영되는지 확인"""
        reranker = BGEReranker()
        docs = [
            ("doc1", "학생 휴학 절차", {"audience": "student", "regulation_title": "학칙"}),
            ("doc2", "교원 휴직 규정", {"audience": "faculty", "regulation_title": "교원인사규정"}),
        ]

        # 학생 대상 컨텍스트
        result = reranker.rerank_with_context(
            "휴학",
            docs,
            context={"target_audience": "student"},
            top_k=2
        )

        # 학생 대상 문서가 상위에 있어야 함
        assert result[0][0] == "doc1"

    def test_rerank_with_context_regulation_boost_factor(self, mock_reranker):
        """규정 일치 시 부스트 팩터가 적용되는지 확인"""
        reranker = BGEReranker()
        docs = [
            ("doc1", "내용", {"regulation_title": "장학규정"}),
            ("doc2", "내용", {"regulation_title": "학칙"}),
        ]

        result = reranker.rerank_with_context(
            "쿼리",
            docs,
            context={"target_regulation": "장학규정", "regulation_boost": 0.2},
            top_k=2
        )

        # 장학규정 문서가 부스트되어 상위에 있어야 함
        assert result[0][0] == "doc1"


class TestRerankerEdgeCases:
    """Test edge cases and error handling."""

    def test_rerank_very_long_document(self, mock_reranker):
        """매우 긴 문서 처리"""
        long_content = "장학금 " * 1000  # 매우 긴 문서
        docs = [("doc1", long_content, {})]

        result = rerank("장학금", docs, top_k=1)
        assert len(result) == 1

    def test_rerank_unicode_content(self, mock_reranker):
        """유니코드 문자 처리"""
        docs = [
            ("doc1", "교원인사규정 제15조 ① 항", {}),
            ("doc2", "학칙 제1조【목적】", {}),
        ]

        result = rerank("교원 제15조", docs, top_k=2)
        assert len(result) == 2

    def test_rerank_special_characters(self, mock_reranker):
        """특수 문자 처리"""
        docs = [
            ("doc1", "제1조(목적) 이 규정은...", {}),
            ("doc2", "별표 1. 장학금 지급 기준", {}),
        ]

        result = rerank("별표 1", docs, top_k=2)
        assert len(result) == 2


class TestRerankerWarmup:
    """Test reranker warmup functionality."""

    def test_warmup_reranker_sets_global_instance(self):
        """warmup_reranker가 전역 reranker 인스턴스를 설정하는지 확인"""
        import src.rag.infrastructure.reranker as reranker_module

        # 초기화 전 상태 확인을 위해 clear
        clear_reranker()
        assert reranker_module._reranker is None

        # warmup 호출 (실제 모델 로드를 피하기 위해 FlagEmbedding 모듈 mock)
        with patch.dict("sys.modules", {"FlagEmbedding": MagicMock()}):
            # get_reranker 내부의 import를 mock하기 위해 모듈 수준에서 처리
            mock_flag_module = MagicMock()
            mock_instance = MagicMock()
            mock_flag_module.FlagReranker.return_value = mock_instance

            with patch.dict("sys.modules", {"FlagEmbedding": mock_flag_module}):
                # 기존 캐시된 인스턴스를 지우고 새로 로드
                clear_reranker()
                warmup_reranker()

                # 전역 인스턴스가 설정되었는지 확인
                assert reranker_module._reranker is mock_instance

        # 정리
        clear_reranker()

    def test_clear_reranker_resets_global_instance(self):
        """clear_reranker가 전역 인스턴스를 리셋하는지 확인"""
        import src.rag.infrastructure.reranker as reranker_module

        # mock 인스턴스 설정
        reranker_module._reranker = MagicMock()
        assert reranker_module._reranker is not None

        # clear 호출
        clear_reranker()

        # 리셋되었는지 확인
        assert reranker_module._reranker is None

    def test_warmup_reranker_is_idempotent(self):
        """warmup_reranker가 여러 번 호출해도 안전한지 확인"""

        clear_reranker()

        # FlagEmbedding 모듈 mock
        mock_flag_module = MagicMock()
        mock_instance = MagicMock()
        mock_flag_module.FlagReranker.return_value = mock_instance

        with patch.dict("sys.modules", {"FlagEmbedding": mock_flag_module}):
            # 여러 번 호출
            warmup_reranker()
            warmup_reranker()
            warmup_reranker()

            # 첫 번째 호출에서만 생성되어야 함
            assert mock_flag_module.FlagReranker.call_count == 1

        clear_reranker()
