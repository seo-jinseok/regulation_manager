"""
Unit tests for Korean Reranker integration (Cycle 5).

Tests cover:
- Korean model selection
- A/B testing framework
- Model fallback behavior
- Performance metrics tracking
"""

import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Tuple
from datetime import datetime

from src.rag.infrastructure.ab_test_framework import (
    ABTestManager,
    ABTestMetrics,
    ABTestSession,
    ABTestRepository,
    RerankerModelType,
    create_ab_manager,
)


class FakeReranker:
    """Fake reranker for testing without loading actual models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0
    
    def compute_score(
        self, pairs: List[List[str]], normalize: bool = True
    ) -> List[float]:
        """Return scores based on keyword matching."""
        self.call_count += 1
        scores = []
        for query, doc in pairs:
            # Korean-specific keyword matching
            query_korean = len([c for c in query if ord('가') <= ord(c) <= ord('힣')])
            doc_korean = len([c for c in doc if ord('가') <= ord(c) <= ord('힣')])
            
            # Higher score for Korean content
            korean_ratio = (query_korean + doc_korean) / (len(query) + len(doc) + 1)
            score = 0.3 + korean_ratio * 0.5  # Base score + Korean bonus
            scores.append(min(1.0, score))
        return scores


class TestABTestMetrics:
    """Test ABTestMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """메트릭 초기화 테스트"""
        metrics = ABTestMetrics(
            model_name="Dongjin-kr/kr-reranker",
            model_type=RerankerModelType.KOREAN,
        )
        
        assert metrics.model_name == "Dongjin-kr/kr-reranker"
        assert metrics.model_type == RerankerModelType.KOREAN
        assert metrics.total_queries == 0
        assert metrics.successful_queries == 0
    
    def test_metrics_to_dict(self):
        """메트릭 직렬화 테스트"""
        metrics = ABTestMetrics(
            model_name="test_model",
            model_type=RerankerModelType.MULTILINGUAL,
        )
        metrics.total_queries = 100
        metrics.successful_queries = 95
        metrics.failed_queries = 5
        metrics.avg_latency_ms = 45.2
        
        data = metrics.to_dict()
        
        assert data["model_name"] == "test_model"
        assert data["model_type"] == "multilingual"
        assert data["total_queries"] == 100
        assert data["successful_queries"] == 95
        assert data["avg_latency_ms"] == 45.2


class TestABTestSession:
    """Test ABTestSession functionality."""
    
    def test_session_creation(self):
        """세션 생성 테스트"""
        session = ABTestSession(
            session_id="test_001",
            test_ratio=0.5,
        )
        
        assert session.session_id == "test_001"
        assert session.test_ratio == 0.5
        assert len(session.model_metrics) == 0
    
    def test_get_metrics_creates_new(self):
        """새 메트릭 생성 테스트"""
        session = ABTestSession(session_id="test_002")
        
        metrics = session.get_metrics("model_a")
        
        assert isinstance(metrics, ABTestMetrics)
        assert metrics.model_name == "model_a"
        assert "model_a" in session.model_metrics
    
    def test_get_metrics_reuses_existing(self):
        """기존 메트릭 재사용 테스트"""
        session = ABTestSession(session_id="test_003")
        
        metrics1 = session.get_metrics("model_b")
        metrics2 = session.get_metrics("model_b")
        
        assert metrics1 is metrics2
    
    def test_record_query(self):
        """쿼리 기록 테스트"""
        session = ABTestSession(session_id="test_004")
        
        session.record_query("model_a", latency_ms=50.0, success=True)
        session.record_query("model_a", latency_ms=60.0, success=False)
        
        metrics = session.get_metrics("model_a")
        assert metrics.total_queries == 2
        assert metrics.successful_queries == 1
        assert metrics.failed_queries == 1
        assert metrics.avg_latency_ms == 55.0
    
    def test_record_query_with_relevance(self):
        """관련성 점수 기록 테스트"""
        session = ABTestSession(session_id="test_005")
        
        session.record_query("model_a", latency_ms=50.0, success=True, relevance_score=0.8)
        session.record_query("model_a", latency_ms=40.0, success=True, relevance_score=0.9)
        
        metrics = session.get_metrics("model_a")
        assert metrics.avg_relevance_score == pytest.approx(0.85)
    
    def test_session_to_dict(self):
        """세션 직렬화 테스트"""
        session = ABTestSession(session_id="test_006")
        session.record_query("model_a", latency_ms=50.0, success=True)
        
        data = session.to_dict()
        
        assert data["session_id"] == "test_006"
        assert "model_metrics" in data
        assert "model_a" in data["model_metrics"]


class TestABTestRepository:
    """Test ABTestRepository persistence."""
    
    @pytest.fixture
    def temp_repo(self, tmp_path):
        """임시 저장소 생성"""
        repo = ABTestRepository(storage_dir=str(tmp_path / "test_metrics"))
        yield repo
        # Cleanup is automatic with tmp_path
    
    def test_save_and_load_session(self, temp_repo):
        """세션 저장 및 로드 테스트"""
        session = ABTestSession(session_id="test_save_001")
        session.record_query("model_a", latency_ms=50.0, success=True)
        
        # Save
        filepath = temp_repo.save_session(session)
        assert filepath.endswith("ab_test_test_save_001.json")
        
        # Load
        loaded = temp_repo.load_session("test_save_001")
        assert loaded is not None
        assert loaded.session_id == "test_save_001"
        assert "model_a" in loaded.model_metrics
        assert loaded.model_metrics["model_a"].total_queries == 1
    
    def test_load_nonexistent_session(self, temp_repo):
        """존재하지 않는 세션 로드 테스트"""
        loaded = temp_repo.load_session("nonexistent")
        assert loaded is None
    
    def test_list_sessions(self, temp_repo):
        """세션 목록 조회 테스트"""
        # Create multiple sessions
        for i in range(3):
            session = ABTestSession(session_id=f"test_list_{i:03d}")
            temp_repo.save_session(session)
        
        sessions = temp_repo.list_sessions()
        assert len(sessions) == 3
        assert "test_list_000" in sessions
        assert "test_list_001" in sessions
        assert "test_list_002" in sessions


class TestABTestManager:
    """Test ABTestManager functionality."""
    
    def test_manager_initialization(self):
        """매니저 초기화 테스트"""
        manager = ABTestManager(
            control_model="BAAI/bge-reranker-v2-m3",
            test_models=["Dongjin-kr/kr-reranker"],
            test_ratio=0.5,
        )
        
        assert manager.control_model == "BAAI/bge-reranker-v2-m3"
        assert "Dongjin-kr/kr-reranker" in manager.test_models
        assert manager.test_ratio == 0.5
        assert manager.session_id is not None
    
    def test_select_model_uses_control_when_random_high(self):
        """랜덤 값이 높을 때 컨트롤 모델 선택 테스트"""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a", "test_b"],
            test_ratio=0.3,
        )
        
        # Mock random to return high value (control model)
        with patch("src.rag.infrastructure.ab_test_framework.random.random", return_value=0.5):
            selected = manager.select_model()
            assert selected == "control"
    
    def test_select_model_uses_test_when_random_low(self):
        """랜덤 값이 낮을 때 테스트 모델 선택 테스트"""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a", "test_b"],
            test_ratio=0.7,
        )
        
        # Mock random to return low value (test model)
        with patch("src.rag.infrastructure.ab_test_framework.random.random", return_value=0.5):
            selected = manager.select_model()
            assert selected in ["test_a", "test_b"]
    
    def test_record_result(self):
        """결과 기록 테스트"""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a"],
            test_ratio=0.5,
        )
        
        manager.record_result("control", latency_ms=50.0, success=True)
        manager.record_result("test_a", latency_ms=45.0, success=True)
        
        control_metrics = manager.session.get_metrics("control")
        test_metrics = manager.session.get_metrics("test_a")
        
        assert control_metrics.total_queries == 1
        assert test_metrics.total_queries == 1
    
    def test_get_summary(self):
        """요약 조회 테스트"""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a"],
            test_ratio=0.5,
        )
        
        # Record some results
        manager.record_result("control", latency_ms=100.0, success=True, relevance_score=0.7)
        manager.record_result("test_a", latency_ms=80.0, success=True, relevance_score=0.8)
        
        summary = manager.get_summary()
        
        assert "session_id" in summary
        assert "models" in summary
        assert "test_a_vs_control" in summary
        assert summary["test_a_vs_control"]["latency_improvement_percent"] == 20.0
        assert summary["test_a_vs_control"]["relevance_improvement_percent"] == pytest.approx(14.29, rel=1e-2)
    
    def test_recommendation_adopt(self):
        """채택 권장사항 테스트"""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a"],
            test_ratio=0.5,
        )
        
        manager.record_result("control", latency_ms=100.0, success=True, relevance_score=0.6)
        manager.record_result("test_a", latency_ms=80.0, success=True, relevance_score=0.75)
        
        summary = manager.get_summary()
        recommendation = summary["test_a_vs_control"]["recommendation"]
        
        assert "ADOPT" in recommendation or "CONSIDER" in recommendation
    
    def test_recommendation_reject(self):
        """거부 권장사항 테스트"""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a"],
            test_ratio=0.5,
        )
        
        manager.record_result("control", latency_ms=50.0, success=True, relevance_score=0.8)
        manager.record_result("test_a", latency_ms=60.0, success=True, relevance_score=0.6)
        
        summary = manager.get_summary()
        recommendation = summary["test_a_vs_control"]["recommendation"]
        
        assert "REJECT" in recommendation or "NEUTRAL" in recommendation


class TestCreateABManager:
    """Test convenience function for creating ABTestManager."""
    
    def test_create_with_defaults(self):
        """기본값으로 매니저 생성 테스트"""
        manager = create_ab_manager()
        
        assert manager.control_model == "BAAI/bge-reranker-v2-m3"
        assert "Dongjin-kr/kr-reranker" in manager.test_models
        assert manager.test_ratio == 0.5
    
    def test_create_with_custom_params(self):
        """사용자 정의 매개변수로 매니저 생성 테스트"""
        manager = create_ab_manager(
            control_model="custom_control",
            test_models=["custom_test"],
            test_ratio=0.7,
        )
        
        assert manager.control_model == "custom_control"
        assert "custom_test" in manager.test_models
        assert manager.test_ratio == 0.7


class TestExtendedReranker:
    """Test extended reranker with Korean model support."""
    
    @pytest.fixture
    def mock_load_model(self):
        """Mock model loading."""
        with patch("src.rag.infrastructure.reranker_extended.load_model") as mock:
            fake = FakeReranker("test_model")
            mock.return_value = fake
            yield mock
    
    def test_select_model_ab_test_strategy(self, mock_load_model):
        """A/B 테스트 전략 모델 선택 테스트"""
        with patch("src.rag.infrastructure.reranker_extended.get_ab_manager") as mock_ab:
            mock_manager = MagicMock()
            mock_manager.select_model.return_value = "Dongjin-kr/kr-reranker"
            mock_ab.return_value = mock_manager
            
            from src.rag.infrastructure.reranker_extended import select_model_for_query
            
            selected = select_model_for_query(
                "장학금 신청 방법",
                strategy="ab_test",
                korean_models=["Dongjin-kr/kr-reranker"],
            )
            
            assert selected == "Dongjin-kr/kr-reranker"
    
    def test_select_model_korean_only(self):
        """한국어 전용 전략 모델 선택 테스트"""
        with patch("src.rag.infrastructure.reranker_extended.get_config") as mock_config:
            config = MagicMock()
            config.reranker.primary_model = "BAAI/bge-reranker-v2-m3"
            config.reranker.korean_models = ["Dongjin-kr/kr-reranker"]
            mock_config.return_value = config
            
            from src.rag.infrastructure.reranker_extended import select_model_for_query
            
            selected = select_model_for_query(
                "장학금",
                strategy="korean_only",
                korean_models=["Dongjin-kr/kr-reranker"],
            )
            
            assert selected == "Dongjin-kr/kr-reranker"
    
    def test_select_model_multilingual_only(self):
        """다국어 전용 전략 모델 선택 테스트"""
        with patch("src.rag.infrastructure.reranker_extended.get_config") as mock_config:
            config = MagicMock()
            config.reranker.primary_model = "BAAI/bge-reranker-v2-m3"
            mock_config.return_value = config
            
            from src.rag.infrastructure.reranker_extended import select_model_for_query
            
            selected = select_model_for_query(
                "scholarship",
                strategy="multilingual_only",
            )
            
            assert selected == "BAAI/bge-reranker-v2-m3"


class TestKoreanRerankerClass:
    """Test KoreanReranker convenience class."""
    
    def test_korean_reranker_initialization(self):
        """한국어 Reranker 초기화 테스트"""
        # Import here to avoid import errors if extended module not available
        try:
            from src.rag.infrastructure.reranker import KoreanReranker
            
            reranker = KoreanReranker(use_ab_testing=True)
            
            # Check implementation exists (may be BGEReranker fallback)
            assert reranker._impl is not None
        except ImportError:
            pytest.skip("Extended reranker module not available")
    
    def test_korean_reranker_with_specific_model(self):
        """특정 모델 지정 한국어 Reranker 테스트"""
        try:
            from src.rag.infrastructure.reranker import KoreanReranker
            
            reranker = KoreanReranker(model_name="Dongjin-kr/kr-reranker")
            
            assert reranker._model_name == "Dongjin-kr/kr-reranker"
        except ImportError:
            pytest.skip("Extended reranker module not available")
    
    def test_korean_reranker_rerank_method(self):
        """한국어 Reranker rerank 메서드 테스트"""
        try:
            from src.rag.infrastructure.reranker import KoreanReranker
            
            reranker = KoreanReranker(model_name="test_model")
            
            # Mock the implementation
            with patch.object(reranker._impl, "rerank") as mock_rerank:
                mock_rerank.return_value = [
                    ("doc1", "content1", 0.9, {"meta": "data"}),
                ]
                
                docs = [("doc1", "장학금 신청", {"meta": "data"})]
                results = reranker.rerank("장학금", docs, top_k=5)
                
                assert len(results) == 1
                assert results[0][0] == "doc1"
                assert results[0][2] == 0.9
        except ImportError:
            pytest.skip("Extended reranker module not available")


class TestRerankerFallback:
    """Test reranker fallback behavior."""
    
    def test_fallback_on_korean_model_failure(self):
        """한국어 모델 실패 시 다국어 모델로 대체 테스트"""
        with patch("src.rag.infrastructure.reranker_extended.get_config") as mock_config:
            config = MagicMock()
            config.reranker.primary_model = "BAAI/bge-reranker-v2-m3"
            config.reranker.fallback_to_multilingual = True
            config.reranker.use_fp16 = True
            config.reranker.model_selection_strategy = "ab_test"
            config.reranker.korean_models = ["Dongjin-kr/kr-reranker"]
            mock_config.return_value = config
            
            with patch("src.rag.infrastructure.reranker_extended.get_ab_manager") as mock_ab:
                mock_manager = MagicMock()
                mock_ab.return_value = mock_manager
                
                with patch("src.rag.infrastructure.reranker_extended.load_model") as mock_load:
                    # First call (Korean model) fails
                    fake_korean = FakeReranker("Dongjin-kr/kr-reranker")
                    fake_korean.compute_score = MagicMock(side_effect=Exception("Model failed"))
                    
                    # Second call (multilingual) succeeds
                    fake_multilingual = FakeReranker("BAAI/bge-reranker-v2-m3")
                    
                    mock_load.side_effect = [fake_korean, fake_multilingual]
                    
                    from src.rag.infrastructure.reranker_extended import rerank
                    
                    docs = [("doc1", "장학금 신청", {})]
                    results = rerank("장학금", docs, model_name="Dongjin-kr/kr-reranker")
                    
                    # Should have fallen back to multilingual model
                    mock_load.assert_called()
                    assert len(results) > 0


class TestPerformanceSummary:
    """Test performance summary functionality."""
    
    def test_get_performance_summary(self):
        """성능 요약 조회 테스트"""
        try:
            from src.rag.infrastructure.reranker import get_model_performance_summary
            
            with patch("src.rag.infrastructure.reranker_extended.get_ab_manager") as mock_ab:
                mock_manager = MagicMock()
                mock_manager.get_summary.return_value = {
                    "session_id": "test_001",
                    "models": {
                        "control": {"total_queries": 100, "avg_latency_ms": 50.0},
                    },
                }
                mock_ab.return_value = mock_manager
                
                summary = get_model_performance_summary()
                
                assert "models" in summary
                assert "control" in summary["models"]
        except ImportError:
            pytest.skip("Extended reranker module not available")
