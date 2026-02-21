"""
Integration tests for Evasive Response Detection in SearchUseCase.

Tests the integration of EvasiveResponseDetector with the SearchUseCase
answer generation flow.

SPEC-RAG-QUALITY-010: Milestone 5 - Integration Tests
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.domain.evaluation.evasive_detector import (
    EvasiveDetectionResult,
    EvasiveResponseDetector,
)


class TestEvasiveDetectionIntegration:
    """Test EvasiveResponseDetector integration with SearchUseCase."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value="테스트 답변입니다.")
        return client

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.get_all_documents = MagicMock(return_value=[])
        store.search = MagicMock(return_value=[])
        return store

    def test_search_usecase_has_evasive_detector(self, mock_llm_client, mock_vector_store):
        """WHEN SearchUseCase is initialized, THEN should have evasive detector."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(
            store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        assert hasattr(usecase, '_evasive_detector')
        assert isinstance(usecase._evasive_detector, EvasiveResponseDetector)

    def test_search_usecase_evasive_metrics_methods(self, mock_llm_client, mock_vector_store):
        """WHEN SearchUseCase is initialized, THEN should have evasive metrics methods."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(
            store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        # Check metrics methods exist
        assert hasattr(usecase, 'get_evasive_detection_metrics')
        assert hasattr(usecase, 'reset_evasive_detection_metrics')
        assert hasattr(usecase, 'print_evasive_detection_metrics')

        # Initial metrics should be zero
        metrics = usecase.get_evasive_detection_metrics()
        assert metrics['evasive_detection_total'] == 0
        assert metrics['evasive_detected_count'] == 0
        assert metrics['evasive_regenerated_count'] == 0
        assert metrics['evasive_rate'] == 0.0

    def test_search_usecase_evasive_detection_config(self, mock_llm_client, mock_vector_store):
        """WHEN SearchUseCase is initialized, THEN should read evasive config."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(
            store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        # Check config attributes exist
        assert hasattr(usecase, '_enable_evasive_detection')
        assert hasattr(usecase, '_max_evasive_regeneration_attempts')

        # Default should be enabled with max 1 attempt
        assert usecase._enable_evasive_detection is True
        assert usecase._max_evasive_regeneration_attempts == 1

    def test_search_usecase_evasive_metrics_tracking(self, mock_llm_client, mock_vector_store):
        """WHEN evasive is detected, THEN metrics should be updated."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(
            store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        # Manually increment metrics to test tracking
        usecase._evasive_detection_total = 10
        usecase._evasive_detected_count = 2
        usecase._evasive_regenerated_count = 1

        metrics = usecase.get_evasive_detection_metrics()
        assert metrics['evasive_detection_total'] == 10
        assert metrics['evasive_detected_count'] == 2
        assert metrics['evasive_regenerated_count'] == 1
        assert metrics['evasive_rate'] == 0.2  # 2/10
        assert metrics['regeneration_success_rate'] == 0.5  # 1/2

        # Reset metrics
        usecase.reset_evasive_detection_metrics()
        metrics = usecase.get_evasive_detection_metrics()
        assert metrics['evasive_detection_total'] == 0
        assert metrics['evasive_detected_count'] == 0
        assert metrics['evasive_regenerated_count'] == 0

    def test_generate_answer_with_evasive_hint_exists(self, mock_llm_client, mock_vector_store):
        """WHEN SearchUseCase is initialized, THEN should have evasive hint generation method."""
        from src.rag.application.search_usecase import SearchUseCase

        usecase = SearchUseCase(
            store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        # Check method exists
        assert hasattr(usecase, '_generate_answer_with_evasive_hint')
        assert callable(usecase._generate_answer_with_evasive_hint)

    def test_full_evasive_detection_flow(self, mock_llm_client, mock_vector_store):
        """WHEN evasive response is detected, THEN full flow should work correctly."""
        from src.rag.application.search_usecase import SearchUseCase

        # Setup LLM to return evasive response first, then good response
        mock_llm_client.generate = MagicMock(
            side_effect=[
                "자세한 내용은 홈페이지를 참고하세요.",  # First: evasive
                "제10조에 따르면 휴학은 2학기까지 가능합니다.",  # Second: good
            ]
        )

        usecase = SearchUseCase(
            store=mock_vector_store,
            llm_client=mock_llm_client,
        )

        # Test detector directly
        detector = usecase._evasive_detector
        result = detector.detect(
            "자세한 내용은 홈페이지를 참고하세요.",
            ["제10조: 휴학은 2학기까지 가능합니다."]
        )

        assert result.is_evasive is True
        assert "homepage_deflection" in result.detected_patterns
        assert result.context_has_info is True
        assert detector.should_regenerate(result) is True
