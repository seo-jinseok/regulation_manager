"""
Tests for SPEC-RAG-QUALITY-009 Milestone 1: Enable Faithfulness Validation by Default.

Tests that:
1. RAGConfig has faithfulness validation settings with correct defaults
2. SearchUseCase reads and uses these settings
3. Default path uses _generate_answer_with_validation when enabled
4. Logging shows validation metrics
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.config import RAGConfig, SearchConfig, reset_config, get_config
from src.rag.application.search_usecase import SearchUseCase
from src.rag.domain.entities import SearchResult, Chunk, ChunkLevel


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["This is a test response."]
        self.call_count = 0

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockVectorStore:
    """Mock vector store for testing."""

    def search(self, query, filter=None, top_k: int = 5):
        return []

    def get_all_documents(self):
        return []


def create_test_search_result(text: str, score: float = 0.8) -> SearchResult:
    """Create a test SearchResult with chunk."""
    chunk = Chunk(
        id="test-1",
        rule_code="TEST001",
        level=ChunkLevel.ARTICLE,
        title="Test Article",
        text=text,
        embedding_text=text,
        full_text=text,
        parent_path=["Test Regulation"],
        token_count=100,
        keywords=[],
        is_searchable=True,
        article_number="1",
    )
    return SearchResult(chunk=chunk, score=score, rank=1)


class TestRAGConfigFaithfulnessSettings:
    """Tests for RAGConfig faithfulness validation settings."""

    def test_default_use_faithfulness_validation_is_true(self):
        """Test that use_faithfulness_validation defaults to True."""
        reset_config()
        config = RAGConfig()
        assert config.use_faithfulness_validation is True

    def test_default_faithfulness_threshold_is_0_6(self):
        """Test that faithfulness_threshold defaults to 0.6."""
        reset_config()
        config = RAGConfig()
        assert config.faithfulness_threshold == 0.6

    def test_default_max_regeneration_attempts_is_2(self):
        """Test that max_regeneration_attempts defaults to 2."""
        reset_config()
        config = RAGConfig()
        assert config.max_regeneration_attempts == 2

    def test_settings_can_be_overridden_via_env(self, monkeypatch):
        """Test that settings can be overridden via environment variables."""
        monkeypatch.setenv("RAG_USE_FAITHFULNESS_VALIDATION", "false")
        monkeypatch.setenv("RAG_FAITHFULNESS_THRESHOLD", "0.8")
        monkeypatch.setenv("RAG_MAX_REGENERATION_ATTEMPTS", "3")

        reset_config()
        config = RAGConfig()

        assert config.use_faithfulness_validation is False
        assert config.faithfulness_threshold == 0.8
        assert config.max_regeneration_attempts == 3


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        assert config.use_faithfulness_validation is True
        assert config.faithfulness_threshold == 0.6
        assert config.max_regeneration_attempts == 2

    def test_search_config_from_rag_config(self):
        """Test creating SearchConfig from RAGConfig."""
        rag_config = RAGConfig(
            use_faithfulness_validation=True,
            faithfulness_threshold=0.7,
            max_regeneration_attempts=3,
        )
        search_config = SearchConfig.from_rag_config(rag_config)

        assert search_config.use_faithfulness_validation is True
        assert search_config.faithfulness_threshold == 0.7
        assert search_config.max_regeneration_attempts == 3

    def test_search_config_validation_threshold_range(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="faithfulness_threshold"):
            SearchConfig(faithfulness_threshold=1.5)

        with pytest.raises(ValueError, match="faithfulness_threshold"):
            SearchConfig(faithfulness_threshold=-0.1)

    def test_search_config_validation_attempts(self):
        """Test that invalid max_regeneration_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_regeneration_attempts"):
            SearchConfig(max_regeneration_attempts=-1)

    def test_search_config_to_metadata(self):
        """Test exporting SearchConfig to metadata dict."""
        config = SearchConfig(
            use_faithfulness_validation=True,
            faithfulness_threshold=0.7,
            max_regeneration_attempts=3,
        )
        metadata = config.to_metadata()

        assert metadata["use_faithfulness_validation"] is True
        assert metadata["faithfulness_threshold"] == 0.7
        assert metadata["max_regeneration_attempts"] == 3


class TestSearchUseCaseFaithfulnessValidation:
    """Tests for SearchUseCase faithfulness validation integration."""

    def test_search_usecase_reads_faithfulness_settings_from_config(self):
        """Test that SearchUseCase reads faithfulness settings from RAGConfig."""
        reset_config()
        store = MockVectorStore()

        usecase = SearchUseCase(store=store)

        assert usecase._use_faithfulness_validation is True
        assert usecase._faithfulness_threshold == 0.6
        assert usecase._max_regeneration_attempts == 2

    def test_ask_uses_faithfulness_validation_when_enabled(self):
        """Test that ask() uses _generate_answer_with_validation when enabled."""
        store = MockVectorStore()
        llm = MockLLMClient(
            responses=["제10조에 따르면 휴학 기간은 2학기까지 가능합니다."]
        )

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        # Mock search to return test results
        test_results = [
            create_test_search_result(
                "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
                score=0.9,
            )
        ]

        with patch.object(usecase, "search", return_value=test_results):
            with patch.object(usecase, "_compute_confidence", return_value=0.9):
                # Spy on _generate_answer_with_validation
                with patch.object(
                    usecase,
                    "_generate_answer_with_validation",
                    wraps=usecase._generate_answer_with_validation,
                ) as spy:
                    answer = usecase.ask("휴학 기간이 어떻게 되나요?")

                    # Verify _generate_answer_with_validation was called
                    spy.assert_called_once()

    def test_ask_falls_back_to_fact_check_when_validation_disabled(self):
        """Test that ask() uses _generate_with_fact_check when validation disabled."""
        reset_config()
        config = RAGConfig(use_faithfulness_validation=False)
        store = MockVectorStore()
        llm = MockLLMClient(
            responses=["제10조에 따르면 휴학 기간은 2학기까지 가능합니다."]
        )

        usecase = SearchUseCase(store=store, llm_client=llm)
        usecase._use_faithfulness_validation = False  # Explicitly disable

        test_results = [
            create_test_search_result(
                "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
                score=0.9,
            )
        ]

        with patch.object(usecase, "search", return_value=test_results):
            with patch.object(usecase, "_compute_confidence", return_value=0.9):
                # Spy on _generate_with_fact_check
                with patch.object(
                    usecase,
                    "_generate_with_fact_check",
                    wraps=usecase._generate_with_fact_check,
                ) as spy:
                    answer = usecase.ask("휴학 기간이 어떻게 되나요?")

                    # Verify _generate_with_fact_check was called
                    spy.assert_called_once()

    def test_validation_metadata_logged_on_success(self, caplog):
        """Test that validation metadata is logged on successful validation."""
        import logging

        store = MockVectorStore()
        llm = MockLLMClient(
            responses=["제10조에 따르면 휴학 기간은 2학기까지 가능합니다."]
        )

        usecase = SearchUseCase(store=store, llm_client=llm)

        test_results = [
            create_test_search_result(
                "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
                score=0.9,
            )
        ]

        with caplog.at_level(logging.INFO):
            with patch.object(usecase, "search", return_value=test_results):
                with patch.object(usecase, "_compute_confidence", return_value=0.9):
                    answer = usecase.ask("휴학 기간이 어떻게 되나요?")

        # Check that faithfulness validation log appears
        log_messages = [record.message for record in caplog.records]
        validation_logs = [msg for msg in log_messages if "Faithfulness validation" in msg]
        assert len(validation_logs) > 0, "Faithfulness validation log should appear"

    def test_fallback_returned_when_all_attempts_fail(self):
        """Test that fallback response is returned when all regeneration attempts fail."""
        store = MockVectorStore()
        # All responses have ungrounded claims
        llm = MockLLMClient(
            responses=[
                "휴학 기간은 5년입니다.",
                "휴학 기간은 10년입니다.",
                "휴학 기간은 3년입니다.",
            ]
        )

        usecase = SearchUseCase(store=store, llm_client=llm)

        test_results = [
            create_test_search_result(
                "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
                score=0.9,
            )
        ]

        with patch.object(usecase, "search", return_value=test_results):
            with patch.object(usecase, "_compute_confidence", return_value=0.9):
                answer = usecase.ask("휴학 기간이 어떻게 되나요?")

                # Should return fallback message
                assert "죄송" in answer.text or "찾을 수 없" in answer.text or "문의" in answer.text
                # Confidence should reflect faithfulness score
                assert answer.confidence < 0.6
