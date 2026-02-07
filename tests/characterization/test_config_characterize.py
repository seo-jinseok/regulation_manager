"""
Characterization tests for RAG Configuration component.

These tests capture the CURRENT BEHAVIOR of the configuration system
to ensure refactoring does not change observable behavior.
"""

import pytest

# Import the component under test
from src.rag.config import get_config, reset_config


@pytest.mark.characterization
class TestRAGConfigCharacterization:
    """Characterization tests for RAG configuration behavior."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_characterize_default_top_k_value(self):
        """
        CHARACTERIZATION TEST: Document current default_top_k value.

        Current behavior: default_top_k is 5
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.default_top_k == 5, (
            "CHARACTERIZATION: Current default_top_k is 5. "
            "If this changes after refactoring, verify the new value is intentional."
        )

    def test_characterize_cache_ttl_default(self):
        """
        CHARACTERIZATION TEST: Document current cache TTL default.

        Current behavior: cache_ttl_hours is 24
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.cache_ttl_hours == 24, (
            "CHARACTERIZATION: Current cache TTL is 24 hours. "
            "This documents the current default value."
        )

    def test_characterize_redis_max_connections_default(self):
        """
        CHARACTERIZATION TEST: Document current Redis max connections.

        Current behavior: redis_max_connections is 50
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.redis_max_connections == 50, (
            "CHARACTERIZATION: Current Redis max connections is 50. "
            "This documents the current default value."
        )

    def test_characterize_enable_self_rag_default(self):
        """
        CHARACTERIZATION TEST: Document current Self-RAG default.

        Current behavior: enable_self_rag is True
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.enable_self_rag is True, (
            "CHARACTERIZATION: Current Self-RAG is enabled by default."
        )

    def test_characterize_enable_hyde_default(self):
        """
        CHARACTERIZATION TEST: Document current HyDE default.

        Current behavior: enable_hyde is True
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.enable_hyde is True, (
            "CHARACTERIZATION: Current HyDE is enabled by default."
        )

    def test_characterize_use_reranker_default(self):
        """
        CHARACTERIZATION TEST: Document current reranker default.

        Current behavior: use_reranker is True
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.use_reranker is True, (
            "CHARACTERIZATION: Current reranker is enabled by default."
        )

    def test_characterize_bm25_tokenize_mode_default(self):
        """
        CHARACTERIZATION TEST: Document current BM25 tokenization mode.

        Current behavior: bm25_tokenize_mode is "kiwi"
        """
        # Arrange & Act
        config = get_config()

        # Assert - Document current behavior
        assert config.bm25_tokenize_mode == "kiwi", (
            "CHARACTERIZATION: Current BM25 tokenization mode is 'kiwi'."
        )

    def test_characterize_config_singleton_behavior(self):
        """
        CHARACTERIZATION TEST: Document config singleton behavior.

        Current behavior: get_config() returns same instance
        """
        # Arrange & Act
        config1 = get_config()
        config2 = get_config()

        # Assert - Document current behavior
        assert config1 is config2, (
            "CHARACTERIZATION: get_config() returns singleton instance."
        )

    def test_characterize_reset_config_behavior(self):
        """
        CHARACTERIZATION TEST: Document reset_config behavior.

        Current behavior: reset_config() creates new instance
        """
        # Arrange
        config1 = get_config()
        config1.default_top_k = 999  # Modify

        # Act
        reset_config()
        config2 = get_config()

        # Assert - Document current behavior
        assert config1 is not config2, (
            "CHARACTERIZATION: reset_config() creates new instance."
        )
        assert config2.default_top_k == 5, (
            "CHARACTERIZATION: New instance has default values."
        )
