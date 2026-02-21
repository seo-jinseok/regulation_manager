"""
Tests for IntentHandler and ClarificationGenerator (SPEC-RAG-QUALITY-010 Milestone 4).

TDD Approach:
1. RED: Write failing tests first
2. GREEN: Implement to make tests pass
3. REFACTOR: Clean up if needed
"""

import pytest

from src.rag.application.intent_classifier import IntentCategory
from src.rag.application.intent_handler import (
    ClarificationGenerator,
    IntentHandler,
    IntentSearchConfig,
)


class TestIntentSearchConfig:
    """Tests for IntentSearchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntentSearchConfig()

        assert config.top_k == 5
        assert config.procedure_boost == 1.0
        assert config.eligibility_boost == 1.0
        assert config.deadline_boost == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = IntentSearchConfig(
            top_k=10,
            procedure_boost=1.5,
            eligibility_boost=1.3,
            deadline_boost=1.4,
        )

        assert config.top_k == 10
        assert config.procedure_boost == 1.5
        assert config.eligibility_boost == 1.3
        assert config.deadline_boost == 1.4

    def test_config_immutability(self):
        """Test that config is a dataclass with frozen=False (mutable for testing)."""
        config = IntentSearchConfig()
        # Should be mutable by default
        config.top_k = 8
        assert config.top_k == 8


class TestIntentHandler:
    """Tests for IntentHandler class."""

    @pytest.fixture
    def handler(self):
        """Create IntentHandler instance for testing."""
        return IntentHandler()

    def test_get_procedure_search_config(self, handler):
        """Test search config for PROCEDURE intent."""
        config = handler.get_search_config(IntentCategory.PROCEDURE)

        assert config.top_k == 8
        assert config.procedure_boost == 1.5
        assert config.eligibility_boost == 1.0
        assert config.deadline_boost == 1.0

    def test_get_eligibility_search_config(self, handler):
        """Test search config for ELIGIBILITY intent."""
        config = handler.get_search_config(IntentCategory.ELIGIBILITY)

        assert config.top_k == 6
        assert config.procedure_boost == 1.0
        assert config.eligibility_boost == 1.3
        assert config.deadline_boost == 1.0

    def test_get_deadline_search_config(self, handler):
        """Test search config for DEADLINE intent."""
        config = handler.get_search_config(IntentCategory.DEADLINE)

        assert config.top_k == 5
        assert config.procedure_boost == 1.0
        assert config.eligibility_boost == 1.0
        assert config.deadline_boost == 1.4

    def test_get_general_search_config(self, handler):
        """Test search config for GENERAL intent."""
        config = handler.get_search_config(IntentCategory.GENERAL)

        assert config.top_k == 5
        assert config.procedure_boost == 1.0
        assert config.eligibility_boost == 1.0
        assert config.deadline_boost == 1.0

    def test_should_generate_clarification_low_confidence(self, handler):
        """Test clarification generation for low confidence."""
        # Confidence below 0.5 should trigger clarification
        assert handler.should_generate_clarification(0.3) is True
        assert handler.should_generate_clarification(0.49) is True
        assert handler.should_generate_clarification(0.0) is True

    def test_should_not_generate_clarification_high_confidence(self, handler):
        """Test no clarification for high confidence."""
        # Confidence >= 0.5 should not trigger clarification
        assert handler.should_generate_clarification(0.5) is False
        assert handler.should_generate_clarification(0.7) is False
        assert handler.should_generate_clarification(1.0) is False

    def test_get_clarification_generator(self, handler):
        """Test getting clarification generator instance."""
        generator = handler.get_clarification_generator()

        assert generator is not None
        assert isinstance(generator, ClarificationGenerator)

    def test_get_clarification_generator_returns_same_instance(self, handler):
        """Test that get_clarification_generator returns consistent instance."""
        generator1 = handler.get_clarification_generator()
        generator2 = handler.get_clarification_generator()

        # Should return same instance (singleton pattern for efficiency)
        assert generator1 is generator2


class TestClarificationGenerator:
    """Tests for ClarificationGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create ClarificationGenerator instance for testing."""
        return ClarificationGenerator()

    def test_generate_procedure_clarification(self, generator):
        """Test clarification generation for PROCEDURE intent."""
        result = generator.generate(
            query="장학금 받고 싶어요",
            intent=IntentCategory.PROCEDURE,
            confidence=0.3,
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should ask about procedure specifics
        assert any(
            keyword in result.lower()
            for keyword in ["신청", "절차", "방법", "어떻게"]
        )

    def test_generate_eligibility_clarification(self, generator):
        """Test clarification generation for ELIGIBILITY intent."""
        result = generator.generate(
            query="장학금",
            intent=IntentCategory.ELIGIBILITY,
            confidence=0.3,
        )

        assert result is not None
        assert isinstance(result, str)
        # Should ask about eligibility criteria
        assert any(
            keyword in result.lower()
            for keyword in ["자격", "조건", "대상", "누구"]
        )

    def test_generate_deadline_clarification(self, generator):
        """Test clarification generation for DEADLINE intent."""
        result = generator.generate(
            query="신청",
            intent=IntentCategory.DEADLINE,
            confidence=0.3,
        )

        assert result is not None
        assert isinstance(result, str)
        # Should ask about deadline/timing
        assert any(
            keyword in result.lower()
            for keyword in ["언제", "기간", "기한", "날짜"]
        )

    def test_no_clarification_for_high_confidence(self, generator):
        """Test that no clarification is generated for high confidence."""
        result = generator.generate(
            query="장학금 신청 방법이 궁금해요",
            intent=IntentCategory.PROCEDURE,
            confidence=0.8,
        )

        # High confidence should return None
        assert result is None

    def test_no_clarification_for_general_intent(self, generator):
        """Test that no clarification is generated for GENERAL intent."""
        result = generator.generate(
            query="규정",
            intent=IntentCategory.GENERAL,
            confidence=0.3,
        )

        # GENERAL intent should return None
        assert result is None

    def test_clarification_includes_original_query_context(self, generator):
        """Test that clarification references original query context."""
        query = "장학금"
        result = generator.generate(
            query=query,
            intent=IntentCategory.PROCEDURE,
            confidence=0.3,
        )

        assert result is not None
        # Should reference the query topic
        assert "장학금" in result or "관련" in result

    def test_clarification_boundary_confidence(self, generator):
        """Test clarification at confidence boundary (0.5)."""
        # At exactly 0.5, should not generate clarification
        result = generator.generate(
            query="휴학 신청",
            intent=IntentCategory.PROCEDURE,
            confidence=0.5,
        )

        assert result is None

    def test_clarification_just_below_boundary(self, generator):
        """Test clarification just below confidence boundary (0.49)."""
        result = generator.generate(
            query="휴학 신청",
            intent=IntentCategory.PROCEDURE,
            confidence=0.49,
        )

        assert result is not None


class TestIntentHandlerIntegration:
    """Integration tests for IntentHandler with IntentClassifier."""

    @pytest.fixture
    def handler(self):
        """Create IntentHandler instance for testing."""
        return IntentHandler()

    def test_handler_has_intent_configs_defined(self, handler):
        """Test that handler has all intent configs defined."""
        assert hasattr(handler, "INTENT_CONFIGS")
        assert IntentCategory.PROCEDURE in handler.INTENT_CONFIGS
        assert IntentCategory.ELIGIBILITY in handler.INTENT_CONFIGS
        assert IntentCategory.DEADLINE in handler.INTENT_CONFIGS
        assert IntentCategory.GENERAL in handler.INTENT_CONFIGS

    def test_intent_configs_are_search_config_instances(self, handler):
        """Test that intent configs are IntentSearchConfig instances."""
        for category in IntentCategory:
            config = handler.INTENT_CONFIGS[category]
            assert isinstance(config, IntentSearchConfig)

    def test_clarification_threshold_configurable(self):
        """Test that clarification threshold is configurable."""
        # Default threshold
        handler_default = IntentHandler()
        assert handler_default.clarification_threshold == 0.5

        # Custom threshold
        handler_custom = IntentHandler(clarification_threshold=0.3)
        assert handler_custom.clarification_threshold == 0.3

        # Verify behavior changes with threshold
        assert handler_custom.should_generate_clarification(0.35) is False
        assert handler_custom.should_generate_clarification(0.25) is True
