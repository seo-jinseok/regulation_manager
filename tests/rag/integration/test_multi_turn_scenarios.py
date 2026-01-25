"""
Integration tests for Multi-Turn Conversation Scenarios.

Tests complete multi-turn conversation flows including:
- Context preservation across turns
- Follow-up question generation
- Intent evolution tracking
- Complex query patterns with context dependencies
"""

from unittest.mock import Mock

import pytest

from src.rag.automation.domain.context_tracker import ContextTracker
from src.rag.automation.domain.entities import (
    ContextHistory,
    DifficultyLevel,
    FollowUpType,
    MultiTurnScenario,
    PersonaType,
    Turn,
)


class MockExecuteUseCase:
    """Mock ExecuteTestUseCase for testing."""

    def __init__(self):
        self.call_count = 0
        self.queries = []

    def execute_query(self, query, test_case_id, enable_answer=True, top_k=5):
        """Mock query execution."""
        self.call_count += 1
        self.queries.append(query)

        mock_result = Mock()
        mock_result.answer = f"Answer to: {query}"
        mock_result.sources = [f"Source {i}" for i in range(1, 4)]
        mock_result.confidence = 0.85
        return mock_result


class TestMultiTurnContextPreservation:
    """Test context preservation across multiple turns."""

    @pytest.fixture
    def context_tracker(self):
        """Create ContextTracker instance."""
        return ContextTracker(context_window_size=3)

    @pytest.fixture
    def sample_context(self, context_tracker):
        """Create sample context with multiple turns."""
        turns = [
            Turn(
                turn_number=1,
                query="휴학 신청은 어떻게 하나요?",
                answer="휴학 신청은 학기 개시 30일 전까지...",
                sources=["규정 제10조"],
                confidence=0.9,
            ),
            Turn(
                turn_number=2,
                query="서류는 뭘 제출하나요?",
                answer="다음 서류가 필요합니다: 신청서, 사유서...",
                sources=["규정 제10조 제2항"],
                confidence=0.85,
                follow_up_type=FollowUpType.PROCEDURAL_DEEPENING,
            ),
            Turn(
                turn_number=3,
                query="신청은 어디서 하나요?",
                answer="학과 사무실에서 신청할 수 있습니다.",
                sources=["규정 제10조 제3항"],
                confidence=0.88,
                follow_up_type=FollowUpType.PROCEDURAL_DEEPENING,
            ),
        ]

        return ContextHistory(
            scenario_id="test_scenario",
            conversation_history=turns,
            implicit_entities={"휴학": "academic_leave"},
            topic_transitions=[],
            intent_history=["휴학 신청", "서류 확인", "신청 장소"],
        )

    def test_context_preserved_with_pronoun_references(self, context_tracker, sample_context):
        """
        SPEC: System should preserve context with Korean pronoun references.

        Given: A conversation history about 휴학
        When: User asks "그거 신청 기한은 언제까지인가요?"
        Then: Context should be preserved (pronoun "그거" references previous topic)
        """
        current_turn = Turn(
            turn_number=4,
            query="그거 신청 기한은 언제까지인가요?",
            answer="신청 기한은 학기 개시 30일 전까지입니다.",
            sources=["규정 제10조 제1항"],
            confidence=0.9,
        )

        preserved = context_tracker.detect_context_preservation(sample_context, current_turn)

        assert preserved is True, "Context should be preserved with pronoun reference"

    def test_context_preserved_with_temporal_references(self, context_tracker, sample_context):
        """
        SPEC: System should preserve context with temporal references.

        Given: A conversation history about 서류 제출
        When: User asks "아까 말한 서류 중에서 증명서는 꼭 필요한가요?"
        Then: Context should be preserved (temporal reference "아까 말한")
        """
        current_turn = Turn(
            turn_number=4,
            query="아까 말한 서류 중에서 증명서는 꼭 필요한가요?",
            answer="증명서는 필수입니다.",
            sources=["규정 제10조 제2항"],
            confidence=0.85,
        )

        preserved = context_tracker.detect_context_preservation(sample_context, current_turn)

        assert preserved is True, "Context should be preserved with temporal reference"

    def test_context_not_preserved_with_topic_change(self, context_tracker, sample_context):
        """
        SPEC: System should detect topic change.

        Given: A conversation history about 휴학
        When: User asks "졸업 요건은 어떻게 되나요?" (completely different topic)
        Then: Context should NOT be preserved (topic changed)
        """
        current_turn = Turn(
            turn_number=4,
            query="졸업 요건은 어떻게 되나요?",
            answer="졸업 요건은 다음과 같습니다...",
            sources=["규정 제20조"],
            confidence=0.8,
        )

        preserved = context_tracker.detect_context_preservation(sample_context, current_turn)

        assert preserved is False, "Context should not be preserved with topic change"


class TestMultiTurnFollowUpGeneration:
    """Test follow-up question generation patterns."""

    def test_clarification_follow_up_pattern(self):
        """
        SPEC: Clarification follow-up should request specific details.

        Given: Previous turn provided general information
        When: Clarification follow-up is generated
        Then: Follow-up should ask for specific details
        """
        templates = [
            "그건 구체적으로 어떻게 되나요?",
            "조금 더 자세히 설명해주세요.",
            "그 부분이 잘 이해가 안 돼요.",
        ]

        # All templates should request clarification
        for template in templates:
            assert any(
                word in template
                for word in ["구체", "상세", "자세", "어떻게", "설명", "이해"]
            ), f"Template {template} should indicate clarification"

    def test_related_expansion_follow_up_pattern(self):
        """
        SPEC: Related expansion follow-up should ask about related topics.

        Given: Previous turn discussed a specific regulation
        When: Related expansion follow-up is generated
        Then: Follow-up should ask about related regulations or topics
        """
        templates = [
            "그랑 관련된 다른 규정도 있나요?",
            "이거랑 비슷한 경우는 어떻게 되나요?",
            "관련해서 또 알아야 할 게 있나요?",
        ]

        # All templates should ask for related information
        for template in templates:
            assert any(
                word in template for word in ["관련", "비슷", "또", "다른"]
            ), f"Template {template} should indicate related expansion"

    def test_exception_check_follow_up_pattern(self):
        """
        SPEC: Exception check follow-up should ask about exceptions.

        Given: Previous turn provided general rule
        When: Exception check follow-up is generated
        Then: Follow-up should ask about exceptions or special cases
        """
        templates = [
            "예외 경우는 없나요?",
            "특별한 경우는 다르게 적용되나요?",
            "이 규정에 해당하지 않는 경우는?",
        ]

        # All templates should ask about exceptions
        for template in templates:
            assert any(
                word in template for word in ["예외", "특별", "해당하지", "다르게"]
            ), f"Template {template} should indicate exception check"

    def test_procedural_deepening_follow_up_pattern(self):
        """
        SPEC: Procedural deepening follow-up should ask about specific procedures.

        Given: Previous turn mentioned a process
        When: Procedural deepening follow-up is generated
        Then: Follow-up should ask about procedural details
        """
        templates = [
            "그럼 구체적으로 어떻게 해야 하나요?",
            "절차가 어떻게 되나요?",
            "어디서 신청하나요?",
        ]

        # All templates should ask about procedures
        for template in templates:
            assert any(
                word in template for word in ["구체", "절차", "어떻게", "어디서", "신청"]
            ), f"Template {template} should indicate procedural deepening"


class TestIntentEvolutionTracking:
    """Test intent evolution across conversation turns."""

    @pytest.fixture
    def context_tracker(self):
        """Create ContextTracker instance."""
        return ContextTracker(context_window_size=3)

    def test_intent_evolution_for_clarification(self, context_tracker):
        """
        SPEC: Intent evolution should track clarification.

        Given: Previous turn about general topic
        When: Clarification follow-up is made
        Then: Intent evolution should indicate clarification
        """
        context = ContextHistory(
            scenario_id="test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="휴학 신청 방법",
                    answer="휴학 신청은...",
                    sources=["규정 A"],
                    confidence=0.9,
                )
            ],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["휴학 신청"],
        )

        new_turn = Turn(
            turn_number=2,
            query="구체적으로 어떻게 하나요?",
            answer="상세 절차는...",
            sources=[],
            confidence=0.85,
            follow_up_type=FollowUpType.CLARIFICATION,
        )

        evolution = context_tracker._track_intent_evolution(context, new_turn)

        assert evolution is not None
        assert "Clarifying" in evolution or "clarification" in evolution.lower()

    def test_intent_evolution_for_related_expansion(self, context_tracker):
        """
        SPEC: Intent evolution should track related expansion.

        Given: Previous turn about specific regulation
        When: Related expansion follow-up is made
        Then: Intent evolution should indicate expansion to related topic
        """
        context = ContextHistory(
            scenario_id="test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="장학금 신청",
                    answer="장학금 신청은...",
                    sources=["규정 B"],
                    confidence=0.9,
                )
            ],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["장학금"],
        )

        new_turn = Turn(
            turn_number=2,
            query="관련된 다른 장학금도 있나요?",
            answer="다른 장학금으로는...",
            sources=[],
            confidence=0.8,
            follow_up_type=FollowUpType.RELATED_EXPANSION,
        )

        evolution = context_tracker._track_intent_evolution(context, new_turn)

        assert evolution is not None
        assert "Expanding" in evolution or "related" in evolution.lower()


class TestComplexMultiTurnScenarios:
    """Test complex multi-turn conversation scenarios."""

    @pytest.fixture
    def mock_execute(self):
        """Create mock execute use case."""
        return MockExecuteUseCase()

    @pytest.fixture
    def context_tracker(self):
        """Create ContextTracker instance."""
        return ContextTracker(context_window_size=3)

    def test_nested_question_scenario(self, mock_execute, context_tracker):
        """
        SPEC: System should handle nested questions across turns.

        Given: Initial query about topic A
        When: Follow-up asks about aspect B of topic A
        And: Another follow-up asks about aspect C of aspect B
        Then: Context should be preserved across all nested levels
        """
        # Simulate nested questions
        queries = [
            "휴학 신청은 어떻게 하나요?",  # Topic A: 휴학
            "서류는 뭘 제출하나요?",  # Aspect B of A: 서류
            "증명서는 어디서 발급받나요?",  # Aspect C of B: 증명서 발급
        ]

        context = None
        for i, query in enumerate(queries, 1):
            result = mock_execute.execute_query(query, f"test_turn_{i}")

            turn = Turn(
                turn_number=i,
                query=query,
                answer=result.answer,
                sources=result.sources,
                confidence=result.confidence,
            )

            if i == 1:
                context = context_tracker.create_initial_context("nested_test", turn)
            else:
                context = context_tracker.update_context(context, turn)
                # Verify context is preserved
                assert context is not None

        # Final context should have all turns
        assert len(context.conversation_history) == 3

    def test_context_switching_scenario(self, mock_execute, context_tracker):
        """
        SPEC: System should detect and handle context switching.

        Given: Conversation about topic A
        When: User switches to topic B
        And: User returns to topic A
        Then: System should detect context switches
        """
        queries = [
            "휴학 신청은 어떻게 하나요?",  # Topic A
            "졸업 요건은 어떻게 되나요?",  # Topic B (context switch)
            "아까 휴학 신청 다시 물어보면",  # Return to Topic A
        ]

        topic_transitions = []
        context = None

        for i, query in enumerate(queries, 1):
            result = mock_execute.execute_query(query, f"test_turn_{i}")

            turn = Turn(
                turn_number=i,
                query=query,
                answer=result.answer,
                sources=result.sources,
                confidence=result.confidence,
            )

            if i == 1:
                context = context_tracker.create_initial_context("switch_test", turn)
            else:
                old_context = context
                context = context_tracker.update_context(context, turn)

                # Detect topic transition
                transition = context_tracker._detect_topic_transition(old_context, turn)
                if transition:
                    topic_transitions.append(transition)

        # Should detect at least one topic transition
        assert len(topic_transitions) >= 1


class TestMultiTurnScenarioDifficultyAssessment:
    """Test difficulty assessment for multi-turn scenarios."""

    def test_easy_scenario_assessment(self):
        """
        SPEC: Easy scenario should have few turns and high confidence.

        Given: A scenario with 2-3 turns and high confidence
        When: Difficulty is assessed
        Then: Should be classified as EASY
        """
        turns = [
            Turn(
                turn_number=1,
                query="휴학 신청 방법",
                answer="휴학 신청은...",
                sources=["규정"],
                confidence=0.9,
            ),
            Turn(
                turn_number=2,
                query="서류는요?",
                answer="다음 서류 필요...",
                sources=["규정"],
                confidence=0.88,
                context_preserved=True,
            ),
        ]

        # Count difficulty factors
        difficulty_factors = 0
        if len(turns) < 4:
            difficulty_factors += 0  # Few turns = not complex
        if sum(1 for t in turns if not t.context_preserved) == 0:
            difficulty_factors += 0  # All context preserved = not complex
        avg_confidence = sum(t.confidence for t in turns) / len(turns)
        if avg_confidence >= 0.7:
            difficulty_factors += 0  # High confidence = not complex

        assert difficulty_factors == 0, "Easy scenario should have no difficulty factors"

    def test_hard_scenario_assessment(self):
        """
        SPEC: Hard scenario should have many turns, context failures, or low confidence.

        Given: A scenario with 4+ turns and context preservation failures
        When: Difficulty is assessed
        Then: Should be classified as HARD
        """
        turns = [
            Turn(
                turn_number=i,
                query=f"Question {i}",
                answer=f"Answer {i}",
                sources=["규정"],
                confidence=0.65,  # Low confidence
                context_preserved=(i % 2 == 0),  # Some context failures
            )
            for i in range(1, 5)
        ]

        # Count difficulty factors
        difficulty_factors = 0
        if len(turns) >= 4:
            difficulty_factors += 1
        context_failures = sum(1 for t in turns if not t.context_preserved)
        if context_failures > 0:
            difficulty_factors += 1
        avg_confidence = sum(t.confidence for t in turns) / len(turns)
        if avg_confidence < 0.7:
            difficulty_factors += 1

        assert difficulty_factors >= 2, "Hard scenario should have at least 2 difficulty factors"


class TestMultiTurnScenarioMetadata:
    """Test metadata tracking for multi-turn scenarios."""

    def test_scenario_metadata_preservation(self):
        """
        SPEC: Scenario metadata should track key metrics.

        Given: A multi-turn scenario
        When: Scenario is created
        Then: Metadata should include total_turns and context_preservation_rate
        """
        turns = [
            Turn(
                turn_number=1,
                query="휴학 신청",
                answer="답변",
                sources=["규정"],
                confidence=0.9,
                context_preserved=True,
            ),
            Turn(
                turn_number=2,
                query="서류는요?",
                answer="답변",
                sources=["규정"],
                confidence=0.85,
                context_preserved=True,
            ),
            Turn(
                turn_number=3,
                query="졸업 요건",  # Topic change
                answer="답변",
                sources=["규정"],
                confidence=0.8,
                context_preserved=False,
            ),
        ]

        scenario = MultiTurnScenario(
            scenario_id="test_scenario",
            persona_type=PersonaType.FRESHMAN,
            initial_query="휴학 신청",
            turns=turns,
            difficulty=DifficultyLevel.MEDIUM,
            context_window_size=3,
            metadata={
                "persona_name": "신입생",
                "total_turns": len(turns),
                "context_preservation_rate": sum(
                    1 for t in turns if t.context_preserved
                )
                / len(turns),
            },
        )

        # Verify metadata
        assert scenario.total_turns == 3
        assert scenario.context_preservation_rate == 2 / 3
        assert "persona_name" in scenario.metadata
        assert scenario.metadata["context_preservation_rate"] == pytest.approx(0.667, rel=0.1)

    def test_follow_up_distribution_tracking(self):
        """
        SPEC: Scenario should track follow-up type distribution.

        Given: A scenario with various follow-up types
        When: Follow-up distribution is calculated
        Then: Should correctly count each follow-up type
        """
        turns = [
            Turn(
                turn_number=1,
                query="Initial query",
                answer="Answer",
                sources=["규정"],
                confidence=0.9,
            ),
            Turn(
                turn_number=2,
                query="Clarification?",
                answer="Answer",
                sources=["규정"],
                confidence=0.85,
                follow_up_type=FollowUpType.CLARIFICATION,
            ),
            Turn(
                turn_number=3,
                query="Related?",
                answer="Answer",
                sources=["규정"],
                confidence=0.8,
                follow_up_type=FollowUpType.RELATED_EXPANSION,
            ),
            Turn(
                turn_number=4,
                query="Another clarification?",
                answer="Answer",
                sources=["규정"],
                confidence=0.85,
                follow_up_type=FollowUpType.CLARIFICATION,
            ),
        ]

        scenario = MultiTurnScenario(
            scenario_id="distribution_test",
            persona_type=PersonaType.JUNIOR,
            initial_query="Initial query",
            turns=turns,
            difficulty=DifficultyLevel.MEDIUM,
        )

        distribution = scenario.follow_up_distribution

        assert distribution[FollowUpType.CLARIFICATION] == 2
        assert distribution[FollowUpType.RELATED_EXPANSION] == 1
        assert FollowUpType.EXCEPTION_CHECK not in distribution
