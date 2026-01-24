"""
Specification tests for ContextTracker (Phase 3).

Tests define expected behavior for multi-turn conversation context tracking.
Following DDD PRESERVE phase: specification tests for greenfield development.
"""


import pytest

from src.rag.automation.domain.context_tracker import ContextTracker
from src.rag.automation.domain.entities import ContextHistory, FollowUpType, Turn


class TestContextTrackerCreation:
    """Specification tests for initial context creation."""

    def test_create_initial_context_creates_context_with_first_turn(self):
        """
        SPEC: create_initial_context should create ContextHistory with initial turn.

        Given: A scenario ID and initial turn
        When: create_initial_context is called
        Then: ContextHistory is created with the turn in conversation_history
        """
        # Arrange
        tracker = ContextTracker(context_window_size=3)
        scenario_id = "test_scenario_001"

        initial_turn = Turn(
            turn_number=1,
            query="휴학 신청은 어떻게 하나요?",
            answer="휴학 신청은...",
            sources=["규정 제1조"],
            confidence=0.85,
        )

        # Act
        context = tracker.create_initial_context(scenario_id, initial_turn)

        # Assert
        assert context.scenario_id == scenario_id
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0] == initial_turn
        assert context.implicit_entities == {}
        assert len(context.intent_history) == 1
        assert context.intent_history[0] == initial_turn.query

    def test_create_initial_context_initializes_empty_tracking_structures(self):
        """
        SPEC: create_initial_context should initialize empty tracking structures.

        Given: A scenario ID and initial turn
        When: create_initial_context is called
        Then: implicit_entities and topic_transitions are empty
        """
        # Arrange
        tracker = ContextTracker()
        scenario_id = "test_scenario_002"

        initial_turn = Turn(
            turn_number=1,
            query="성적 정정은 어떻게 하나요?",
            answer="성적 정정은...",
            sources=[],
            confidence=0.75,
        )

        # Act
        context = tracker.create_initial_context(scenario_id, initial_turn)

        # Assert
        assert context.implicit_entities == {}
        assert context.topic_transitions == []


class TestContextTrackerUpdate:
    """Specification tests for context update."""

    @pytest.fixture
    def sample_context(self):
        """Create a sample context for testing."""
        turn1 = Turn(
            turn_number=1,
            query="휴학 신청은 언제까지 하나요?",
            answer="휴학 신청은 학기 개시 30일 전까지...",
            sources=["규정 제10조"],
            confidence=0.9,
        )

        return ContextHistory(
            scenario_id="test_scenario",
            conversation_history=[turn1],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["휴학 신청"],
        )

    def test_update_context_adds_new_turn_to_history(self, sample_context):
        """
        SPEC: update_context should add new turn to conversation history.

        Given: An existing context with one turn
        When: update_context is called with a new turn
        Then: New turn is added to conversation_history
        """
        # Arrange
        tracker = ContextTracker()

        new_turn = Turn(
            turn_number=2,
            query="그건 구체적으로 어떻게 되나요?",
            answer="구체적으로는...",
            sources=["규정 제10조 제2항"],
            confidence=0.88,
            follow_up_type=FollowUpType.CLARIFICATION,
        )

        # Act
        updated_context = tracker.update_context(sample_context, new_turn)

        # Assert
        assert len(updated_context.conversation_history) == 2
        assert updated_context.conversation_history[1] == new_turn
        assert updated_context.conversation_history[0] == sample_context.conversation_history[0]

    def test_update_context_extends_intent_history(self, sample_context):
        """
        SPEC: update_context should extend intent history with evolution.

        Given: An existing context
        When: update_context is called with a turn that has intent evolution
        Then: intent_history is extended with the new evolution
        """
        # Arrange
        tracker = ContextTracker()

        new_turn = Turn(
            turn_number=2,
            query="더 자세히 알려주세요",
            answer="상세 절차는...",
            sources=[],
            confidence=0.85,
            intent_evolution="Clarifying: 휴학 신청...",
        )

        # Act
        updated_context = tracker.update_context(sample_context, new_turn)

        # Assert
        assert len(updated_context.intent_history) == 2
        assert updated_context.intent_history[1] == "Clarifying: 휴학 신청..."


class TestContextPreservationDetection:
    """Specification tests for context preservation detection."""

    @pytest.fixture
    def context_with_history(self):
        """Create a context with conversation history."""
        turn1 = Turn(
            turn_number=1,
            query="휴학 신청 방법",
            answer="휴학 신청은...",
            sources=["규정 A"],
            confidence=0.9,
        )

        turn2 = Turn(
            turn_number=2,
            query="서류는 뭘 제출하나요?",
            answer="다음 서류가 필요합니다...",
            sources=["규정 A-1"],
            confidence=0.85,
        )

        return ContextHistory(
            scenario_id="test",
            conversation_history=[turn1, turn2],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["휴학", "서류"],
        )

    def test_detect_context_preservation_with_pronoun_reference(self, context_with_history):
        """
        SPEC: detect_context_preservation should return True with pronoun references.

        Given: A context with previous turns
        When: Current turn contains pronouns referencing previous context
        Then: Returns True (context preserved)
        """
        # Arrange
        tracker = ContextTracker()

        current_turn = Turn(
            turn_number=3,
            query="그거 신청은 어디서 하나요?",  # "그거" references previous
            answer="신청은 학과 사무실에서...",
            sources=["규정 A-2"],
            confidence=0.8,
        )

        # Act
        preserved = tracker.detect_context_preservation(context_with_history, current_turn)

        # Assert
        assert preserved is True

    def test_detect_context_preservation_without_references(self, context_with_history):
        """
        SPEC: detect_context_preservation should return False without references.

        Given: A context with previous turns
        When: Current turn has no reference to previous context
        Then: Returns False (context not preserved)
        """
        # Arrange
        tracker = ContextTracker()

        current_turn = Turn(
            turn_number=3,
            query="졸업 요건은 어떻게 되나요?",  # Completely different topic
            answer="졸업 요건은...",
            sources=["규정 B"],
            confidence=0.8,
        )

        # Act
        preserved = tracker.detect_context_preservation(context_with_history, current_turn)

        # Assert
        assert preserved is False


class TestContextSummary:
    """Specification tests for context summary generation."""

    @pytest.fixture
    def rich_context(self):
        """Create a context with rich information."""
        turns = [
            Turn(
                turn_number=i,
                query=f"질문 {i}",
                answer=f"답변 {i}",
                sources=[f"규정 {i}"],
                confidence=0.8,
            )
            for i in range(1, 6)
        ]

        return ContextHistory(
            scenario_id="rich_scenario",
            conversation_history=turns,
            implicit_entities={"entity1": "value1", "entity2": "value2"},
            topic_transitions=["transition1"],
            intent_history=["intent1", "intent2"],
        )

    def test_get_context_summary_includes_conversation_length(self, rich_context):
        """
        SPEC: get_context_summary should include total turn count.

        Given: A context with multiple turns
        When: get_context_summary is called
        Then: Summary includes the total number of turns
        """
        # Arrange
        tracker = ContextTracker()

        # Act
        summary = tracker.get_context_summary(rich_context)

        # Assert
        assert "5 turns" in summary

    def test_get_context_summary_includes_entity_count(self, rich_context):
        """
        SPEC: get_context_summary should include implicit entity count.

        Given: A context with extracted implicit entities
        When: get_context_summary is called
        Then: Summary includes the count of implicit entities
        """
        # Arrange
        tracker = ContextTracker()

        # Act
        summary = tracker.get_context_summary(rich_context)

        # Assert
        assert "2" in summary or "Implicit entities" in summary

    def test_get_context_summary_includes_transition_count(self, rich_context):
        """
        SPEC: get_context_summary should include topic transition count.

        Given: A context with topic transitions
        When: get_context_summary is called
        Then: Summary includes the count of topic transitions
        """
        # Arrange
        tracker = ContextTracker()

        # Act
        summary = tracker.get_context_summary(rich_context)

        # Assert
        assert "1" in summary or "transitions" in summary.lower()
