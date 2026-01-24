"""
Context Tracker for Multi-Turn Conversations.

Domain service for tracking conversation history, extracting implicit information,
and monitoring intent evolution across multiple turns.

Clean Architecture: Domain layer contains business logic for context management.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .entities import ContextHistory, Turn

logger = logging.getLogger(__name__)


class ContextTracker:
    """
    Domain service for tracking conversation context.

    Manages conversation history, extracts implicit information,
    and tracks intent evolution across multi-turn conversations.
    """

    def __init__(self, context_window_size: int = 3):
        """
        Initialize the context tracker.

        Args:
            context_window_size: Number of previous turns to consider as context.
        """
        self.context_window_size = context_window_size

    def create_initial_context(
        self,
        scenario_id: str,
        initial_turn: "Turn",
    ) -> "ContextHistory":
        """
        Create initial context history for a new conversation.

        Args:
            scenario_id: Unique identifier for the scenario.
            initial_turn: First turn in the conversation.

        Returns:
            ContextHistory with initial turn.
        """
        from .entities import ContextHistory

        return ContextHistory(
            scenario_id=scenario_id,
            conversation_history=[initial_turn],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[initial_turn.query],
        )

    def update_context(
        self,
        context: "ContextHistory",
        new_turn: "Turn",
    ) -> "ContextHistory":
        """
        Update context history with a new turn.

        Args:
            context: Existing context history.
            new_turn: New turn to add.

        Returns:
            Updated ContextHistory.
        """
        # Extract implicit information from the turn
        implicit_info = self._extract_implicit_info(new_turn)

        # Track intent evolution (use pre-set value if available, otherwise track)
        intent_evolution = new_turn.intent_evolution or self._track_intent_evolution(
            context, new_turn
        )

        # Detect topic transitions
        topic_transition = self._detect_topic_transition(context, new_turn)

        # Create updated context (immutable, so return new instance)
        from .entities import ContextHistory

        updated_history = context.conversation_history + [new_turn]
        updated_entities = {**context.implicit_entities, **implicit_info}
        updated_transitions = (
            context.topic_transitions + [topic_transition]
            if topic_transition
            else context.topic_transitions
        )
        updated_intents = (
            context.intent_history + [intent_evolution]
            if intent_evolution
            else context.intent_history
        )

        return ContextHistory(
            scenario_id=context.scenario_id,
            conversation_history=updated_history,
            implicit_entities=updated_entities,
            topic_transitions=updated_transitions,
            intent_history=updated_intents,
        )

    def get_context_summary(
        self,
        context: "ContextHistory",
    ) -> str:
        """
        Generate a summary of conversation context.

        Args:
            context: Context history to summarize.

        Returns:
            String summary of context.
        """
        recent_turns = context.get_recent_context(self.context_window_size)

        summary_parts = [
            f"Conversation has {len(context.conversation_history)} turns.",
            f"Recent context: {len(recent_turns)} turns.",
        ]

        if context.implicit_entities:
            summary_parts.append(
                f"Implicit entities extracted: {len(context.implicit_entities)}."
            )

        if context.topic_transitions:
            summary_parts.append(
                f"Topic transitions: {len(context.topic_transitions)}."
            )

        return " ".join(summary_parts)

    def detect_context_preservation(
        self,
        context: "ContextHistory",
        current_turn: "Turn",
    ) -> bool:
        """
        Detect if the system preserved context from previous turns.

        Args:
            context: Previous context history.
            current_turn: Current turn to evaluate.

        Returns:
            True if context was preserved, False otherwise.
        """
        if not context.conversation_history:
            return True  # No previous context to preserve

        # Get recent context
        recent_turns = context.get_recent_context(self.context_window_size)

        # Check if current answer references previous information
        context_preserved = self._check_context_reference(current_turn, recent_turns)

        return context_preserved

    def _extract_implicit_info(self, turn: "Turn") -> Dict[str, str]:
        """
        Extract implicit information from a turn.

        Args:
            turn: Turn to analyze.

        Returns:
            Dictionary of implicit entities extracted.
        """
        implicit_info = {}

        # Extract from query (pronouns, references to previous answers)
        query_lower = turn.query.lower()

        # Common implicit references in Korean
        implicit_markers = [
            "그거",
            "그것",
            "그건",
            "이거",
            "이건",
            "언제",
            "어디서",
            "누가",
            "어떻게",
        ]

        for marker in implicit_markers:
            if marker in query_lower:
                implicit_info[f"implicit_{marker}"] = turn.query

        # Extract from answer (entities mentioned without explicit query)
        if turn.answer:
            # Simple entity extraction (can be enhanced with NER)
            answer_sentences = turn.answer.split(".")
            for sentence in answer_sentences[:2]:  # First 2 sentences
                if any(keyword in sentence for keyword in ["규정", "조항", "법", "칙"]):
                    implicit_info["mentioned_regulation"] = sentence.strip()
                    break

        return implicit_info

    def _track_intent_evolution(
        self,
        context: "ContextHistory",
        new_turn: "Turn",
    ) -> Optional[str]:
        """
        Track how intent evolved from previous turn.

        Args:
            context: Previous context history.
            new_turn: New turn to analyze.

        Returns:
            Description of intent evolution, or None.
        """
        if not context.conversation_history:
            return None

        previous_turn = context.conversation_history[-1]

        # Compare queries to detect intent shift
        intent_evolution = None

        if new_turn.follow_up_type:
            from .entities import FollowUpType

            follow_up = new_turn.follow_up_type

            if follow_up == FollowUpType.CLARIFICATION:
                intent_evolution = f"Clarifying: {previous_turn.query[:50]}..."
            elif follow_up == FollowUpType.RELATED_EXPANSION:
                intent_evolution = (
                    f"Expanding to related topic of {previous_turn.query[:30]}..."
                )
            elif follow_up == FollowUpType.EXCEPTION_CHECK:
                intent_evolution = (
                    f"Checking exceptions to {previous_turn.query[:30]}..."
                )
            elif follow_up == FollowUpType.PROCEDURAL_DEEPENING:
                intent_evolution = (
                    f"Deepening procedure for {previous_turn.query[:30]}..."
                )
            elif follow_up == FollowUpType.CONDITION_CHANGE:
                intent_evolution = (
                    f"Changing conditions for {previous_turn.query[:30]}..."
                )
            elif follow_up == FollowUpType.CONFIRMATION:
                intent_evolution = (
                    f"Confirming understanding of {previous_turn.query[:30]}..."
                )
            elif follow_up == FollowUpType.GO_BACK:
                intent_evolution = "Returning to earlier topic"
            elif follow_up == FollowUpType.COMPARISON:
                intent_evolution = (
                    f"Comparing options for {previous_turn.query[:30]}..."
                )

        return intent_evolution

    def _detect_topic_transition(
        self,
        context: "ContextHistory",
        new_turn: "Turn",
    ) -> Optional[str]:
        """
        Detect if topic changed significantly.

        Args:
            context: Previous context history.
            new_turn: New turn to analyze.

        Returns:
            Description of topic transition, or None.
        """
        if not context.conversation_history:
            return None

        previous_turn = context.conversation_history[-1]

        # Simple keyword-based transition detection
        # In production, would use semantic similarity
        prev_keywords = set(previous_turn.query.lower().split())
        new_keywords = set(new_turn.query.lower().split())

        overlap = len(prev_keywords & new_keywords)
        total_unique = len(prev_keywords | new_keywords)

        if total_unique > 0:
            similarity = overlap / total_unique
            if similarity < 0.3:  # Low similarity indicates topic change
                return f"Topic transition: {previous_turn.query[:30]}... -> {new_turn.query[:30]}..."

        return None

    def _check_context_reference(
        self,
        current_turn: "Turn",
        previous_turns: List["Turn"],
    ) -> bool:
        """
        Check if current turn references context from previous turns.

        Args:
            current_turn: Current turn to evaluate.
            previous_turns: Previous turns to check for references.

        Returns:
            True if context was referenced, False otherwise.
        """
        if not previous_turns:
            return True

        # Check for pronouns and references in query
        query_lower = current_turn.query.lower()

        # Korean pronouns and references
        context_markers = [
            "그",
            "이",
            "저",
            "위에서",
            "아까",
            "방금",
            "먼저",
            "그다음",
            "그리고",
        ]

        has_context_marker = any(marker in query_lower for marker in context_markers)

        # Check if answer refers to previous information
        answer_refers = False
        if current_turn.answer:
            for prev_turn in previous_turns:
                if prev_turn.sources and any(
                    source in current_turn.answer for source in prev_turn.sources
                ):
                    answer_refers = True
                    break

        return has_context_marker or answer_refers
