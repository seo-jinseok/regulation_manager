"""
Multi-Turn Conversation Simulator.

Infrastructure service for simulating multi-turn conversations with
context-aware follow-up question generation.

Clean Architecture: Infrastructure layer uses external dependencies (LLM).
"""

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..application.execute_test_usecase import ExecuteTestUseCase
    from ..domain.context_tracker import ContextHistory, ContextTracker
    from ..domain.entities import (
        DifficultyLevel,
        FollowUpType,
        MultiTurnScenario,
        Persona,
        Turn,
    )

logger = logging.getLogger(__name__)


class MultiTurnSimulator:
    """
    Infrastructure service for simulating multi-turn conversations.

    Generates context-aware follow-up questions and executes multi-turn
    scenarios to test RAG system's context management capabilities.
    """

    def __init__(
        self,
        execute_usecase: "ExecuteTestUseCase",
        context_tracker: "ContextTracker",
        llm_client: Optional[object] = None,
    ):
        """
        Initialize the multi-turn simulator.

        Args:
            execute_usecase: Use case for executing individual queries.
            context_tracker: Domain service for tracking context.
            llm_client: Optional LLM client for generating follow-up questions.
        """
        self.execute = execute_usecase
        self.context_tracker = context_tracker
        self.llm_client = llm_client

    def generate_scenario(
        self,
        scenario_id: str,
        persona: "Persona",
        initial_query: str,
        min_turns: int = 3,
        max_turns: int = 5,
        context_window_size: int = 3,
    ) -> "MultiTurnScenario":
        """
        Generate a multi-turn conversation scenario.

        Args:
            scenario_id: Unique identifier for the scenario.
            persona: User persona for the conversation.
            initial_query: Starting query for the conversation.
            min_turns: Minimum number of turns to generate.
            max_turns: Maximum number of turns to generate.
            context_window_size: Context window size for tracking.

        Returns:
            MultiTurnScenario with generated turns.
        """
        from ..domain.entities import MultiTurnScenario, Turn

        logger.info(f"Generating multi-turn scenario: {scenario_id}")
        logger.info(f"Persona: {persona.persona_type.value}, Min turns: {min_turns}")

        # Execute initial turn
        initial_result = self.execute.execute_query(
            query=initial_query,
            test_case_id=f"{scenario_id}_turn_1",
            enable_answer=True,
            top_k=5,
        )

        # Create initial turn
        initial_turn = Turn(
            turn_number=1,
            query=initial_query,
            answer=initial_result.answer,
            sources=initial_result.sources,
            confidence=initial_result.confidence,
            follow_up_type=None,
            intent_evolution=None,
            implicit_info_extracted=[],
            context_preserved=True,
        )

        # Create initial context
        context = self.context_tracker.create_initial_context(scenario_id, initial_turn)

        # Generate subsequent turns
        turns: List[Turn] = [initial_turn]
        turn_count = 1

        while turn_count < max_turns:
            turn_count += 1

            # Generate follow-up question
            follow_up_type = self._select_follow_up_type(turn_count, min_turns)
            follow_up_query = self._generate_follow_up_question(
                context=context,
                follow_up_type=follow_up_type,
                persona=persona,
            )

            if not follow_up_query:
                logger.info(f"No follow-up generated at turn {turn_count}, stopping")
                break

            # Execute follow-up turn
            follow_up_result = self.execute.execute_query(
                query=follow_up_query,
                test_case_id=f"{scenario_id}_turn_{turn_count}",
                enable_answer=True,
                top_k=5,
            )

            # Detect context preservation
            context_preserved = self.context_tracker.detect_context_preservation(
                context,
                Turn(
                    turn_number=turn_count,
                    query=follow_up_query,
                    answer=follow_up_result.answer,
                    sources=follow_up_result.sources,
                    confidence=follow_up_result.confidence,
                ),
            )

            # Track intent evolution
            intent_evolution = self.context_tracker._track_intent_evolution(
                context,
                Turn(
                    turn_number=turn_count,
                    query=follow_up_query,
                    answer=follow_up_result.answer,
                    sources=follow_up_result.sources,
                    confidence=follow_up_result.confidence,
                    follow_up_type=follow_up_type,
                ),
            )

            # Create follow-up turn
            follow_up_turn = Turn(
                turn_number=turn_count,
                query=follow_up_query,
                answer=follow_up_result.answer,
                sources=follow_up_result.sources,
                confidence=follow_up_result.confidence,
                follow_up_type=follow_up_type,
                intent_evolution=intent_evolution,
                implicit_info_extracted=list(
                    self.context_tracker._extract_implicit_info(
                        Turn(
                            turn_number=turn_count,
                            query=follow_up_query,
                            answer=follow_up_result.answer,
                            sources=follow_up_result.sources,
                            confidence=follow_up_result.confidence,
                        )
                    ).keys()
                ),
                context_preserved=context_preserved,
            )

            # Update context
            context = self.context_tracker.update_context(context, follow_up_turn)
            turns.append(follow_up_turn)

            # Check if we've reached minimum turns
            if turn_count >= min_turns:
                # Consider stopping if conversation seems complete
                if self._is_conversation_complete(context):
                    logger.info(f"Conversation appears complete at turn {turn_count}")
                    break

        # Determine difficulty based on conversation complexity
        difficulty = self._assess_scenario_difficulty(turns)

        # Create scenario
        scenario = MultiTurnScenario(
            scenario_id=scenario_id,
            persona_type=persona.persona_type,
            initial_query=initial_query,
            turns=turns,
            difficulty=difficulty,
            context_window_size=context_window_size,
            metadata={
                "persona_name": persona.name,
                "total_turns": len(turns),
                "context_preservation_rate": float(
                    sum(1 for t in turns if t.context_preserved) / len(turns)
                )
                if turns
                else 1.0,
            },
        )

        logger.info(
            f"Scenario generated: {len(turns)} turns, "
            f"context preservation: {scenario.context_preservation_rate:.2%}"
        )

        return scenario

    def _select_follow_up_type(
        self,
        turn_number: int,
        min_turns: int,
    ) -> Optional["FollowUpType"]:
        """
        Select the type of follow-up question to generate.

        Args:
            turn_number: Current turn number.
            min_turns: Minimum number of turns required.

        Returns:
            Selected FollowUpType, or None if should stop.
        """
        from ..domain.entities import FollowUpType

        # Early turns: focus on clarification and deepening
        if turn_number <= 2:
            return FollowUpType.CLARIFICATION

        # Middle turns: expand and check exceptions
        if turn_number <= 3:
            return FollowUpType.RELATED_EXPANSION

        # Later turns: vary between different types
        if turn_number == 4:
            return FollowUpType.EXCEPTION_CHECK

        if turn_number == 5:
            return FollowUpType.PROCEDURAL_DEEPENING

        # After 5 turns, consider stopping
        if turn_number >= min_turns:
            # 30% chance to continue, otherwise stop
            import random

            if random.random() < 0.3:
                types = [
                    FollowUpType.CONDITION_CHANGE,
                    FollowUpType.CONFIRMATION,
                    FollowUpType.COMPARISON,
                ]
                return random.choice(types)

        return None

    def _generate_follow_up_question(
        self,
        context: "ContextHistory",
        follow_up_type: "FollowUpType",
        persona: "Persona",
    ) -> Optional[str]:
        """
        Generate a context-aware follow-up question.

        Args:
            context: Conversation context history.
            follow_up_type: Type of follow-up to generate.
            persona: User persona.

        Returns:
            Generated follow-up question, or None if generation failed.
        """
        from ..domain.entities import FollowUpType

        # Get recent context
        recent_turns = context.get_recent_context(
            self.context_tracker.context_window_size
        )

        if not recent_turns:
            return None

        last_turn = recent_turns[-1]

        # Template-based generation (can be enhanced with LLM)
        templates = {
            FollowUpType.CLARIFICATION: [
                "그건 구체적으로 어떻게 되나요?",
                "조금 더 자세히 설명해주세요.",
                "그 부분이 잘 이해가 안 돼요.",
            ],
            FollowUpType.RELATED_EXPANSION: [
                "그랑 관련된 다른 규정도 있나요?",
                "이거랑 비슷한 경우는 어떻게 되나요?",
                "관련해서 또 알아야 할 게 있나요?",
            ],
            FollowUpType.EXCEPTION_CHECK: [
                "예외 경우는 없나요?",
                "특별한 경우는 다르게 적용되나요?",
                "이 규정에 해당하지 않는 경우는?",
            ],
            FollowUpType.PROCEDURAL_DEEPENING: [
                "그럼 구체적으로 어떻게 해야 하나요?",
                "절차가 어떻게 되나요?",
                "어디서 신청하나요?",
            ],
            FollowUpType.CONDITION_CHANGE: [
                "상황이 이렇게 바뀌면 어떻게 되나요?",
                "만약에 [조건]이라면 달라지나요?",
                "조건이 바뀌면 결과도 달라지나요?",
            ],
            FollowUpType.CONFIRMATION: [
                "그렇게 이해하면 맞나요?",
                "제대로 이해한 건가요?",
                "정리하면 이런 건가요?",
            ],
            FollowUpType.GO_BACK: [
                "처음에 질문한 거 다시 물어보면",
                "아까 한 말 다시 생각해보면",
                "맨 처음 주제로 돌아가서",
            ],
            FollowUpType.COMPARISON: [
                "A랑 B랑 차이가 뭔가요?",
                "어떤 게 더 나은가요?",
                "각각 장단점이 있나요?",
            ],
        }

        # Get template for follow-up type
        type_templates = templates.get(follow_up_type, [])

        if not type_templates:
            return None

        # Select template based on persona
        import random

        template = random.choice(type_templates)

        # Customize template based on context
        follow_up = self._customize_template(template, last_turn, persona)

        return follow_up

    def _customize_template(
        self,
        template: str,
        last_turn: "Turn",
        persona: "Persona",
    ) -> str:
        """
        Customize a follow-up template based on context.

        Args:
            template: Base template string.
            last_turn: Most recent turn for context.
            persona: User persona.

        Returns:
            Customized follow-up question.
        """
        # Simple customization: inject persona-specific style
        # In production, would use LLM for more sophisticated generation

        customized = template

        # Add persona-specific prefix if available
        if persona.query_styles:
            import random

            prefix = random.choice(persona.query_styles)
            customized = f"{prefix} {template}"

        return customized

    def _is_conversation_complete(self, context: "ContextHistory") -> bool:
        """
        Determine if conversation has reached a natural conclusion.

        Args:
            context: Conversation context history.

        Returns:
            True if conversation appears complete, False otherwise.
        """
        # Check if last turn's answer suggests completion
        if not context.conversation_history:
            return False

        last_turn = context.conversation_history[-1]

        # Check for completion indicators in answer
        completion_indicators = [
            "더 이상",
            "없습니다",
            "마지막으로",
            "완료",
            "종료",
        ]

        answer_lower = last_turn.answer.lower() if last_turn.answer else ""

        if any(indicator in answer_lower for indicator in completion_indicators):
            return True

        # Check if we've had many turns without new information
        if len(context.conversation_history) > 5:
            # Check if last few turns had similar confidence scores
            recent_confidences = [
                t.confidence for t in context.conversation_history[-3:]
            ]
            if max(recent_confidences) - min(recent_confidences) < 0.1:
                return True

        return False

    def _assess_scenario_difficulty(self, turns: List["Turn"]) -> "DifficultyLevel":
        """
        Assess the difficulty level of a multi-turn scenario.

        Args:
            turns: List of turns in the scenario.

        Returns:
            DifficultyLevel assessment.
        """
        from ..domain.entities import DifficultyLevel

        if not turns:
            return DifficultyLevel.EASY

        # Count factors that increase difficulty
        difficulty_factors = 0

        # More turns = more complex
        if len(turns) >= 4:
            difficulty_factors += 1

        # Context preservation failures increase difficulty
        context_failures = sum(1 for t in turns if not t.context_preserved)
        if context_failures > 0:
            difficulty_factors += 1

        # Low confidence indicates difficulty
        avg_confidence = sum(t.confidence for t in turns) / len(turns)
        if avg_confidence < 0.7:
            difficulty_factors += 1

        # Determine difficulty
        if difficulty_factors >= 2:
            return DifficultyLevel.HARD
        elif difficulty_factors == 1:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.EASY
