"""
Comprehensive Test Use Case for RAG Testing Automation.

Application layer use case for executing comprehensive test scenarios
including ambiguous queries, multi-turn conversations, and edge cases.

Clean Architecture: Application orchestrates domain and infrastructure.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..domain.entities import PersonaType
from ..domain.extended_entities import (
    AmbiguityType,
    EdgeCaseCategory,
)
from ..infrastructure.test_scenario_templates import (
    AmbiguousQueryTemplates,
    EdgeCaseTemplates,
    MultiTurnScenarioTemplates,
)

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveTestSession:
    """
    A comprehensive test session containing all test scenario types.

    Includes ambiguous queries, multi-turn conversations, and edge cases.
    """

    session_id: str
    started_at: datetime

    # Test counts
    ambiguous_query_count: int
    multi_turn_scenario_count: int
    edge_case_count: int

    # Test results
    ambiguous_results: List[Dict[str, Any]] = field(default_factory=list)
    multi_turn_results: List[Dict[str, Any]] = field(default_factory=list)
    edge_case_results: List[Dict[str, Any]] = field(default_factory=list)

    # Summary statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0

    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        """Check if the session is completed."""
        return self.completed_at is not None

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests


@dataclass
class AmbiguousQueryResult:
    """Result of executing an ambiguous query test."""

    query_id: str
    query: str
    ambiguity_type: AmbiguityType

    # System response
    detected_ambiguity: bool
    requested_clarification: bool
    provided_answer: str
    confidence: float

    # Evaluation
    pass_ambiguity_detection: bool
    pass_clarification_request: bool
    pass_answer_quality: bool
    overall_pass: bool

    execution_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTurnResult:
    """Result of executing a multi-turn conversation scenario."""

    scenario_id: str
    persona_type: PersonaType
    total_turns: int

    # Context management metrics
    context_preservation_rate: float
    topic_transitions: List[str]

    # Turn-by-turn results
    turn_results: List[Dict[str, Any]]

    # Overall evaluation
    pass_context_preservation: bool
    pass_follow_up_handling: bool
    pass_overall_quality: bool
    overall_pass: bool

    execution_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeCaseResult:
    """Result of executing an edge case scenario."""

    scenario_id: str
    category: EdgeCaseCategory
    query: str

    # System response characteristics
    showed_empathy: bool
    response_speed: str
    provided_contact: bool
    offered_alternatives: bool
    escalated: bool

    # Answer quality
    provided_answer: str
    confidence: float

    # Evaluation
    pass_empathy_level: bool
    pass_response_speed: bool
    pass_answer_quality: bool
    overall_pass: bool

    execution_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveTestUseCase:
    """
    Use case for executing comprehensive test scenarios.

    Orchestrates testing of ambiguous queries, multi-turn conversations,
    and edge cases to evaluate RAG system quality.
    """

    def __init__(self, query_executor, context_tracker):
        """
        Initialize the comprehensive test use case.

        Args:
            query_executor: Service for executing queries.
            context_tracker: Service for tracking conversation context.
        """
        self.query_executor = query_executor
        self.context_tracker = context_tracker

    def execute_comprehensive_test(
        self,
        session_id: str,
        test_ambiguous: bool = True,
        test_multi_turn: bool = True,
        test_edge_cases: bool = True,
    ) -> ComprehensiveTestSession:
        """
        Execute comprehensive test suite.

        Args:
            session_id: Unique session identifier.
            test_ambiguous: Whether to test ambiguous queries.
            test_multi_turn: Whether to test multi-turn conversations.
            test_edge_cases: Whether to test edge cases.

        Returns:
            ComprehensiveTestSession with all test results.
        """
        logger.info(f"Starting comprehensive test session: {session_id}")
        session = ComprehensiveTestSession(
            session_id=session_id,
            started_at=datetime.now(),
            ambiguous_query_count=30 if test_ambiguous else 0,
            multi_turn_scenario_count=15 if test_multi_turn else 0,
            edge_case_count=15 if test_edge_cases else 0,
        )

        # Execute ambiguous query tests
        if test_ambiguous:
            logger.info("Executing ambiguous query tests...")
            ambiguous_results = self._execute_ambiguous_queries()
            session.ambiguous_results = ambiguous_results

        # Execute multi-turn conversation tests
        if test_multi_turn:
            logger.info("Executing multi-turn conversation tests...")
            multi_turn_results = self._execute_multi_turn_scenarios()
            session.multi_turn_results = multi_turn_results

        # Execute edge case tests
        if test_edge_cases:
            logger.info("Executing edge case tests...")
            edge_case_results = self._execute_edge_cases()
            session.edge_case_results = edge_case_results

        # Calculate summary statistics
        session.total_tests = (
            len(session.ambiguous_results)
            + len(session.multi_turn_results)
            + len(session.edge_case_results)
        )
        session.passed_tests = (
            sum(1 for r in session.ambiguous_results if r.get("overall_pass", False))
            + sum(1 for r in session.multi_turn_results if r.get("overall_pass", False))
            + sum(1 for r in session.edge_case_results if r.get("overall_pass", False))
        )
        session.failed_tests = session.total_tests - session.passed_tests
        session.completed_at = datetime.now()

        logger.info(
            f"Completed comprehensive test session: {session_id} "
            f"({session.passed_tests}/{session.total_tests} passed)"
        )

        return session

    def _execute_ambiguous_queries(self) -> List[Dict[str, Any]]:
        """Execute all ambiguous query tests."""
        templates = AmbiguousQueryTemplates.get_all_templates()
        results = []

        for template in templates:
            result = self._execute_ambiguous_query(template)
            results.append(result)

        return results

    def _execute_ambiguous_query(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single ambiguous query test."""
        import time

        start_time = time.time()

        # Execute query
        query_result = self.query_executor.execute_query(
            query=template["query"],
            test_case_id=f"amb-{template.get('query_id', 'unknown')}",
            enable_answer=True,
            top_k=5,
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Evaluate result
        detected_ambiguity = self._detect_ambiguity_in_response(query_result.answer)
        requested_clarification = self._detect_clarification_request(
            query_result.answer
        )

        # Determine pass/fail
        pass_ambiguity_detection = (
            detected_ambiguity
            if template.get("should_detect_ambiguity", True)
            else True
        )
        pass_clarification = (
            requested_clarification
            if template.get("should_request_clarification", True)
            else True
        )
        pass_answer_quality = len(query_result.answer) > 50  # Basic quality check
        overall_pass = (
            pass_ambiguity_detection and pass_clarification and pass_answer_quality
        )

        return {
            "query_id": template.get("query_id", "unknown"),
            "query": template["query"],
            "ambiguity_type": template["ambiguity_type"].value,
            "detected_ambiguity": detected_ambiguity,
            "requested_clarification": requested_clarification,
            "provided_answer": query_result.answer,
            "confidence": query_result.confidence,
            "pass_ambiguity_detection": pass_ambiguity_detection,
            "pass_clarification_request": pass_clarification,
            "pass_answer_quality": pass_answer_quality,
            "overall_pass": overall_pass,
            "execution_time_ms": execution_time_ms,
        }

    def _execute_multi_turn_scenarios(self) -> List[Dict[str, Any]]:
        """Execute all multi-turn conversation scenario tests."""
        templates = MultiTurnScenarioTemplates.get_all_scenarios()
        results = []

        for template in templates:
            result = self._execute_multi_turn_scenario(template)
            results.append(result)

        return results

    def _execute_multi_turn_scenario(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single multi-turn conversation scenario test."""
        import time

        start_time = time.time()

        # Execute initial turn
        initial_result = self.query_executor.execute_query(
            query=template["initial_query"],
            test_case_id=f"{template['scenario_id']}_turn_1",
            enable_answer=True,
            top_k=5,
        )

        # Create context
        context = self.context_tracker.create_initial_context(
            template["scenario_id"],
            {
                "turn_number": 1,
                "query": template["initial_query"],
                "answer": initial_result.answer,
                "confidence": initial_result.confidence,
            },
        )

        # Execute follow-up turns
        turn_results = []
        context_preserved_count = 0

        for turn_template in template["turns"]:
            turn_result = self.query_executor.execute_query(
                query=turn_template["query"],
                test_case_id=f"{template['scenario_id']}_turn_{turn_template['turn_number']}",
                enable_answer=True,
                top_k=5,
            )

            # Check context preservation
            context_preserved = self._check_context_preservation(
                context, turn_result.answer
            )
            if context_preserved:
                context_preserved_count += 1

            turn_results.append(
                {
                    "turn_number": turn_template["turn_number"],
                    "query": turn_template["query"],
                    "answer": turn_result.answer,
                    "confidence": turn_result.confidence,
                    "context_preserved": context_preserved,
                }
            )

            # Update context
            context = self.context_tracker.update_context(
                context,
                {
                    "turn_number": turn_template["turn_number"],
                    "query": turn_template["query"],
                    "answer": turn_result.answer,
                    "confidence": turn_result.confidence,
                },
            )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Calculate metrics
        total_turns = len(template["turns"]) + 1  # +1 for initial turn
        context_preservation_rate = context_preserved_count / len(template["turns"])

        # Determine pass/fail
        pass_context_preservation = context_preservation_rate >= template.get(
            "expected_context_preservation_rate", 0.8
        )
        pass_follow_up_handling = all(r["confidence"] > 0.5 for r in turn_results)
        pass_overall_quality = context_preservation_rate >= 0.7
        overall_pass = (
            pass_context_preservation
            and pass_follow_up_handling
            and pass_overall_quality
        )

        return {
            "scenario_id": template["scenario_id"],
            "persona_type": template["persona_type"].value,
            "total_turns": total_turns,
            "context_preservation_rate": context_preservation_rate,
            "turn_results": turn_results,
            "pass_context_preservation": pass_context_preservation,
            "pass_follow_up_handling": pass_follow_up_handling,
            "pass_overall_quality": pass_overall_quality,
            "overall_pass": overall_pass,
            "execution_time_ms": execution_time_ms,
        }

    def _execute_edge_cases(self) -> List[Dict[str, Any]]:
        """Execute all edge case scenario tests."""
        templates = EdgeCaseTemplates.get_all_templates()
        results = []

        for template in templates:
            result = self._execute_edge_case(template)
            results.append(result)

        return results

    def _execute_edge_case(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single edge case scenario test."""
        import time

        start_time = time.time()

        # Execute query
        query_result = self.query_executor.execute_query(
            query=template["query"],
            test_case_id=template["scenario_id"],
            enable_answer=True,
            top_k=5,
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Analyze response
        showed_empathy = self._detect_empathy(query_result.answer)
        response_speed = "fast" if execution_time_ms < 2000 else "normal"
        provided_contact = self._detect_contact_info(query_result.answer)
        offered_alternatives = self._detect_alternatives(query_result.answer)
        escalated = self._detect_escalation(query_result.answer)

        # Determine pass/fail
        pass_empathy = (
            showed_empathy if template.get("should_show_empathy", False) else True
        )
        pass_response_speed = response_speed == template.get(
            "expected_response_speed", "normal"
        )
        pass_answer_quality = len(query_result.answer) > 50
        overall_pass = pass_empathy and pass_response_speed and pass_answer_quality

        return {
            "scenario_id": template["scenario_id"],
            "category": template["category"].value,
            "query": template["query"],
            "showed_empathy": showed_empathy,
            "response_speed": response_speed,
            "provided_contact": provided_contact,
            "offered_alternatives": offered_alternatives,
            "escalated": escalated,
            "provided_answer": query_result.answer,
            "confidence": query_result.confidence,
            "pass_empathy_level": pass_empathy,
            "pass_response_speed": pass_response_speed,
            "pass_answer_quality": pass_answer_quality,
            "overall_pass": overall_pass,
            "execution_time_ms": execution_time_ms,
        }

    def _detect_ambiguity_in_response(self, answer: str) -> bool:
        """
        Detect if system identified ambiguity in response.

        Enhanced detection using multiple patterns:
        - Direct clarification questions
        - Multiple option enumeration
        - Context request indicators
        - Uncertainty acknowledgment
        """
        # Direct clarification patterns
        clarification_patterns = [
            "어떤",
            "구체적으로",
            "명확히",
            "어떤 것",
            "정확히",
            "무엇을",
            "말씀해",
        ]

        # Multiple interpretation indicators
        interpretation_patterns = [
            "경우에 따라",
            "두 가지",
            "여러 가지",
            "다음과 같이",
            "예를 들어",
            "먼저 확인해야",
        ]

        # Context request patterns
        context_patterns = [
            "어떤 절차",
            "어떤 신청",
            "어떤 규정",
            "구체적인 상황",
            "자세한 내용",
        ]

        # Acknowledgment of uncertainty
        uncertainty_patterns = [
            "명확하지 않",
            "확인이 필요",
            "추가 정보",
            "상황에 따라",
        ]

        all_patterns = (
            clarification_patterns
            + interpretation_patterns
            + context_patterns
            + uncertainty_patterns
        )

        # Check if any pattern is present
        pattern_found = any(indicator in answer for indicator in all_patterns)

        # Additional check: if answer asks multiple questions, it's detecting ambiguity
        question_count = answer.count("?") + answer.count("?")
        asks_multiple = question_count >= 2

        # Additional check: if answer enumerates options, it's clarifying
        enumerates_options = (
            "1)" in answer
            or "①" in answer
            or "첫째" in answer
            or ("하나는" in answer and "다른 하나는" in answer)
        )

        return pattern_found or asks_multiple or enumerates_options

    def _detect_clarification_request(self, answer: str) -> bool:
        """Detect if system requested clarification."""
        clarification_indicators = [
            "어떤",
            "무엇을",
            "구체적",
            "명확히",
            "말씀해",
        ]
        return any(indicator in answer for indicator in clarification_indicators)

    def _check_context_preservation(self, context: Any, answer: str) -> bool:
        """
        Check if system preserved context from previous turns.

        Enhanced context preservation checking:
        - Semantic reference to previous topics
        - Pronoun resolution
        - Topic continuity
        - Entity reference
        """
        if (
            not hasattr(context, "conversation_history")
            or not context.conversation_history
        ):
            return True  # No previous context to preserve

        # Get previous turns for reference checking
        previous_turns = context.conversation_history[-3:]  # Last 3 turns

        # Check 1: Pronoun and reference word usage (indicates context awareness)
        context_markers = [
            "그",
            "이",
            "저",  # Demonstrative pronouns
            "위에서",
            "아까",
            "방금",
            "먼저",  # Temporal references
            "그다음",
            "그리고",
            "또한",  # Continuation markers
            "해당",
            "관련",
            "같은",  # Entity references
        ]
        has_context_marker = any(marker in answer for marker in context_markers)

        # Check 2: Topic continuity - answer references previous topics
        topic_continuity = False
        previous_topics = set()
        for turn in previous_turns:
            if hasattr(turn, "query") and turn.query:
                # Extract key topics (nouns, keywords)
                words = turn.query.split()
                previous_topics.update(words)

        # Check if answer contains any of the previous topics
        if previous_topics:
            topic_overlap = previous_topics & set(answer.split())
            topic_continuity = len(topic_overlap) > 0

        # Check 3: Semantic coherence indicators
        coherence_markers = [
            "따라서",
            "그러므로",  # Conclusion markers
            "추가로",
            "또한",  # Addition markers
            "마찬가지로",
            "비슷하게",  # Similarity markers
        ]
        has_coherence = any(marker in answer for marker in coherence_markers)

        # Check 4: Answer length should be reasonable for context-aware response
        # Too short (< 30 chars) suggests no context, too long (> 2000) might be generic
        reasonable_length = 30 <= len(answer) <= 2000

        # Context is preserved if multiple checks pass
        checks_passed = sum(
            [
                has_context_marker,
                topic_continuity,
                has_coherence,
                reasonable_length,
            ]
        )

        # Require at least 2 out of 4 checks to pass
        return checks_passed >= 2

    def _detect_empathy(self, answer: str) -> bool:
        """
        Detect if response shows empathy.

        Enhanced empathy detection:
        - Emotional acknowledgment
        - Supportive language
        - Understanding indicators
        - Concern expressions
        """
        # Strong empathy indicators (emotional acknowledgment)
        strong_empathy = [
            "이해합니다",
            "힘드시겠네요",
            "어려우시겠네요",
            "걱정되시는",
            "감정을 느낍니다",
            "마음이 쓰이네요",
        ]

        # Supportive language
        supportive = [
            "도와",
            "지원",
            "상담",
            "안내",
            "도움",
            "협력",
        ]

        # Understanding indicators
        understanding = [
            "이해",
            "알겠습니다",
            "그렇군요",
            "그러시군요",
            "공감",
        ]

        # Concern expressions
        concern = [
            "걱정",
            "어려움",
            "힘듦",
            "불안",
            " concerns",
        ]

        # Check for strong empathy (highest weight)
        has_strong_empathy = any(indicator in answer for indicator in strong_empathy)

        # Check for multiple categories of empathy (medium empathy)
        category_count = sum(
            [
                any(indicator in answer for indicator in supportive),
                any(indicator in answer for indicator in understanding),
                any(indicator in answer for indicator in concern),
            ]
        )

        # Empathy is detected if strong empathy present OR at least 2 categories
        return has_strong_empathy or category_count >= 2

    def _detect_contact_info(self, answer: str) -> bool:
        """
        Detect if response provides contact information.

        Enhanced contact info detection:
        - Phone numbers
        - Email addresses
        - Office locations
        - Contact references
        """
        # Direct contact indicators
        contact_indicators = [
            "연락처",
            "전화",
            "이메일",
            "방문",
            "상담",
            "담당자",
            "문의",
        ]

        # Pattern-based detection
        import re

        # Phone number pattern (Korean formats)
        phone_patterns = [
            r"\d{2,3}-\d{3,4}-\d{4}",  # 02-1234-5678 or 010-1234-5678
            r"\d{10,11}",  # 0212345678 or 01012345678
        ]

        # Email pattern
        email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"

        # Check for direct contact indicators
        has_contact_indicator = any(
            indicator in answer for indicator in contact_indicators
        )

        # Check for phone numbers
        has_phone = any(re.search(pattern, answer) for pattern in phone_patterns)

        # Check for email
        has_email = re.search(email_pattern, answer) is not None

        # Check for office/location indicators
        location_indicators = [
            "사무실",
            "부서",
            "층",
            "호",
            "빌딩",
            "위치",
        ]
        has_location = any(indicator in answer for indicator in location_indicators)

        # Contact info is present if any check passes
        return has_contact_indicator or has_phone or has_email or has_location

    def _detect_alternatives(self, answer: str) -> bool:
        """
        Detect if response offers alternatives.

        Enhanced alternatives detection:
        - Direct alternative language
        - Option enumeration
        - Conditional suggestions
        - Workaround proposals
        """
        # Direct alternative indicators
        alternative_indicators = [
            "또는",
            "또 다른",
            "대안",
            "다른 방법",
            "대신",
            "대체",
        ]

        # Option enumeration patterns
        option_patterns = [
            "1.",
            "2.",
            "3.",  # Numbered options
            "①",
            "②",
            "③",  # Circled numbers
            "첫째",
            "둘째",
            "셋째",  # Ordinal numbers
            "하나는",
            "다른 하나는",  # Contrastive patterns
        ]

        # Conditional suggestions
        conditional_patterns = [
            "만약",
            "경우에는",
            "할 수 없다면",
            "안 되면",
            "불가능하다면",
        ]

        # Workaround proposals
        workaround_patterns = [
            "대신",
            "다른 방법으로",
            "다른 경로로",
            "별도로",
            "추가적인",
        ]

        # Check all pattern categories
        has_alternative_indicator = any(
            indicator in answer for indicator in alternative_indicators
        )
        has_options = any(pattern in answer for pattern in option_patterns)
        has_conditional = any(pattern in answer for pattern in conditional_patterns)
        has_workaround = any(pattern in answer for pattern in workaround_patterns)

        # Alternatives are present if multiple checks pass
        checks_passed = sum(
            [
                has_alternative_indicator,
                has_options,
                has_conditional,
                has_workaround,
            ]
        )

        return checks_passed >= 2

    def _detect_escalation(self, answer: str) -> bool:
        """
        Detect if response escalates to human support.

        Enhanced escalation detection:
        - Human agent references
        - Direct contact suggestions
        - In-person meeting proposals
        - Special assistance offers
        """
        # Human agent indicators
        escalation_indicators = [
            "상담사",
            "담당자",
            "직원",
            "도와드리",
            "상담원",
            "전문가",
        ]

        # Direct contact suggestions
        contact_escalation = [
            "직접 방문",
            "상담 예약",
            "전화 상담",
            "대면 상담",
            "방문하시면",
        ]

        # Special assistance offers
        assistance_escalation = [
            "별도 안내",
            "개별 상담",
            "전문 지원",
            "특별 도움",
        ]

        # Urgent escalation phrases
        urgent_escalation = [
            "즉시 연락",
            "긴급 연락처",
            "바로 상담",
            "당장 도와",
        ]

        # Check for human agent reference
        has_human_agent = any(
            indicator in answer for indicator in escalation_indicators
        )
        has_contact_escalation = any(
            indicator in answer for indicator in contact_escalation
        )
        has_assistance_escalation = any(
            indicator in answer for indicator in assistance_escalation
        )
        has_urgent_escalation = any(
            indicator in answer for indicator in urgent_escalation
        )

        # Escalation is detected if human agent is mentioned OR
        # multiple escalation categories are present
        checks_passed = sum(
            [
                has_human_agent,
                has_contact_escalation,
                has_assistance_escalation,
                has_urgent_escalation,
            ]
        )

        return has_human_agent or checks_passed >= 2

    def generate_session_report(self, session: ComprehensiveTestSession) -> str:
        """
        Generate a comprehensive test session report.

        Args:
            session: The test session to report on.

        Returns:
            Formatted report string.
        """
        report_lines = []
        report_lines.append("# Comprehensive RAG Test Report")
        report_lines.append(f"\n**Session ID**: {session.session_id}")
        report_lines.append(f"**Started**: {session.started_at.isoformat()}")
        report_lines.append(
            f"**Completed**: {session.completed_at.isoformat() if session.completed_at else 'N/A'}"
        )

        report_lines.append("\n## Summary")
        report_lines.append(f"- **Total Tests**: {session.total_tests}")
        report_lines.append(f"- **Passed**: {session.passed_tests}")
        report_lines.append(f"- **Failed**: {session.failed_tests}")
        report_lines.append(f"- **Pass Rate**: {session.pass_rate:.1%}")

        # Ambiguous query results
        if session.ambiguous_results:
            report_lines.append("\n## Ambiguous Query Results")
            passed = sum(1 for r in session.ambiguous_results if r.get("overall_pass"))
            report_lines.append(f"- **Total**: {len(session.ambiguous_results)}")
            report_lines.append(f"- **Passed**: {passed}")
            report_lines.append(
                f"- **Failed**: {len(session.ambiguous_results) - passed}"
            )

        # Multi-turn results
        if session.multi_turn_results:
            report_lines.append("\n## Multi-Turn Conversation Results")
            passed = sum(1 for r in session.multi_turn_results if r.get("overall_pass"))
            report_lines.append(f"- **Total**: {len(session.multi_turn_results)}")
            report_lines.append(f"- **Passed**: {passed}")
            report_lines.append(
                f"- **Failed**: {len(session.multi_turn_results) - passed}"
            )

            # Average context preservation
            avg_context = sum(
                r.get("context_preservation_rate", 0)
                for r in session.multi_turn_results
            ) / len(session.multi_turn_results)
            report_lines.append(f"- **Avg Context Preservation**: {avg_context:.1%}")

        # Edge case results
        if session.edge_case_results:
            report_lines.append("\n## Edge Case Results")
            passed = sum(1 for r in session.edge_case_results if r.get("overall_pass"))
            report_lines.append(f"- **Total**: {len(session.edge_case_results)}")
            report_lines.append(f"- **Passed**: {passed}")
            report_lines.append(
                f"- **Failed**: {len(session.edge_case_results) - passed}"
            )

            # Empathy detection
            empathy_count = sum(
                1 for r in session.edge_case_results if r.get("showed_empathy")
            )
            report_lines.append(
                f"- **Empathy Shown**: {empathy_count}/{len(session.edge_case_results)}"
            )

        return "\n".join(report_lines)
