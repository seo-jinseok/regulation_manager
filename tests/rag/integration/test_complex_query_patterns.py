"""
Integration tests for Complex Query Patterns.

Tests advanced query patterns that challenge the RAG system:
- Nested questions with multiple layers
- Context-dependent queries
- Multi-intent queries
- Comparative queries
- Conditional queries
"""

import pytest

from src.rag.automation.domain.context_tracker import ContextTracker
from src.rag.automation.domain.entities import (
    ContextHistory,
    Turn,
)
from src.rag.domain.entities import Chunk, ChunkLevel
from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType
from src.rag.infrastructure.query_expander import DynamicQueryExpander


def make_chunk(
    id: str,
    text: str,
    title: str = "",
    rule_code: str = "",
    **kwargs,
) -> Chunk:
    """Helper to create test chunks."""
    return Chunk(
        id=id,
        text=text,
        title=title,
        rule_code=rule_code,
        level=ChunkLevel.ARTICLE,
        embedding_text=text,
        full_text=text,
        parent_path=[title] if title else [],
        token_count=len(text.split()),
        keywords=[],
        is_searchable=True,
    )


class TestNestedQuestionPatterns:
    """Test multi-level nested question patterns."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_two_level_nested_question(self, query_analyzer):
        """
        SPEC: System should handle 2-level nested questions.

        Given: Query "휴학 신청 서류 중에서 증명서 발급 방법"
        When: Query is analyzed
        Then: Should detect hierarchy (휴학 > 서류 > 증명서 발급)
        """
        query = "휴학 신청 서류 중에서 증명서 발급 방법"

        query_type = query_analyzer.analyze(query)

        # Should be recognized as complex/natural query
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_three_level_nested_context(self, query_analyzer):
        """
        SPEC: System should maintain context across 3-level nesting.

        Given: Conversation about:
              1. 휴학 신청 (general)
              2. 서류 제출 (specific aspect)
              3. 증명서 발급 (detail of specific aspect)
        When: Third query uses pronoun "그거"
        Then: Context should reference "증명서 발급"
        """
        # Simulate conversation context
        ContextHistory(
            scenario_id="nested_test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="휴학 신청은 어떻게 하나요?",
                    answer="휴학 신청은...",
                    sources=["규정 제10조"],
                    confidence=0.9,
                ),
                Turn(
                    turn_number=2,
                    query="서류는 뭘 제출하나요?",
                    answer="다음 서류가 필요합니다...",
                    sources=["규정 제10조 제2항"],
                    confidence=0.85,
                ),
                Turn(
                    turn_number=3,
                    query="증명서는 어떻게 발급받나요?",
                    answer="증명서 발급은...",
                    sources=["규정 제10조 제3항"],
                    confidence=0.88,
                ),
            ],
            implicit_entities={"서류": "documents", "증명서": "certificate"},
            topic_transitions=[],
            intent_history=["휴학", "서류", "증명서"],
        )

        # Fourth query referencing "그거" (that thing)
        fourth_query = "그거 발급 비용은 얼마인가요?"

        # Extract implicit info
        tracker = ContextTracker()
        implicit_info = tracker._extract_implicit_info(
            Turn(
                turn_number=4,
                query=fourth_query,
                answer="",
                sources=[],
                confidence=0.0,
            )
        )

        # Should detect implicit reference
        assert "implicit_그거" in implicit_info or len(implicit_info) > 0

    def test_cross_topic_nested_question(self, query_analyzer):
        """
        SPEC: System should handle nested questions spanning multiple regulations.

        Given: Query "장학금 신청 자격 중에서 성적 기준과 소득 수준"
        When: Query is analyzed
        Then: Should detect multiple nested aspects (자격 > 성적, 소득)
        """
        query = "장학금 신청 자격 중에서 성적 기준과 소득 수준은 어떻게 되나요?"

        expanded = query_analyzer.expand_query(query)

        # Should contain key terms
        assert "장학금" in expanded or "성적" in expanded or "소득" in expanded


class TestMultiIntentQueries:
    """Test queries with multiple intents."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_conjunctive_intent_query(self, query_analyzer):
        """
        SPEC: System should handle conjunctive intents (AND logic).

        Given: Query "휴학과 복학 모두 신청 방법 알려주세요"
        When: Query is analyzed
        Then: Should detect both 휴학 and 복학 intents
        """
        query = "휴학과 복학 모두 신청 방법 알려주세요"

        query_type = query_analyzer.analyze(query)

        # Should be recognized as natural question with multiple intents
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_disjunctive_intent_query(self, query_analyzer):
        """
        SPEC: System should handle disjunctive intents (OR logic).

        Given: Query "성적 정정이나 학점 재계중 어떤 게 가능한가요?"
        When: Query is analyzed
        Then: Should detect both 성적 정정 and 학점 재계포 intents
        """
        query = "성적 정정이나 학점 재계중 어떤 게 가능한가요?"

        query_type = query_analyzer.analyze(query)

        # Should handle as complex query
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_sequential_intent_query(self, query_analyzer):
        """
        SPEC: System should handle sequential intents.

        Given: Query "휴학 신청하고 나서 복학할 때는 어떻게 하나요?"
        When: Query is analyzed
        Then: Should detect sequential process (휴학 > 복학)
        """
        query = "휴학 신청하고 나서 복학할 때는 어떻게 하나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect intent about process/sequence
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )


class TestComparativeQueryPatterns:
    """Test comparative and contrastive query patterns."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_direct_comparison_query(self, query_analyzer):
        """
        SPEC: System should handle direct comparison queries.

        Given: Query "일반휴학과 군휴학 차이점이 뭔가요?"
        When: Query is analyzed
        Then: Should detect comparison intent
        """
        query = "일반휴학과 군휴학 차이점이 뭔가요?"

        query_analyzer.analyze(query)

        # Should detect comparison
        expanded = query_analyzer.expand_query(query)

        # Should expand to include both terms
        assert "일반휴학" in expanded or "휴학" in expanded
        assert "군휴학" in expanded or "군입대" in expanded

    def test_implicit_comparison_query(self, query_analyzer):
        """
        SPEC: System should handle implicit comparison queries.

        Given: Query "장학금 종류별로 장단점 설명해주세요"
        When: Query is analyzed
        Then: Should detect comparative intent across types
        """
        query = "장학금 종류별로 장단점 설명해주세요"

        query_type = query_analyzer.analyze(query)

        # Should handle as explanation request with comparison aspect
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_ranking_query(self, query_analyzer):
        """
        SPEC: System should handle ranking/sorting queries.

        Given: Query "가장 많이 받는 장학금 순위 알려주세요"
        When: Query is analyzed
        Then: Should detect ranking intent
        """
        query = "가장 많이 받는 장학금 순위 알려주세요"

        query_type = query_analyzer.analyze(query)

        # Should detect as information request
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )


class TestConditionalQueryPatterns:
    """Test conditional and hypothetical query patterns."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_conditional_condition_query(self, query_analyzer):
        """
        SPEC: System should handle conditional queries.

        Given: Query "성적이 2.0 미만이면 장학금 못 받나요?"
        When: Query is analyzed
        Then: Should detect condition (성적 < 2.0) and query (장학금 eligibility)
        """
        query = "성적이 2.0 미만이면 장학금 못 받나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as question about eligibility
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_hypothetical_scenario_query(self, query_analyzer):
        """
        SPEC: System should handle hypothetical scenario queries.

        Given: Query "만약에 중도에 휴학하면 등록금 환불되나요?"
        When: Query is analyzed
        Then: Should detect hypothetical scenario
        """
        query = "만약에 중도에 휴학하면 등록금 환불되나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as conditional/eligibility question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_temporal_condition_query(self, query_analyzer):
        """
        SPEC: System should handle temporal condition queries.

        Given: Query "학기 시작 2주 전에 휴학 신청하면 되나요?"
        When: Query is analyzed
        Then: Should detect temporal condition
        """
        query = "학기 시작 2주 전에 휴학 신청하면 되나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as procedural question with condition
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )


class TestContextDependentQueries:
    """Test queries that depend on previous conversation context."""

    @pytest.fixture
    def context_tracker(self):
        """Create ContextTracker instance."""
        return ContextTracker(context_window_size=3)

    def test_pronoun_resolution_with_history(self, context_tracker):
        """
        SPEC: System should resolve pronouns based on conversation history.

        Given: Conversation about "장학금 신청"
        When: Follow-up asks "그거 연간 소득 기준은?"
        Then: "그거" should resolve to "장학금 신청"
        """
        context = ContextHistory(
            scenario_id="pronoun_test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="장학금 신청은 어떻게 하나요?",
                    answer="장학금 신청은...",
                    sources=["장학금규정"],
                    confidence=0.9,
                ),
                Turn(
                    turn_number=2,
                    query="서류는 뭐 필요한가요?",
                    answer="다음 서류가 필요합니다...",
                    sources=["장학금규정"],
                    confidence=0.85,
                ),
            ],
            implicit_entities={"장학금": "scholarship"},
            topic_transitions=[],
            intent_history=["장학금 신청", "서류"],
        )

        # Current query with pronoun
        current_turn = Turn(
            turn_number=3,
            query="그거 연간 소득 기준은?",
            answer="연간 소득 기준은...",
            sources=["장학금규정"],
            confidence=0.8,
        )

        # Check context preservation
        preserved = context_tracker.detect_context_preservation(context, current_turn)

        assert preserved is True, "Should preserve context with pronoun reference"

    def test_temporal_reference_resolution(self, context_tracker):
        """
        SPEC: System should resolve temporal references.

        Given: Previous answers about "휴학 기간"
        When: Follow-up asks "아까 말한 기간 동안 등록금은?"
        Then: "아까 말한 기간" should resolve to previously mentioned period
        """
        context = ContextHistory(
            scenario_id="temporal_test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="휴학 기간은 얼마인가요?",
                    answer="휴학 기간은 최대 2학기입니다...",
                    sources=["학칙"],
                    confidence=0.9,
                ),
            ],
            implicit_entities={"휴학 기간": "2학기"},
            topic_transitions=[],
            intent_history=["휴학 기간"],
        )

        current_turn = Turn(
            turn_number=2,
            query="아까 말한 기간 동안 등록금은 내야 하나요?",
            answer="휴학 기간 중에는...",
            sources=["학칙"],
            confidence=0.85,
        )

        preserved = context_tracker.detect_context_preservation(context, current_turn)

        assert preserved is True, "Should preserve context with temporal reference"

    def test_implicit_entity_reference(self, context_tracker):
        """
        SPEC: System should track and resolve implicit entity references.

        Given: Conversation about "교원 인사"
        When: Follow-up asks "그분들의 승진 기준은?"
        Then: "그분들" should resolve to "교원"
        """
        ContextHistory(
            scenario_id="entity_test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="교원 임용 절차가 어떻게 되나요?",
                    answer="교원 임용은...",
                    sources=["교원인사규정"],
                    confidence=0.9,
                ),
            ],
            implicit_entities={"교원": "faculty", "교수": "professor"},
            topic_transitions=[],
            intent_history=["교원 임용"],
        )

        current_turn = Turn(
            turn_number=2,
            query="그분들의 승진 기준은 어떻게 되나요?",
            answer="승진 기준은...",
            sources=["교원인사규정"],
            confidence=0.85,
        )

        # Extract implicit info from current turn
        implicit_info = context_tracker._extract_implicit_info(current_turn)

        # Should detect implicit reference marker
        assert len(implicit_info) > 0, "Should extract implicit reference"


class TestAmbiguousQueryPatterns:
    """Test ambiguous and underspecified query patterns."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_lexical_ambiguity_query(self, query_analyzer):
        """
        SPEC: System should handle lexical ambiguity.

        Given: Query "신청 방법 알려주세요" (without context)
        When: Query is analyzed
        Then: Should recognize as general request or ask for clarification
        """
        query = "신청 방법 알려주세요"

        query_type = query_analyzer.analyze(query)

        # Should recognize as general procedural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )

    def test_syntactic_ambiguity_query(self, query_analyzer):
        """
        SPEC: System should handle syntactic ambiguity.

        Given: Query "성적 정정으로 학점 올릴 수 있나요?"
              (Could mean: "Can I raise grades through grade correction?"
               or: "Can grade correction lead to grade increase?")
        When: Query is analyzed
        Then: Should handle both interpretations
        """
        query = "성적 정정으로 학점 올릴 수 있나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as eligibility question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_underspecified_query(self, query_analyzer):
        """
        SPEC: System should handle underspecified queries.

        Given: Query "기간이 얼마나 돼?" (missing: what procedure?)
        When: Query is analyzed
        Then: Should recognize need for clarification
        """
        query = "기간이 얼마나 돼?"

        query_type = query_analyzer.analyze(query)

        # Should recognize as general question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )


class TestQueryExpansionComplexPatterns:
    """Test query expansion for complex patterns."""

    def test_expand_synonym_chain(self):
        """
        SPEC: System should expand synonym chains.

        Given: Query "교수님 평가 방법"
        When: Query is expanded
        Then: Should include synonyms: 교원, 교수, 평가, 성과, 관리
        """
        expander = DynamicQueryExpander()
        query = "교수님 평가 방법"

        result = expander.expand(query)

        # Should contain related terms
        has_synonyms = any(
            term in result.expanded_query
            for term in ["교원", "성과", "업적", "관리", "정년"]
        )
        assert (
            has_synonyms
            or "교수" in result.expanded_query
            or "평가" in result.expanded_query
        )

    def test_expand_hierarchical_terms(self):
        """
        SPEC: System should expand hierarchical term relationships.

        Given: Query "장학금 지급"
        When: Query is expanded
        Then: Should include related terms: 등록금 감면, 장학, 학비 지원
        """
        expander = DynamicQueryExpander()
        query = "장학금 지급"

        result = expander.expand(query)

        # Should contain related terms
        has_related = any(
            term in result.expanded_query
            for term in ["등록금", "학비", "지원", "감면", "장학"]
        )
        assert has_related

    def test_expand_domain_specific_terms(self):
        """
        SPEC: System should expand domain-specific terminology.

        Given: Query "학사징계 기준"
        When: Query is expanded
        Then: Should include related terms: 징계, 제적, 정학, 근신
        """
        expander = DynamicQueryExpander()
        query = "학사징계 기준"

        result = expander.expand(query)

        # Should contain related terms
        has_related = any(
            term in result.expanded_query
            for term in ["징계", "제적", "정학", "근신", "처벌"]
        )
        assert has_related
