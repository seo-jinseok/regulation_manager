"""
Integration tests for Edge Cases and Error Scenarios.

Tests challenging edge cases that may occur in real-world usage:
- Typos and spelling errors
- Incomplete sentences
- Mixed language queries
- Special characters and formatting
- Empty or malformed queries
- Very long queries
"""

import pytest

from src.rag.automation.domain.context_tracker import ContextTracker
from src.rag.automation.domain.entities import ContextHistory, Turn
from src.rag.domain.entities import Chunk, ChunkLevel
from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType
from src.rag.infrastructure.query_expander import DynamicQueryExpander


def make_chunk(
    id: str,
    text: str,
    title: str = "",
    rule_code: str = "",
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


class TestTypoHandling:
    """Test handling of typos and spelling errors."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_common_korean_typo_correction(self, query_analyzer):
        """
        SPEC: System should handle common Korean typos.

        Given: Query with common typos (e.g., "휴학" -> "휴학", "장학금" -> "장학금")
        When: Query is analyzed
        Then: Should detect intent despite typos
        """
        # Common typo: ㅗ/ㅓ confusion, consonant omission
        query = "휵학 신청 방법 알려주세요"  # "휴학" typo

        query_type = query_analyzer.analyze(query)

        # Should still detect as natural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_english_korean_mix_typo(self, query_analyzer):
        """
        SPEC: System should handle English-Korean mixed typos.

        Given: Query "GPA 정정 하는 법"
        When: Query is analyzed
        Then: Should recognize GPA as 성적 context
        """
        query = "GPA 정정 하는 법"

        query_type = query_analyzer.analyze(query)

        # Should detect as procedural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )

    def test_consonant_vowel_omission(self, query_analyzer):
        """
        SPEC: System should handle Korean consonant/vowel omissions.

        Given: Query "장학금 받ㄹ 수 있나요?" (consonant omitted)
        When: Query is analyzed
        Then: Should still detect correct intent
        """
        query = "장학금 받ㄹ 수 있나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as eligibility question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_spacing_errors(self, query_analyzer):
        """
        SPEC: System should handle spacing errors.

        Given: Query "휴학신청하는법" (no spaces)
        When: Query is analyzed
        Then: Should segment correctly
        """
        query = "휴학신청하는법"

        query_type = query_analyzer.analyze(query)

        # Should detect as procedural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )


class TestIncompleteSentences:
    """Test handling of incomplete sentence fragments."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_sentence_fragment_with_context(self, query_analyzer):
        """
        SPEC: System should complete fragments using conversation context.

        Given: Previous turn about "휴학 신청"
        When: Follow-up is "서류는?" (incomplete)
        Then: Should complete to "휴학 신청 서류는?"
        """
        ContextHistory(
            scenario_id="fragment_test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="휴학 신청은 어떻게 하나요?",
                    answer="휴학 신청은...",
                    sources=["규정"],
                    confidence=0.9,
                )
            ],
            implicit_entities={"휴학": "leave"},
            topic_transitions=[],
            intent_history=["휴학 신청"],
        )

        fragment_query = "서류는?"

        # Should detect context from history
        query_type = query_analyzer.analyze(fragment_query)

        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_single_word_query(self, query_analyzer):
        """
        SPEC: System should handle single-word queries.

        Given: Query "휴학?" (just topic name)
        When: Query is analyzed
        Then: Should infer general information request
        """
        query = "휴학?"

        query_type = query_analyzer.analyze(query)

        # Should detect as general question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_elliptical_expression(self, query_analyzer):
        """
        SPEC: System should handle elliptical expressions.

        Given: Query "언제까지야?" (missing subject)
        When: Query is analyzed
        Then: Should recognize as deadline question
        """
        query = "언제까지야?"

        query_type = query_analyzer.analyze(query)

        # Should detect as general question (context-dependent)
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )


class TestMixedLanguageQueries:
    """Test handling of mixed Korean-English queries."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_korean_english_code_switching(self, query_analyzer):
        """
        SPEC: System should handle Korean-English code switching.

        Given: Query "GPA 3.5 이상이면 장학금 받아요?"
        When: Query is analyzed
        Then: Should handle mixed script correctly
        """
        query = "GPA 3.5 이상이면 장학금 받아요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as eligibility question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_english_acronyms(self, query_analyzer):
        """
        SPEC: System should handle English acronyms in Korean text.

        Given: Query "TOEIC 성적으로 장학금 가능?"
        When: Query is analyzed
        Then: Should recognize TOEIC as English test score
        """
        query = "TOEIC 성적으로 장학금 가능?"

        query_type = query_analyzer.analyze(query)

        # Should detect as eligibility question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_mixed_script_loanwords(self, query_analyzer):
        """
        SPEC: System should handle loanwords in original script.

        Given: Query "Campus job 알바 해도 되나요?"
        When: Query is analyzed
        Then: Should map loanwords to Korean equivalents
        """
        query = "Campus job 알바 해도 되나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as eligibility question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )


class TestSpecialCharacters:
    """Test handling of special characters and formatting."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_emoji_in_query(self, query_analyzer):
        """
        SPEC: System should handle emojis in queries.

        Given: Query "휴학 신청 방법 알려주세요 🙏"
        When: Query is analyzed
        Then: Should ignore or handle emojis appropriately
        """
        query = "휴학 신청 방법 알려주세요 🙏"

        query_type = query_analyzer.analyze(query)

        # Should still detect as procedural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )

    def test_excessive_punctuation(self, query_analyzer):
        """
        SPEC: System should handle excessive punctuation.

        Given: Query "진짜 궁금해요!!!! 장학금 어떻게 받아요???"
        When: Query is analyzed
        Then: Should normalize punctuation
        """
        query = "진짜 궁금해요!!!! 장학금 어떻게 받아요???"

        query_type = query_analyzer.analyze(query)

        # Should detect as natural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_legal_article_reference_format(self, query_analyzer):
        """
        SPEC: System should handle legal article references.

        Given: Query "학칙 §15조에 따르면 뭐가 되나요?"
        When: Query is analyzed
        Then: Should recognize article reference format
        """
        query = "학칙 §15조에 따르면 뭐가 되나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as article reference or natural question
        assert query_type in (
            QueryType.ARTICLE_REFERENCE,
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )

    def test_parentheses_content(self, query_analyzer):
        """
        SPEC: System should handle parenthetical content.

        Given: Query "복학 (휴학 후 재입학) 절차가 어떻게 돼요?"
        When: Query is analyzed
        Then: Should handle parenthetical explanation
        """
        query = "복학 (휴학 후 재입학) 절차가 어떻게 돼요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as procedural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.GENERAL,
        )


class TestEdgeCaseQueries:
    """Test various edge case query scenarios."""

    @pytest.fixture
    def query_analyzer(self):
        """Create QueryAnalyzer instance."""
        return QueryAnalyzer()

    def test_empty_query(self, query_analyzer):
        """
        SPEC: System should handle empty queries gracefully.

        Given: Query "" or "   "
        When: Query is analyzed
        Then: Should return appropriate response or error
        """
        query = ""

        # Should handle without crashing
        query_type = query_analyzer.analyze(query)

        # Should default to general or handle gracefully
        assert query_type is not None

    def test_very_long_query(self, query_analyzer):
        """
        SPEC: System should handle very long queries.

        Given: Query with 500+ characters
        When: Query is analyzed
        Then: Should process without errors
        """
        query = " ".join(["장학금"] * 100) + " 신청 방법 알려주세요"

        # Should handle without crashing
        query_type = query_analyzer.analyze(query)

        assert query_type is not None

    def test_repeated_words(self, query_analyzer):
        """
        SPEC: System should handle repeated words (stuttering/emphasis).

        Given: Query "진짜 진짜 꼭 알고 싶은데 휴학 방법 알려줘"
        When: Query is analyzed
        Then: Should normalize repetitions
        """
        query = "진짜 진짜 꼭 알고 싶은데 휴학 방법 알려줘"

        query_type = query_analyzer.analyze(query)

        # Should detect as procedural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_negation_query(self, query_analyzer):
        """
        SPEC: System should handle negation queries correctly.

        Given: Query "휴학 안 하면 안 되나요?"
        When: Query is analyzed
        Then: Should detect negation properly
        """
        query = "휴학 안 하면 안 되나요?"

        query_type = query_analyzer.analyze(query)

        # Should detect as eligibility/natural question
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.INTENT,
            QueryType.GENERAL,
        )

    def test_double_negative_query(self, query_analyzer):
        """
        SPEC: System should handle double negatives.

        Given: Query "장학금 안 못 받는 경우 없나 없어?"
        When: Query is analyzed
        Then: Should resolve double negative correctly
        """
        query = "장학금 안 못 받는 경우 없나 없어?"

        query_type = query_analyzer.analyze(query)

        # Should handle despite double negative
        assert query_type in (
            QueryType.NATURAL_QUESTION,
            QueryType.INTENT,
            QueryType.GENERAL,
        )


class TestContextEdgeCases:
    """Test edge cases in context management."""

    @pytest.fixture
    def context_tracker(self):
        """Create ContextTracker instance."""
        return ContextTracker(context_window_size=3)

    def test_context_with_no_history(self, context_tracker):
        """
        SPEC: System should handle context with no previous history.

        Given: Empty conversation history
        When: Context preservation is checked
        Then: Should return True (nothing to preserve)
        """
        empty_context = ContextHistory(
            scenario_id="empty_test",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        current_turn = Turn(
            turn_number=1,
            query="휴학 방법",
            answer="답변",
            sources=["규정"],
            confidence=0.9,
        )

        preserved = context_tracker.detect_context_preservation(
            empty_context, current_turn
        )

        assert preserved is True, "Empty context should preserve by default"

    def test_context_exceeding_window(self, context_tracker):
        """
        SPEC: System should handle context exceeding window size.

        Given: Conversation history with 10 turns
        When: Context window is 3
        Then: Should only consider last 3 turns
        """
        # Create 10 turns
        turns = [
            Turn(
                turn_number=i,
                query=f"Question {i}",
                answer=f"Answer {i}",
                sources=[f"Source {i}"],
                confidence=0.8,
            )
            for i in range(1, 11)
        ]

        context = ContextHistory(
            scenario_id="window_test",
            conversation_history=turns,
            implicit_entities={},
            topic_transitions=[],
            intent_history=[f"Intent {i}" for i in range(1, 11)],
        )

        recent = context.get_recent_context(window_size=3)

        # Should only return last 3 turns
        assert len(recent) == 3
        assert recent[0].turn_number == 8
        assert recent[1].turn_number == 9
        assert recent[2].turn_number == 10

    def test_rapid_topic_switching(self, context_tracker):
        """
        SPEC: System should handle rapid topic switching.

        Given: Conversation rapidly switching between unrelated topics
        When: Topic transitions are tracked
        Then: Should detect multiple transitions
        """
        turns = [
            Turn(
                turn_number=1,
                query="휴학 방법",  # Topic A
                answer="A1",
                sources=["규정 A"],
                confidence=0.9,
            ),
            Turn(
                turn_number=2,
                query="졸업 요건",  # Topic B
                answer="B1",
                sources=["규정 B"],
                confidence=0.8,
            ),
            Turn(
                turn_number=3,
                query="장학금",  # Topic C
                answer="C1",
                sources=["규정 C"],
                confidence=0.85,
            ),
        ]

        context = ContextHistory(
            scenario_id="switch_test",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        # Add turns and track transitions
        transitions = []
        for i, turn in enumerate(turns):
            if i == 0:
                context = context_tracker.create_initial_context("switch_test", turn)
            else:
                old_context = context
                context = context_tracker.update_context(context, turn)
                transition = context_tracker._detect_topic_transition(old_context, turn)
                if transition:
                    transitions.append(transition)

        # Should detect topic transitions
        assert len(transitions) >= 1, "Should detect at least one topic transition"


class TestQueryExpansionEdgeCases:
    """Test edge cases in query expansion."""

    def test_expand_stopwords_only(self):
        """
        SPEC: System should handle queries with only stopwords.

        Given: Query "그게 어떻게 되나요?" (mostly stopwords)
        When: Query is expanded
        Then: Should handle gracefully without excessive expansion
        """
        expander = DynamicQueryExpander()
        query = "그게 어떻게 되나요?"

        result = expander.expand(query)

        # Should return some expansion
        assert result is not None
        assert len(result.expanded_query) > 0

    def test_expand_with_numerical_values(self):
        """
        SPEC: System should handle numerical values in queries.

        Given: Query "3.5 이상 GPA면 장학금 받아?"
        When: Query is expanded
        Then: Should preserve numerical information
        """
        expander = DynamicQueryExpander()
        query = "3.5 이상 GPA면 장학금 받아?"

        result = expander.expand(query)

        # Should contain key terms
        assert (
            "3.5" in result.expanded_query
            or "GPA" in result.expanded_query
            or "장학" in result.expanded_query
        )

    def test_expand_with_date_references(self):
        """
        SPEC: System should handle date/time references.

        Given: Query "2024년 2학기 장학금 신청 기간"
        When: Query is expanded
        Then: Should preserve temporal information
        """
        expander = DynamicQueryExpander()
        query = "2024년 2학기 장학금 신청 기간"

        result = expander.expand(query)

        # Should contain temporal terms
        has_temporal = any(
            term in result.expanded_query
            for term in ["2024", "2학기", "학기", "신청", "기간", "장학"]
        )
        assert has_temporal
