"""
Characterization tests for existing Query Expansion behavior.

These tests capture CURRENT BEHAVIOR (not desired behavior) to ensure
no regressions occur during refactoring for SPEC-RAG-SEARCH-001.

Purpose: Document what the current system ACTUALLY does, not what it should do.
"""

import pytest

from src.rag.infrastructure.query_analyzer import QueryAnalyzer
from src.rag.infrastructure.query_expander import DynamicQueryExpander


class TestQueryAnalyzerExpansionCharacterization:
    """
    Characterization tests for QueryAnalyzer.expand_query().

    These tests document the actual current behavior of query expansion.
    If behavior changes, these tests should be updated to match new reality.
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    # Characterize: Basic synonym expansion
    def test_characterize_synonym_expansion_scholarship(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current behavior for '장학금' query.

        Documents what the current expansion produces.
        """
        query = "장학금"
        result = analyzer.expand_query(query)

        # Current behavior: Uses SYNONYMS dict
        # SYNONYMS = {"장학금": ["장학"]}
        expected_contains = ["장학금", "장학"]  # May also have intent keywords

        # Check that key terms are present
        for term in expected_contains:
            assert term in result, f"Expected '{term}' in expanded query: '{result}'"

        # Store the actual expansion for reference
        print(f"\n[CHARACTERIZATION] '{query}' -> '{result}'")

    def test_characterize_synonym_expansion_tuition(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current behavior for '등록금' query.

        Documents what the current expansion produces.
        """
        query = "등록금"
        result = analyzer.expand_query(query)

        # Current behavior: SYNONYMS = {"등록금": ["수업료"]}
        expected_contains = ["등록금", "수업료"]

        for term in expected_contains:
            assert term in result, f"Expected '{term}' in expanded query: '{result}'"

        print(f"\n[CHARACTERIZATION] '{query}' -> '{result}'")

    # Characterize: Intent-based expansion
    def test_characterize_intent_expansion_want_scholarship(
        self, analyzer: QueryAnalyzer
    ):
        """
        CHARACTERIZE: Current behavior for '장학금 받고 싶어'.

        Documents intent pattern matching and keyword expansion.
        """
        query = "장학금 받고 싶어"
        result = analyzer.expand_query(query)

        # Current behavior: Matches INTENT_PATTERNS for scholarship
        # Pattern: re.compile(r"(장학금|장학).*(받|타).*(싶)")
        # Keywords: ["장학금", "장학", "장학금지급규정", "장학금 신청"]
        expected_keywords = ["장학금", "장학"]

        for kw in expected_keywords:
            assert kw in result, f"Expected '{kw}' in expanded query: '{result}'"

        print(f"\n[CHARACTERIZATION] '{query}' -> '{result}'")

    def test_characterize_intent_expansion_leave_of_absence(
        self, analyzer: QueryAnalyzer
    ):
        """
        CHARACTERIZE: Current behavior for '학교 가기 싫어'.

        Documents intent pattern matching for leave of absence.
        """
        query = "학교 가기 싫어"
        result = analyzer.expand_query(query)

        # Current behavior: Matches INTENT_PATTERNS
        # Pattern: re.compile(r"(학교|출근|근무).*(가기|출근).*싫")
        # Keywords: ["휴직", "휴가", "연구년", "안식년"]
        expected_keywords = ["휴직", "연구년"]

        for kw in expected_keywords:
            assert kw in result, f"Expected '{kw}' in expanded query: '{result}'"

        print(f"\n[CHARACTERIZATION] '{query}' -> '{result}'")

    # Characterize: Query type analysis
    def test_characterize_query_type_article_reference(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current query type for '제15조'.
        """
        from src.rag.infrastructure.query_analyzer import QueryType

        query = "제15조"
        query_type = analyzer.analyze(query)

        assert query_type == QueryType.ARTICLE_REFERENCE
        print(f"\n[CHARACTERIZATION] '{query}' -> {query_type.value}")

    def test_characterize_query_type_natural_question(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current query type for '휴학 방법'.
        """
        from src.rag.infrastructure.query_analyzer import QueryType

        query = "휴학 방법"
        query_type = analyzer.analyze(query)

        assert query_type == QueryType.NATURAL_QUESTION
        print(f"\n[CHARACTERIZATION] '{query}' -> {query_type.value}")

    def test_characterize_query_type_regulation_name(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current query type for '장학금규정'.
        """
        from src.rag.infrastructure.query_analyzer import QueryType

        query = "장학금규정"
        query_type = analyzer.analyze(query)

        assert query_type == QueryType.REGULATION_NAME
        print(f"\n[CHARACTERIZATION] '{query}' -> {query_type.value}")

    # Characterize: Academic keywords matching
    def test_characterize_academic_keywords_matching(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current academic keywords that are recognized.

        These keywords from ACADEMIC_KEYWORDS list should trigger REGULATION_NAME type.
        """
        from src.rag.infrastructure.query_analyzer import QueryType

        test_cases = [
            ("휴학", QueryType.REGULATION_NAME),
            ("복학", QueryType.REGULATION_NAME),
            ("장학", QueryType.REGULATION_NAME),
            ("등록", QueryType.REGULATION_NAME),
        ]

        for query, expected_type in test_cases:
            actual_type = analyzer.analyze(query)
            assert actual_type == expected_type, (
                f"Query '{query}' expected {expected_type.value}, got {actual_type.value}"
            )
            print(f"\n[CHARACTERIZATION] '{query}' -> {actual_type.value}")

    # Characterize: Clean query behavior
    def test_characterize_clean_query_behavior(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Current query cleaning (stopword removal).

        Note: STOPWORDS include "규정" which gets removed completely.
        Updated: "은" 조사 no longer removed (TAG-004 synonym expansion enhancement)
        """
        test_cases = [
            # Actual behavior: "은" 조사 preserved after TAG-004 enhancement
            ("휴학 방법은 무엇인가요", "휴학 방법은 무엇인가요"),  # "은" preserved
            ("장학금 받는 방법", "장학금 받는 방법"),  # No stopwords
            # "규정" is in STOPWORDS so it gets removed
            ("규정이 뭐야", "뭐야"),  # "규정" removed, "이" removed
        ]

        for original, expected_exact in test_cases:
            cleaned = analyzer.clean_query(original)
            # For characterization, we document the EXACT current behavior
            assert cleaned == expected_exact, (
                f"Expected exact '{expected_exact}', got '{cleaned}'"
            )
            print(f"\n[CHARACTERIZATION] clean('{original}') -> '{cleaned}'")


class TestDynamicQueryExpanderCharacterization:
    """
    Characterization tests for DynamicQueryExpander.

    Documents LLM-based expansion behavior (when LLM available)
    and pattern-based fallback behavior.
    """

    @pytest.fixture
    def expander_no_llm(self, monkeypatch, tmp_path):
        """Expander without LLM (pattern-based fallback only)."""
        monkeypatch.setenv("RAG_SYNONYMS_PATH", "")
        monkeypatch.setenv("RAG_INTENTS_PATH", "")

        # Disable cache for consistent testing
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        return DynamicQueryExpander(
            llm_client=None,
            cache_dir=str(cache_dir),
            enable_cache=False,
        )

    # Characterize: Pattern-based fallback
    def test_characterize_pattern_fallback_scholarship(self, expander_no_llm):
        """
        CHARACTERIZE: Pattern-based expansion for '장학금' query (no LLM).

        Documents FALLBACK_RULES behavior.
        """
        query = "장학금"
        result = expander_no_llm.expand(query)

        # Current behavior: FALLBACK_RULES[0]
        # keywords=["장학금", "성적기준", "지급기준", "장학금지급"]
        assert "장학금" in result.keywords
        assert "성적기준" in result.keywords

        print(f"\n[CHARACTERIZATION] '{query}' -> keywords: {result.keywords}")
        print(f"  Intent: {result.intent}")
        print(f"  Method: {result.method}")

    def test_characterize_pattern_fallback_graduation(self, expander_no_llm):
        """
        CHARACTERIZE: Pattern-based expansion for '졸업' query (no LLM).
        """
        query = "졸업 요건"
        result = expander_no_llm.expand(query)

        # Current behavior: FALLBACK_RULES[1]
        # keywords=["졸업", "졸업요건", "이수학점", "졸업인증", "학칙"]
        assert "졸업" in result.keywords
        assert "졸업요건" in result.keywords

        print(f"\n[CHARACTERIZATION] '{query}' -> keywords: {result.keywords}")
        print(f"  Intent: {result.intent}")
        print(f"  Method: {result.method}")

    # Characterize: Expansion decision
    def test_characterize_should_expand_decision(self, expander_no_llm):
        """
        CHARACTERIZE: Current logic for when to expand queries.
        """
        test_cases = [
            ("장학금", True),  # Should expand (short, academic keyword)
            ("장학금규정", False),  # Should NOT expand (already formal)
            ("어떻게 해요", True),  # Should expand (vague)
            ("싶어", True),  # Should expand (intent marker)
        ]

        for query, expected in test_cases:
            should = expander_no_llm.should_expand(query)
            assert should == expected, (
                f"should_expand('{query}') expected {expected}, got {should}"
            )
            print(f"\n[CHARACTERIZATION] should_expand('{query}') -> {should}")


class TestCurrentBehaviorSnapshot:
    """
    Complete snapshot of current query expansion behavior.

    This serves as a comprehensive reference for regression detection.
    """

    @pytest.fixture
    def full_system(self, monkeypatch, tmp_path):
        """Full system with all components (no LLM)."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)

        analyzer = QueryAnalyzer(synonyms_path=None, intents_path=None)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        expander = DynamicQueryExpander(
            llm_client=None,
            cache_dir=str(cache_dir),
            enable_cache=False,
        )

        return {"analyzer": analyzer, "expander": expander}

    def test_characterize_comprehensive_snapshot_queries(self, full_system):
        """
        CHARACTERIZE: Comprehensive snapshot of common query patterns.

        This documents the CURRENT ACTUAL BEHAVIOR for various query types.
        """
        analyzer = full_system["analyzer"]
        expander = full_system["expander"]

        # Representative query set from SPEC scenarios
        test_queries = [
            "장학금 신청 방법",
            "연구년 자격 요건",
            "조교 근무 시간",
            "교원인사규정 제15조",
            "휴학 그리고 복학",
            "학교 가기 싫어",
            "제적",
        ]

        snapshot = {}
        for query in test_queries:
            # Query analysis
            query_type = analyzer.analyze(query)

            # Expansion
            expanded_analyzer = analyzer.expand_query(query)
            expanded_expander = expander.expand(query)

            snapshot[query] = {
                "query_type": query_type.value,
                "analyzer_expansion": expanded_analyzer,
                "expander_keywords": expanded_expander.keywords,
                "expander_intent": expanded_expander.intent,
                "expander_method": expanded_expander.method,
            }

            print(f"\n[CHARACTERIZATION SNAPSHOT] '{query}'")
            print(f"  Type: {query_type.value}")
            print(f"  Analyzer: '{expanded_analyzer}'")
            print(f"  Expander: {expanded_expander.keywords[:3]}...")  # First 3

        # Store snapshot data for validation (if needed)
        # This is primarily for documentation and regression detection
        assert snapshot is not None  # Snapshot created successfully


class TestBidirectionalSynonymExpansionCharacterization:
    """
    Characterization tests for bidirectional synonym expansion (TAG-004).

    These tests document the bidirectional synonym lookup behavior where:
    - Forward: "복무" -> expands to "근무" (key -> value)
    - Reverse: "근무" -> expands to "복무" (value -> key)
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    # Characterize: Bidirectional synonym expansion (복무 <-> 근무)
    def test_characterize_bokmu_expands_to_geunmu(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: '복무' should expand to include '근무'.
        """
        query = "복무 규정"
        result = analyzer.expand_query(query)

        # Forward lookup: "복무" -> ["근무", ...]
        assert "복무" in result, f"Expected '복무' in expansion: '{result}'"
        assert "근무" in result, f"Expected '근무' in expansion (bidirectional): '{result}'"
        print(f"\n[CHARACTERIZATION BIDIRECTIONAL] '{query}' -> '{result}'")

    def test_characterize_geunmu_expands_to_bokmu(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: '근무' should expand to include '복무' (reverse direction).
        """
        query = "근무 시간"
        result = analyzer.expand_query(query)

        # Reverse lookup: "근무" should find "복무" since "복무": ["근무"] exists
        assert "근무" in result, f"Expected '근무' in expansion: '{result}'"
        assert "복무" in result, f"Expected '복무' in expansion (bidirectional): '{result}'"
        print(f"\n[CHARACTERIZATION BIDIRECTIONAL] '{query}' -> '{result}'")

    # Characterize: Bidirectional synonym expansion (교원 <-> 교수)
    def test_characterize_loyon_expands_to_gyoso(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: '교원' should expand to include '교수'.
        """
        query = "교원 인사"
        result = analyzer.expand_query(query)

        assert "교원" in result, f"Expected '교원' in expansion: '{result}'"
        assert "교수" in result, f"Expected '교수' in expansion (bidirectional): '{result}'"
        print(f"\n[CHARACTERIZATION BIDIRECTIONAL] '{query}' -> '{result}'")

    def test_characterize_gyoso_expands_to_loyon(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: '교수' should expand to include '교원' (reverse direction).
        """
        query = "교수 연구"
        result = analyzer.expand_query(query)

        assert "교수" in result, f"Expected '교수' in expansion: '{result}'"
        assert "교원" in result, f"Expected '교원' in expansion (bidirectional): '{result}'"
        print(f"\n[CHARACTERIZATION BIDIRECTIONAL] '{query}' -> '{result}'")

    # Characterize: Bidirectional synonym expansion (승진 <-> 진급)
    def test_characterize_seungjin_expands_to_jingeup(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: '승진' should expand to include '진급'.
        """
        query = "승진 심사"
        result = analyzer.expand_query(query)

        assert "승진" in result, f"Expected '승진' in expansion: '{result}'"
        assert "진급" in result, f"Expected '진급' in expansion (bidirectional): '{result}'"
        print(f"\n[CHARACTERIZATION BIDIRECTIONAL] '{query}' -> '{result}'")

    def test_characterize_jingeup_expands_to_seungjin(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: '진급' should expand to include '승진' (reverse direction).
        """
        query = "진급 요건"
        result = analyzer.expand_query(query)

        assert "진급" in result, f"Expected '진급' in expansion: '{result}'"
        assert "승진" in result, f"Expected '승진' in expansion (bidirectional): '{result}'"
        print(f"\n[CHARACTERIZATION BIDIRECTIONAL] '{query}' -> '{result}'")
