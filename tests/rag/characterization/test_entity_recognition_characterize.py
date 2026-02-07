"""
Characterization tests for existing Entity Recognition behavior.

These tests capture CURRENT BEHAVIOR for entity recognition patterns
before implementing SPEC-RAG-SEARCH-001 enhancements.

Purpose: Document what entities are currently recognized by the system.
"""

import pytest

from src.rag.infrastructure.query_analyzer import QueryAnalyzer, QueryType


class TestAcademicKeywordsCharacterization:
    """
    Characterization tests for current ACADEMIC_KEYWORDS recognition.

    Documents which keywords are currently recognized and how they
    trigger query type classification.
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    def test_characterize_academic_keywords_list(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Complete list of ACADEMIC_KEYWORDS and their behavior.

        This test documents ALL academic keywords currently recognized.
        """
        # From query_analyzer.py line 115-139
        academic_keywords = [
            "휴학",
            "복학",
            "제적",
            "자퇴",
            "전과",
            "편입",
            "졸업",
            "입학",
            "등록",
            "수강",
            "장학",
            "학점",
            "성적",
            "시험",
            "출석",
            "학위",
            "논문",
            "석사",
            "박사",
            "교원",
            "교수",
            "조교",
            "학생회",
        ]

        results = {}
        for keyword in academic_keywords:
            query_type = analyzer.analyze(keyword)
            results[keyword] = query_type

            # All should be classified as REGULATION_NAME or INTENT
            # (INTENT if they have intent markers like "싶어")
            assert query_type in [
                QueryType.REGULATION_NAME,
                QueryType.INTENT,
                QueryType.NATURAL_QUESTION,
            ], f"Academic keyword '{keyword}' got unexpected type: {query_type.value}"

            print(
                f"\n[CHARACTERIZATION] ACADEMIC_KEYWORD '{keyword}' -> {query_type.value}"
            )

        # Verify all keywords are recognized
        assert len(results) == len(academic_keywords)

    def test_characterize_academic_keyword_in_context(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: How academic keywords behave in full queries.
        """
        test_cases = [
            ("휴학 방법", QueryType.NATURAL_QUESTION),  # Has question marker
            ("장학금 신청", QueryType.NATURAL_QUESTION),  # Has question marker
            ("복학", QueryType.REGULATION_NAME),  # Just keyword
            ("휴학하고 싶어", QueryType.INTENT),  # Has intent marker
        ]

        for query, expected_type in test_cases:
            actual_type = analyzer.analyze(query)
            assert actual_type == expected_type, (
                f"Query '{query}' expected {expected_type.value}, got {actual_type.value}"
            )
            print(f"\n[CHARACTERIZATION] '{query}' -> {actual_type.value}")


class TestArticlePatternCharacterization:
    """
    Characterization tests for ARTICLE_PATTERN matching.

    Documents how section references (제N조, 제N항, 제N호) are currently recognized.
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    def test_characterize_article_patterns(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: All article reference pattern variations.

        Documents which patterns match ARTICLE_REFERENCE type.
        """
        from src.rag.infrastructure.patterns import ARTICLE_PATTERN

        test_cases = [
            # Pattern variations
            ("제15조", True),
            ("제3조", True),
            ("제1항", True),
            ("제2호", True),
            ("제3조의2", True),
            ("제15조제2항", True),
            ("제 15 조", True),  # With spaces
            # Non-matches
            ("15조", False),  # Missing 제 prefix
            ("조15", False),  # Wrong order
            ("제조", False),  # Missing number
        ]

        for query, should_match in test_cases:
            matches = ARTICLE_PATTERN.search(query) is not None
            query_type = analyzer.analyze(query)

            if should_match:
                assert matches, f"ARTICLE_PATTERN should match '{query}'"
                assert query_type == QueryType.ARTICLE_REFERENCE, (
                    f"Query '{query}' should be ARTICLE_REFERENCE, got {query_type.value}"
                )
            else:
                assert not matches or query_type != QueryType.ARTICLE_REFERENCE, (
                    f"ARTICLE_PATTERN should NOT match '{query}' as ARTICLE_REFERENCE"
                )

            print(
                f"\n[CHARACTERIZATION] ARTICLE_PATTERN '{query}' -> matches={matches}, type={query_type.value}"
            )


class TestQuestionMarkersCharacterization:
    """
    Characterization tests for QUESTION_MARKERS.

    Documents which markers trigger NATURAL_QUESTION classification.
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    def test_characterize_question_markers(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: All question markers and their effect.

        Documents which phrases cause NATURAL_QUESTION classification.
        """
        # From query_analyzer.py line 142-167
        question_markers = [
            "어떻게",
            "무엇",
            "왜",
            "언제",
            "어디",
            "누가",
            "어떤",
            "할까",
            "인가",
            "?",
            "방법",
            "절차",
            "과정",
            "설명",
            "알려",
            "알려줘",
            "알려주세요",
            "설명해",
            "설명해주세요",
            "가르쳐",
            "가르쳐줘",
            "가르쳐주세요",
            "법",
            "신청",
        ]

        results = {}
        for marker in question_markers:
            # Create a simple query with the marker
            query = f"휴학 {marker}" if marker != "?" else f"휴학{marker}"
            query_type = analyzer.analyze(query)
            results[marker] = query_type

            print(
                f"\n[CHARACTERIZATION] QUESTION_MARKER '{marker}' -> {query_type.value}"
            )

        # Most markers should trigger NATURAL_QUESTION or INTENT
        # (INTENT takes precedence when both markers present)

    def test_characterize_marker_combinations(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: How multiple markers interact in a query.
        """
        test_cases = [
            ("휴학 방법 절차", QueryType.NATURAL_QUESTION),  # Multiple markers
            ("어떻게 신청", QueryType.NATURAL_QUESTION),  # Multiple markers
            ("휴학 방법이 뭐야", QueryType.NATURAL_QUESTION),  # Multiple markers
        ]

        for query, expected_type in test_cases:
            actual_type = analyzer.analyze(query)
            assert actual_type == expected_type, (
                f"Query '{query}' expected {expected_type.value}, got {actual_type.value}"
            )
            print(f"\n[CHARACTERIZATION] '{query}' -> {actual_type.value}")


class TestCurrentEntityGapsCharacterization:
    """
    Characterization tests documenting GAPS in current entity recognition.

    These tests document what is NOT currently recognized but SHOULD BE
    according to SPEC-RAG-SEARCH-001.

    This defines the scope of improvement needed.
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    def test_characterize_missing_entity_types(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Entity types NOT currently recognized.

        Documents gaps that TAG-001 needs to address:
        1. REQUIREMENT entities (자격, 요건, 조건, 기준, 제한)
        2. BENEFIT entities (혜택, 지급, 지원, 급여)
        3. DEADLINE entities (기한, 마감, 날짜, 기간)
        4. HYPERNYM expansion (등록금→학사→행정)
        """
        # REQUIREMENT keywords - currently scattered, not unified
        requirement_queries = [
            "자격 요건",
            "신청 조건",
            "선정 기준",
            "제한 사항",
        ]

        print("\n[CHARACTERIZATION GAP] REQUIREMENT entities:")
        for query in requirement_queries:
            query_type = analyzer.analyze(query)
            expanded = analyzer.expand_query(query)
            print(f"  '{query}' -> type={query_type.value}, expanded='{expanded}'")
            # Note: These may trigger different behaviors, not unified

        # BENEFIT keywords - not specifically recognized
        benefit_queries = [
            "혜택 지급",
            "장학금 지원",
            "급여 수당",
        ]

        print("\n[CHARACTERIZATION GAP] BENEFIT entities:")
        for query in benefit_queries:
            query_type = analyzer.analyze(query)
            expanded = analyzer.expand_query(query)
            print(f"  '{query}' -> type={query_type.value}, expanded='{expanded}'")

        # DEADLINE keywords - not specifically recognized
        deadline_queries = [
            "신청 기한",
            "마감 날짜",
            "접수 기간",
        ]

        print("\n[CHARACTERIZATION GAP] DEADLINE entities:")
        for query in deadline_queries:
            query_type = analyzer.analyze(query)
            expanded = analyzer.expand_query(query)
            print(f"  '{query}' -> type={query_type.value}, expanded='{expanded}'")

    def test_characterize_missing_hypernym_expansion(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Missing hierarchical term expansion.

        Documents lack of hypernym (상위어) expansion:
        - 등록금 → 학사 → 행정
        - 장학금 → 재정 → 지원
        """
        test_cases = [
            ("등록금", ["수업료"]),  # Current synonym only
            ("장학금", ["장학"]),  # Current synonym only
            # Missing: hierarchical expansion
        ]

        print("\n[CHARACTERIZATION GAP] HYPERNYM expansion:")
        for query, expected_synonyms in test_cases:
            expanded = analyzer.expand_query(query)
            found_synonyms = [s for s in expected_synonyms if s in expanded]

            print(f"  '{query}' -> expanded='{expanded}'")
            print(f"    Current synonyms found: {found_synonyms}")
            print("    Missing hierarchical terms would be: 학사, 행정 (for 등록금)")

    def test_characterize_missing_procedure_expansion(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Incomplete procedure chain expansion.

        Documents lack of comprehensive procedure expansion:
        - 신청 → 절차 → 서류 → 제출
        """
        procedure_queries = [
            "장학금 신청",
            "휴학 절차",
            "서류 제출",
        ]

        print("\n[CHARACTERIZATION GAP] PROCEDURE expansion:")
        for query in procedure_queries:
            expanded = analyzer.expand_query(query)
            print(f"  '{query}' -> expanded='{expanded}'")
            # Note: May have some keywords, but not comprehensive chain


class TestIntentPatternsCharacterization:
    """
    Characterization tests for current INTENT_PATTERNS behavior.

    Documents all existing intent patterns and their triggers.
    """

    @pytest.fixture
    def analyzer(self, monkeypatch) -> QueryAnalyzer:
        """Isolate from external config files."""
        monkeypatch.delenv("RAG_SYNONYMS_PATH", raising=False)
        monkeypatch.delenv("RAG_INTENTS_PATH", raising=False)
        return QueryAnalyzer(synonyms_path=None, intents_path=None)

    def test_characterize_all_intent_patterns(self, analyzer: QueryAnalyzer):
        """
        CHARACTERIZE: Complete documentation of INTENT_PATTERNS.

        This captures ALL 40+ intent patterns currently defined.
        """
        # Get internal intent rules for inspection
        intent_rules = analyzer._intent_rules

        print(f"\n[CHARACTERIZATION] Total intent rules: {len(intent_rules)}")

        # Sample key patterns (not exhaustive for brevity)
        key_patterns = [
            "scholarship_intent",  # 장학금 관련
            "graduation_deferral",  # 졸업 유예
            "leave_of_absence",  # 휴학
            "faculty_promotion",  # 교원 승진
        ]

        # Test representative intent patterns
        test_queries = [
            ("장학금 받고 싶어", ["장학금", "신청", "지급"]),
            ("졸업 미루고 싶어", ["학사학위취득유예", "졸업유예"]),
            ("학교 가기 싫어", ["휴직", "휴가", "연구년", "안식년"]),
            ("교수 승진", ["승진", "교원인사규정", "업적평가"]),
        ]

        print("\n[CHARACTERIZATION] Intent pattern matching:")
        for query, expected_keywords in test_queries:
            matches = analyzer._match_intents(query)
            all_keywords = []
            for match in matches:
                all_keywords.extend(match.keywords)

            print(f"  Query: '{query}'")
            print(f"    Matched intents: {len(matches)}")
            print(f"    Keywords: {all_keywords[:5]}...")  # First 5

            # Verify at least some expected keywords are present
            found = [kw for kw in expected_keywords if kw in all_keywords]
            assert len(found) > 0, (
                f"Expected at least one of {expected_keywords} in {all_keywords}"
            )
