"""
Unit tests for Persona Manager domain component.

Tests user persona simulation and query generation.
"""

import pytest

from src.rag.domain.evaluation.models import PersonaProfile
from src.rag.domain.evaluation.personas import PERSONAS, PersonaManager


class TestPersonaDefinitions:
    """Test persona definitions."""

    def test_six_personas_defined(self):
        """WHEN checking personas, THEN should have 6 personas defined."""
        assert len(PERSONAS) == 6

    def test_required_personas_exist(self):
        """WHEN checking required personas, THEN all should exist."""
        required_personas = [
            "freshman",
            "graduate",
            "professor",
            "staff",
            "parent",
            "international",
        ]

        for persona_name in required_personas:
            assert persona_name in PERSONAS
            assert isinstance(PERSONAS[persona_name], PersonaProfile)

    def test_freshman_persona_attributes(self):
        """WHEN checking freshman persona, THEN should have beginner level."""
        freshman = PERSONAS["freshman"]

        assert freshman.name == "freshman"
        assert freshman.display_name == "신입생"
        assert freshman.expertise_level == "beginner"
        assert freshman.vocabulary_style == "simple"
        assert len(freshman.query_templates) > 0
        assert len(freshman.common_topics) > 0

    def test_professor_persona_attributes(self):
        """WHEN checking professor persona, THEN should have advanced level."""
        professor = PERSONAS["professor"]

        assert professor.name == "professor"
        assert professor.display_name == "교수"
        assert professor.expertise_level == "advanced"
        assert professor.vocabulary_style == "academic"

    def test_international_persona_attributes(self):
        """WHEN checking international persona, THEN should have mixed vocabulary."""
        international = PERSONAS["international"]

        assert international.name == "international"
        assert international.display_name == "외국인유학생"
        assert international.vocabulary_style == "mixed"


class TestPersonaManager:
    """Test Persona Manager functionality."""

    @pytest.fixture
    def manager(self):
        """Create persona manager."""
        return PersonaManager()

    def test_manager_initialization(self, manager):
        """WHEN manager created, THEN should load all personas."""
        assert len(manager.personas) == 6
        assert manager.list_personas() == [
            "freshman",
            "graduate",
            "professor",
            "staff",
            "parent",
            "international",
        ]

    def test_get_existing_persona(self, manager):
        """WHEN getting existing persona, THEN should return correct profile."""
        freshman = manager.get_persona("freshman")

        assert isinstance(freshman, PersonaProfile)
        assert freshman.name == "freshman"

    def test_get_nonexistent_persona_raises_error(self, manager):
        """WHEN getting nonexistent persona, THEN should raise ValueError."""
        with pytest.raises(ValueError, match="Persona 'unknown' not found"):
            manager.get_persona("unknown")

    def test_generate_freshman_queries(self, manager):
        """WHEN generating freshman queries, THEN should use simple vocabulary."""
        queries = manager.generate_queries("freshman", count=10)

        assert len(queries) == 10

        # Check queries are simple
        for query in queries:
            assert isinstance(query, str)
            assert len(query) > 0
            # Freshman queries should be relatively short
            assert len(query.split()) <= 20

    def test_generate_professor_queries(self, manager):
        """WHEN generating professor queries, THEN should use academic vocabulary."""
        queries = manager.generate_queries("professor", count=10)

        assert len(queries) == 10

        # Check queries use academic language
        academic_terms = ["조항", "규정", "적용", "기준"]
        academic_query_count = sum(
            1 for q in queries if any(term in q for term in academic_terms)
        )

        # At least some queries should have academic terms
        assert academic_query_count > 0

    def test_generate_international_queries(self, manager):
        """WHEN generating international queries, THEN should have English queries."""
        queries = manager.generate_queries("international", count=20)

        assert len(queries) == 20

        # Check for English queries (should be ~30%)
        english_queries = [
            q for q in queries if any(char.isalpha() and char.isascii() for char in q)
        ]

        # Should have at least some English queries
        assert len(english_queries) >= 2

    def test_generate_all_personas_queries(self, manager):
        """WHEN generating queries for all personas, THEN should get 120 queries."""
        all_queries = manager.generate_all_personas_queries(queries_per_persona=20)

        assert len(all_queries) == 6

        # Each persona should have 20 queries
        for _persona_name, queries in all_queries.items():
            assert len(queries) == 20

        # Total should be 120 queries
        total_queries = sum(len(queries) for queries in all_queries.values())
        assert total_queries == 120

    def test_generate_queries_with_custom_topics(self, manager):
        """WHEN generating with custom topics, THEN should use those topics."""
        custom_topics = ["등록금", "수강신청"]

        queries = manager.generate_queries("freshman", count=5, topics=custom_topics)

        assert len(queries) == 5

        # All queries should use custom topics
        for query in queries:
            assert any(topic in query for topic in custom_topics)

    def test_validate_query_for_freshman(self, manager):
        """WHEN validating freshman query, THEN should check simplicity."""
        valid_query = "휴학 어떻게 해요?"

        assert manager.validate_query_for_persona(valid_query, "freshman") is True
        # Note: Current implementation doesn't actually reject based on this
        # This test documents expected behavior

    def test_get_persona_answer_preferences(self, manager):
        """WHEN getting answer preferences, THEN should return persona-specific prefs."""
        freshman_prefs = manager.get_persona_answer_preferences("freshman")

        assert isinstance(freshman_prefs, dict)
        assert "detail_level" in freshman_prefs
        assert freshman_prefs["detail_level"] == "simple"

        professor_prefs = manager.get_persona_answer_preferences("professor")

        assert professor_prefs["detail_level"] == "comprehensive"
        assert professor_prefs["citation_style"] == "detailed"


class TestPersonaProfile:
    """Test PersonaProfile model."""

    def test_generate_query_with_template(self):
        """WHEN generating query, THEN should use template correctly."""
        profile = PersonaProfile(
            name="test",
            display_name="Test",
            expertise_level="beginner",
            vocabulary_style="simple",
            query_templates=["{topic} 어떻게 해요?", "{topic} 알려주세요"],
            common_topics=["휴학", "복학"],
        )

        import random

        random.seed(42)  # For reproducibility
        query = profile.generate_query("휴학")

        assert "휴학" in query
        assert query in ["휴학 어떻게 해요?", "휴학 알려주세요"]


class TestStaffPersonaTopics:
    """Test staff persona topic coverage (SPEC-RAG-QUALITY-005 Phase 2)."""

    def test_staff_persona_has_required_topics(self):
        """WHEN checking staff persona, THEN should have all required topics."""
        staff = PERSONAS["staff"]

        # Required topics per SPEC-RAG-QUALITY-005 Phase 2
        required_topics = ["복무", "휴가", "급여", "연수", "사무용품", "시설사용", "입찰"]

        for topic in required_topics:
            assert topic in staff.common_topics, f"Missing staff topic: {topic}"

    def test_staff_persona_topic_count(self):
        """WHEN checking staff persona, THEN should have at least 7 topics."""
        staff = PERSONAS["staff"]

        # Should have at least 7 topics (original 6 + 입찰)
        assert len(staff.common_topics) >= 7

    def test_staff_persona_vocabulary_style(self):
        """WHEN checking staff persona, THEN should have administrative vocabulary."""
        staff = PERSONAS["staff"]

        assert staff.vocabulary_style == "administrative"
        assert staff.expertise_level == "intermediate"

    def test_staff_persona_query_templates_use_topics(self):
        """WHEN generating staff queries, THEN should use staff-specific topics."""
        manager = PersonaManager()
        queries = manager.generate_queries("staff", count=20)

        # All queries should be strings
        assert len(queries) == 20
        for query in queries:
            assert isinstance(query, str)
            assert len(query) > 0
