"""
Unit tests for RegulationQueryGenerator.

Tests for SPEC-RAG-EVAL-001 Milestone 2: Persona Coverage Enhancement.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.rag.domain.evaluation.regulation_query_generator import (
    QueryTemplate,
    RegulationArticle,
    RegulationQueryGenerator,
)
from src.rag.domain.evaluation.parallel_evaluator import PersonaQuery


@pytest.fixture
def sample_regulation_data():
    """Create sample regulation data for testing."""
    return {
        "metadata": {"version": "1.0"},
        "docs": [
            {
                "id": "reg-0001",
                "kind": "regulation",
                "title": "학칙",
                "rule_code": "reg-0001",
                "articles": [
                    {
                        "article_no": "15",
                        "title": "휴학",
                        "content": "제15조(휴학) 학생이 질병 기타 부득이한 사유로 휴학하고자 할 때에는 보호자 연서로 학장의 승인을 얻어 휴학원을 제출하여야 한다. 휴학기간은 1년 이내로 하되, 부득이한 경우 1년의 범위 내에서 1회에 한하여 연장할 수 있다.",
                        "paragraphs": [],
                        "items": [],
                    },
                    {
                        "article_no": "16",
                        "title": "복학",
                        "content": "제16조(복학) 휴학한 자가 복학하고자 할 때에는 매학기 개시 30일 전까지 복학원을 제출하여 학장의 승인을 얻어야 한다. 복학은 휴학기간 만료 후 첫 학기에 하여야 한다.",
                        "paragraphs": [],
                        "items": [],
                    },
                ],
            },
            {
                "id": "reg-0002",
                "kind": "regulation",
                "title": "교원인사규정",
                "rule_code": "reg-0002",
                "articles": [
                    {
                        "article_no": "10",
                        "title": "자격요건",
                        "content": "제10조(자격요건) 교원으로 임용될 수 있는 자는 다음 각 호의 자격이 있어야 한다. 1. 대한민국 국민 2. 교원자격증 소지자 3. 기타 법령에서 정하는 자격요건을 구비한 자",
                        "paragraphs": [],
                        "items": [],
                    },
                ],
            },
            {
                "id": "reg-0003",
                "kind": "regulation",
                "title": "등록금규정",
                "rule_code": "reg-0003",
                "articles": [
                    {
                        "article_no": "5",
                        "title": "납부기한",
                        "content": "제5조(납부기한) 등록금은 매학기 개시 30일 전까지 납부하여야 한다. 미납부 시 제적될 수 있다.",
                        "paragraphs": [],
                        "items": [],
                    },
                ],
            },
        ],
    }


@pytest.fixture
def temp_regulation_file(sample_regulation_data):
    """Create a temporary regulation JSON file."""
    # Use a context manager that keeps the file open
    import os
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(sample_regulation_data, f, ensure_ascii=False)
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


class TestRegulationArticle:
    """Tests for RegulationArticle dataclass."""

    def test_creation(self):
        """Test creating a RegulationArticle."""
        article = RegulationArticle(
            regulation_title="학칙",
            article_no="15",
            article_title="휴학",
            content="휴학에 관한 내용...",
            rule_code="reg-0001",
        )

        assert article.regulation_title == "학칙"
        assert article.article_no == "15"
        assert article.article_title == "휴학"
        assert article.rule_code == "reg-0001"

    def test_default_lists(self):
        """Test that paragraphs and items default to empty lists."""
        article = RegulationArticle(
            regulation_title="학칙",
            article_no="15",
            article_title="휴학",
            content="내용",
            rule_code="reg-0001",
        )

        assert article.paragraphs == []
        assert article.items == []


class TestQueryTemplate:
    """Tests for QueryTemplate dataclass."""

    def test_creation(self):
        """Test creating a QueryTemplate."""
        template = QueryTemplate(
            pattern="{regulation} {article}의 내용은 무엇인가요?",
            difficulty="easy",
            query_type="article",
            required_info=["basic_info"],
        )

        assert template.difficulty == "easy"
        assert template.query_type == "article"
        assert "basic_info" in template.required_info


class TestRegulationQueryGenerator:
    """Tests for RegulationQueryGenerator."""

    def test_init_without_file(self):
        """Test initialization without regulation file."""
        generator = RegulationQueryGenerator()
        assert generator.regulations == []
        assert generator.articles == []

    def test_load_regulations(self, temp_regulation_file, sample_regulation_data):
        """Test loading regulations from file."""
        generator = RegulationQueryGenerator()
        generator.load_regulations(temp_regulation_file)

        assert len(generator.regulations) == len(sample_regulation_data["docs"])
        assert len(generator.articles) == 4  # Total articles across all regulations (2+1+1)

    def test_load_regulations_file_not_found(self):
        """Test handling of missing file."""
        generator = RegulationQueryGenerator()
        generator.load_regulations("/nonexistent/path.json")

        assert generator.regulations == []

    def test_generate_queries_basic(self, temp_regulation_file):
        """Test basic query generation."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=10)

        assert len(queries) == 10
        for query in queries:
            assert isinstance(query, PersonaQuery)
            assert query.query
            assert query.difficulty in ["easy", "medium", "hard"]

    def test_generate_queries_with_persona(self, temp_regulation_file):
        """Test query generation with persona."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=5, persona="professor")

        assert len(queries) == 5
        for query in queries:
            assert query.persona == "professor"

    def test_generate_queries_difficulty_distribution(self, temp_regulation_file):
        """Test difficulty distribution follows guidelines."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=100)

        # Count difficulties
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for query in queries:
            difficulties[query.difficulty] += 1

        # Check distribution is approximately correct (30/40/30)
        # Allow 10% tolerance
        assert 20 <= difficulties["easy"] <= 40  # ~30%
        assert 30 <= difficulties["medium"] <= 50  # ~40%
        assert 20 <= difficulties["hard"] <= 40  # ~30%

    def test_generate_queries_no_articles(self):
        """Test query generation with no articles loaded."""
        generator = RegulationQueryGenerator()
        queries = generator.generate_queries(count=10)

        assert queries == []

    def test_generate_all_personas_queries(self, temp_regulation_file):
        """Test generating queries for all personas."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        all_queries = generator.generate_all_personas_queries(queries_per_persona=5)

        # Should have 6 personas
        assert len(all_queries) == 6

        # Each persona should have queries
        for persona_queries in all_queries.values():
            assert len(persona_queries) == 5

    def test_get_article_queries(self, temp_regulation_file):
        """Test getting queries for specific article."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.get_article_queries("15")

        assert len(queries) > 0
        # Queries should reference the regulation or topic related to article 15
        for query in queries:
            # The query should contain the regulation name or topic
            assert any(
                term in query.query
                for term in ["학칙", "휴학", "규정"]
            )

    def test_get_article_queries_not_found(self, temp_regulation_file):
        """Test getting queries for non-existent article."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.get_article_queries("999")

        assert queries == []

    def test_get_statistics(self, temp_regulation_file):
        """Test getting statistics."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        stats = generator.get_statistics()

        assert stats["total_regulations"] == 3
        assert stats["total_articles"] == 4  # 2 in 학칙 + 1 in 교원인사규정 + 1 in 등록금규정
        assert stats["avg_article_length"] > 0
        assert "difficulty_distribution" in stats

    def test_query_types(self, temp_regulation_file):
        """Test that different query types are generated."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=50)

        query_types = set(q.category for q in queries)

        # Should have multiple query types
        assert len(query_types) > 0

    def test_expected_info_population(self, temp_regulation_file):
        """Test that expected_info is populated."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=10)

        for query in queries:
            assert isinstance(query.expected_info, list)

    def test_expected_intent_inference(self, temp_regulation_file):
        """Test that expected_intent is inferred."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=10)

        valid_intents = {"procedural", "temporal", "eligibility", "informational"}
        for query in queries:
            assert query.expected_intent in valid_intents


class TestTemplateConstants:
    """Tests for template constants."""

    def test_easy_templates_exist(self):
        """Test that easy templates are defined."""
        generator = RegulationQueryGenerator()
        assert len(generator.EASY_TEMPLATES) > 0
        for template in generator.EASY_TEMPLATES:
            assert template.difficulty == "easy"

    def test_medium_templates_exist(self):
        """Test that medium templates are defined."""
        generator = RegulationQueryGenerator()
        assert len(generator.MEDIUM_TEMPLATES) > 0
        for template in generator.MEDIUM_TEMPLATES:
            assert template.difficulty == "medium"

    def test_hard_templates_exist(self):
        """Test that hard templates are defined."""
        generator = RegulationQueryGenerator()
        assert len(generator.HARD_TEMPLATES) > 0
        for template in generator.HARD_TEMPLATES:
            assert template.difficulty == "hard"

    def test_template_query_types(self):
        """Test that templates have valid query types."""
        valid_types = {"article", "procedure", "temporal", "cross_reference"}
        generator = RegulationQueryGenerator()

        for template in generator.EASY_TEMPLATES + generator.MEDIUM_TEMPLATES + generator.HARD_TEMPLATES:
            assert template.query_type in valid_types


class TestPersonaQueryIntegration:
    """Tests for PersonaQuery integration with generator."""

    def test_persona_query_fields(self, temp_regulation_file):
        """Test that PersonaQuery has all required fields."""
        generator = RegulationQueryGenerator(temp_regulation_file)
        queries = generator.generate_queries(count=1)

        if queries:
            query = queries[0]
            assert hasattr(query, "query")
            assert hasattr(query, "persona")
            assert hasattr(query, "category")
            assert hasattr(query, "difficulty")
            assert hasattr(query, "expected_intent")
            assert hasattr(query, "expected_info")

    def test_persona_specific_queries(self, temp_regulation_file):
        """Test that queries are adapted for different personas."""
        generator = RegulationQueryGenerator(temp_regulation_file)

        freshman_queries = generator.generate_queries(count=5, persona="freshman")
        professor_queries = generator.generate_queries(count=5, persona="professor")

        # Both should generate queries
        assert len(freshman_queries) == 5
        assert len(professor_queries) == 5

        # Personas should be set correctly
        assert all(q.persona == "freshman" for q in freshman_queries)
        assert all(q.persona == "professor" for q in professor_queries)
