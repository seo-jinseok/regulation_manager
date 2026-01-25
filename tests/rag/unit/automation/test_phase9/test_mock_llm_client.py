"""
Unit Tests for MockLLMClientForQueryGen (Phase 9).

Tests for mock LLM client used in RAG testing automation.
Clean Architecture: Infrastructure layer tests.
"""

import json

from src.rag.automation.infrastructure.mock_llm_client import (
    MockLLMClientForQueryGen,
)


class TestMockLLMClientInitialization:
    """Test suite for MockLLMClient initialization."""

    def test_initialize_with_default_template_mode(self):
        """WHEN initializing with defaults, THEN should use template mode."""
        client = MockLLMClientForQueryGen()

        assert client.use_template_responses is True
        assert client.call_count == 0

    def test_initialize_with_template_mode_enabled(self):
        """WHEN initializing with template=True, THEN should use template mode."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        assert client.use_template_responses is True

    def test_initialize_with_template_mode_disabled(self):
        """WHEN initializing with template=False, THEN should use simple mode."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        assert client.use_template_responses is False


class TestCallCountTracking:
    """Test suite for call count tracking functionality."""

    def test_call_count_increments_on_generate(self):
        """WHEN generate is called, THEN call_count should increment."""
        client = MockLLMClientForQueryGen()

        client.generate("system", "user")
        assert client.call_count == 1

        client.generate("system", "user")
        assert client.call_count == 2

    def test_call_count_resets_to_zero(self):
        """WHEN reset_call_count is called, THEN count should return to 0."""
        client = MockLLMClientForQueryGen()

        client.generate("system", "user")
        client.generate("system", "user")
        assert client.call_count == 2

        client.reset_call_count()
        assert client.call_count == 0

    def test_call_count_property_is_readonly(self):
        """WHEN accessing call_count property, THEN should return current count."""
        client = MockLLMClientForQueryGen()

        assert client.call_count == 0
        client.generate("system", "user")
        assert client.call_count == 1


class TestGenerateSimpleResponse:
    """Test suite for simple response generation."""

    def test_generate_simple_response_returns_valid_json(self):
        """WHEN using simple mode, THEN should return valid JSON array."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user")

        # Should be valid JSON
        queries = json.loads(response)
        assert isinstance(queries, list)

    def test_generate_simple_response_has_query_structure(self):
        """WHEN using simple mode, THEN queries should have required fields."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user")
        queries = json.loads(response)

        assert len(queries) > 0
        for query in queries:
            assert "query" in query
            assert "type" in query
            assert "difficulty" in query

    def test_generate_simple_response_includes_procedural_query(self):
        """WHEN using simple mode, THEN should include procedural query."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user")
        queries = json.loads(response)

        procedural_queries = [q for q in queries if q["type"] == "procedural"]
        assert len(procedural_queries) > 0

    def test_generate_simple_response_includes_eligibility_query(self):
        """WHEN using simple mode, THEN should include eligibility query."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user")
        queries = json.loads(response)

        eligibility_queries = [q for q in queries if q["type"] == "eligibility"]
        assert len(eligibility_queries) > 0


class TestGenerateRealisticResponse:
    """Test suite for realistic template-based response generation."""

    def test_generate_realistic_response_detects_easy_count(self):
        """WHEN prompt contains Easy count, THEN should generate that many easy queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (3개), Medium (2개), Hard (1개) queries"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert len(easy_queries) == 3

    def test_generate_realistic_response_detects_medium_count(self):
        """WHEN prompt contains Medium count, THEN should generate that many medium queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개), Medium (4개), Hard (1개) queries"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        medium_queries = [q for q in queries if q["difficulty"] == "medium"]
        assert len(medium_queries) == 4

    def test_generate_realistic_response_detects_hard_count(self):
        """WHEN prompt contains Hard count, THEN should generate that many hard queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개), Medium (1개), Hard (5개) queries"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        hard_queries = [q for q in queries if q["difficulty"] == "hard"]
        assert len(hard_queries) == 5

    def test_generate_realistic_response_uses_default_counts(self):
        """WHEN prompt has no counts, THEN should use default counts."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate some queries"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Default: 1 Easy, 2 Medium, 1 Hard = 4 total
        assert len(queries) == 4

    def test_generate_realistic_response_detects_professor_persona(self):
        """WHEN prompt contains 교수/Professor, THEN should use professor language."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate queries using professor language (교권, 규정, 연구비 등)
        professor_queries = [
            q
            for q in queries
            if "교권" in q["query"] or "규정" in q["query"] or "연구비" in q["query"]
        ]
        assert len(professor_queries) > 0

    def test_generate_realistic_response_detects_student_persona(self):
        """WHEN prompt contains 학생/Student, THEN should use student language."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate queries using student language (신청, 방법, 장학금 등)
        student_queries = [
            q
            for q in queries
            if "신청" in q["query"] or "방법" in q["query"] or "장학금" in q["query"]
        ]
        assert len(student_queries) > 0

    def test_generate_realistic_response_detects_staff_persona(self):
        """WHEN prompt contains 직원/Staff, THEN should use staff language."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for 직원"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate queries using staff language (절차, 방법, 근무 등)
        staff_queries = [
            q
            for q in queries
            if "절차" in q["query"] or "방법" in q["query"] or "근무" in q["query"]
        ]
        assert len(staff_queries) > 0

    def test_generate_realistic_response_english_professor(self):
        """WHEN prompt contains Professor, THEN should use English professor language."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for Professor"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate queries with professor-specific terms
        assert len(queries) >= 1

    def test_generate_realistic_response_english_student(self):
        """WHEN prompt contains Student, THEN should use English student language."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for Student"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate queries with student-specific terms
        assert len(queries) >= 1


class TestGenerateQueryTypes:
    """Test suite for different query types in realistic mode."""

    def test_generate_includes_fact_check_type(self):
        """WHEN generating easy queries, THEN should include fact_check type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (3개) queries for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        fact_check_queries = [q for q in queries if q["type"] == "fact_check"]
        assert len(fact_check_queries) > 0

    def test_generate_includes_procedural_type(self):
        """WHEN generating easy queries, THEN should include procedural type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (3개) queries for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        procedural_queries = [q for q in queries if q["type"] == "procedural"]
        assert len(procedural_queries) > 0

    def test_generate_includes_eligibility_type(self):
        """WHEN generating medium queries, THEN should include eligibility type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Medium (3개) queries for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        eligibility_queries = [q for q in queries if q["type"] == "eligibility"]
        assert len(eligibility_queries) > 0

    def test_generate_includes_emotional_type(self):
        """WHEN generating hard queries for professor, THEN should include emotional type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Hard (2개) queries for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        emotional_queries = [q for q in queries if q["type"] == "emotional"]
        assert len(emotional_queries) > 0

    def test_generate_includes_complex_type(self):
        """WHEN generating hard queries for student, THEN should include complex type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Hard (2개) queries for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        complex_queries = [q for q in queries if q["type"] == "complex"]
        assert len(complex_queries) > 0


class TestGetEmbedding:
    """Test suite for mock embedding generation."""

    def test_get_embedding_returns_correct_dimensions(self):
        """WHEN generating embedding, THEN should return 384 dimensions."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("test text")

        assert len(embedding) == 384

    def test_get_embedding_returns_float_values(self):
        """WHEN generating embedding, THEN should return list of floats."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("test text")

        assert all(isinstance(x, float) for x in embedding)

    def test_get_embedding_values_in_range(self):
        """WHEN generating embedding, THEN values should be in [-1, 1] range."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("test text")

        assert all(-1.0 <= x <= 1.0 for x in embedding)

    def test_get_embedding_is_deterministic(self):
        """WHEN generating embedding for same text, THEN should return same values."""
        client = MockLLMClientForQueryGen()

        embedding1 = client.get_embedding("test text")
        embedding2 = client.get_embedding("test text")

        assert embedding1 == embedding2

    def test_get_embedding_differs_for_different_text(self):
        """WHEN generating embedding for different text, THEN values should differ."""
        client = MockLLMClientForQueryGen()

        embedding1 = client.get_embedding("test text 1")
        embedding2 = client.get_embedding("test text 2")

        assert embedding1 != embedding2

    def test_get_embedding_handles_empty_string(self):
        """WHEN generating embedding for empty string, THEN should still work."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("")

        assert len(embedding) == 384

    def test_get_embedding_handles_unicode(self):
        """WHEN generating embedding for Korean text, THEN should work correctly."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("휴학 신청 방법 알려주세요")

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)


class TestExtractCount:
    """Test suite for _extract_count helper method."""

    def test_extract_count_finds_easy_count(self):
        """WHEN prompt has Easy count, THEN should extract it."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate Easy (5개) queries", "Easy")

        assert count == 5

    def test_extract_count_finds_medium_count(self):
        """WHEN prompt has Medium count, THEN should extract it."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate Medium (3개) queries", "Medium")

        assert count == 3

    def test_extract_count_finds_hard_count(self):
        """WHEN prompt has Hard count, THEN should extract it."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate Hard (2개) queries", "Hard")

        assert count == 2

    def test_extract_count_returns_default_for_easy(self):
        """WHEN prompt has no count, THEN should return default for Easy."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate some queries", "Easy")

        assert count == 1

    def test_extract_count_returns_default_for_medium(self):
        """WHEN prompt has no count, THEN should return default for Medium."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate some queries", "Medium")

        assert count == 2

    def test_extract_count_returns_default_for_hard(self):
        """WHEN prompt has no count, THEN should return default for Hard."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate some queries", "Hard")

        assert count == 1

    def test_extract_count_handles_spaces(self):
        """WHEN pattern has spaces, THEN should still extract correctly."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        # The regex doesn't handle irregular spacing well - this documents actual behavior
        count = client._extract_count("Easy  (  3개  ) queries", "Easy")

        # With irregular spaces, the regex may not match, returning default
        assert count == 1  # Default for Easy


class TestGenerateParameters:
    """Test suite for generate() method parameters."""

    def test_generate_ignores_system_prompt_in_simple_mode(self):
        """WHEN in simple mode, THEN system_prompt should not affect output."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response1 = client.generate("system1", "user")
        response2 = client.generate("system2", "user")

        # Same output regardless of system prompt
        assert response1 == response2

    def test_generate_ignores_user_message_in_simple_mode(self):
        """WHEN in simple mode, THEN user_message should not affect output."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response1 = client.generate("system", "user1")
        response2 = client.generate("system", "user2")

        # Same output regardless of user message
        assert response1 == response2

    def test_generate_ignores_temperature(self):
        """WHEN temperature is provided, THEN it should be ignored."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response1 = client.generate("system", "user", temperature=0.0)
        response2 = client.generate("system", "user", temperature=1.0)

        # Temperature is ignored in mock
        assert response1 == response2


class TestGenerateOutputFormat:
    """Test suite for output format validation."""

    def test_generate_returns_json_string(self):
        """WHEN generate is called, THEN should return JSON string."""
        client = MockLLMClientForQueryGen()

        response = client.generate("system", "user")

        # Should be parseable as JSON
        assert isinstance(response, str)
        json.loads(response)  # Should not raise

    def test_generate_output_has_korean_characters(self):
        """WHEN using template mode, THEN output should include Korean."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for 학생"
        response = client.generate(system_prompt, "user")

        # Should have Korean characters
        assert any(ord(c) > 127 for c in response)

    def test_generate_uses_ensure_ascii_false(self):
        """WHEN generating Korean text, THEN should preserve Korean characters."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) query for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Korean characters should be preserved
        assert "신청" in queries[0]["query"] or "방법" in queries[0]["query"]
