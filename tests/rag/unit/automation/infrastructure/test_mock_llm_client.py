"""
Comprehensive Unit Tests for MockLLMClientForQueryGen.

Tests for mock LLM client used in RAG testing automation.
Focuses on edge cases, error handling, and full code coverage.
"""

import json

from src.rag.automation.infrastructure.mock_llm_client import (
    MockLLMClientForQueryGen,
)


class TestMockLLMClientEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_generate_with_empty_system_prompt(self):
        """WHEN system_prompt is empty, THEN should use default counts."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("", "user message")
        queries = json.loads(response)

        # Should generate with default counts
        assert len(queries) == 4  # 1 Easy, 2 Medium, 1 Hard

    def test_generate_with_empty_user_message(self):
        """WHEN user_message is empty, THEN should still generate queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("system prompt", "")
        queries = json.loads(response)

        # Should work fine
        assert isinstance(queries, list)
        assert len(queries) > 0

    def test_generate_with_both_empty_prompts(self):
        """WHEN both prompts are empty, THEN should still generate queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("", "")
        queries = json.loads(response)

        # Should use defaults
        assert len(queries) == 4

    def test_generate_with_negative_temperature(self):
        """WHEN temperature is negative, THEN should be ignored."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user", temperature=-1.0)
        queries = json.loads(response)

        # Should work fine (temperature is ignored)
        assert isinstance(queries, list)

    def test_generate_with_high_temperature(self):
        """WHEN temperature > 1.0, THEN should be ignored."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user", temperature=2.5)
        queries = json.loads(response)

        # Should work fine (temperature is ignored)
        assert isinstance(queries, list)

    def test_generate_with_zero_temperature(self):
        """WHEN temperature is 0.0, THEN should work normally."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user", temperature=0.0)
        queries = json.loads(response)

        assert isinstance(queries, list)

    def test_generate_with_very_long_system_prompt(self):
        """WHEN system_prompt is very long, THEN should still extract counts."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        long_prompt = (
            "Generate "
            + "padding " * 1000
            + "Easy (3개), Medium (2개), Hard (1개) queries"
        )
        response = client.generate(long_prompt, "user")
        queries = json.loads(response)

        # Should still extract counts
        assert len(queries) == 6  # 3 Easy + 2 Medium + 1 Hard

    def test_generate_with_special_characters_in_prompt(self):
        """WHEN prompt has special characters, THEN should still work."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        prompt = "Generate Easy (1개) query with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        response = client.generate(prompt, "user")
        queries = json.loads(response)

        # Should generate valid queries
        assert isinstance(queries, list)

    def test_generate_with_newlines_in_prompt(self):
        """WHEN prompt has newlines, THEN should still extract counts."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        prompt = "Generate\nEasy (2개),\nMedium (3개)\nqueries"
        response = client.generate(prompt, "user")
        queries = json.loads(response)

        # Should handle newlines (with default for missing Hard)
        assert len(queries) == 6  # 2 Easy + 3 Medium + 1 Hard (default)

    def test_extract_count_with_multiple_matches(self):
        """WHEN prompt has multiple Easy patterns, THEN should extract first."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        # Should find the first match
        count = client._extract_count("Easy (3개) and also Easy (5개)", "Easy")

        # Regex finds first match
        assert count == 3

    def test_extract_count_case_sensitivity(self):
        """WHEN difficulty case varies, THEN should match exact case."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        # Should match exact case
        count_easy = client._extract_count("Easy (2개)", "Easy")
        count_easy_lower = client._extract_count("easy (2개)", "Easy")

        # Exact case match
        assert count_easy == 2
        # Case doesn't match, returns default
        assert count_easy_lower == 1

    def test_extract_count_with_korean_and_english(self):
        """WHEN prompt has both Korean and English, THEN should extract correctly."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Generate Easy (3개) 쉬운 쿼리", "Easy")

        assert count == 3

    def test_extract_count_with_zero_count(self):
        """WHEN count is 0, THEN should return 0."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Easy (0개)", "Easy")

        assert count == 0

    def test_extract_count_with_large_count(self):
        """WHEN count is large, THEN should return large number."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("Easy (100개)", "Easy")

        assert count == 100


class TestGetEmbeddingEdgeCases:
    """Test suite for embedding generation edge cases."""

    def test_get_embedding_with_whitespace_only(self):
        """WHEN text is whitespace only, THEN should generate embedding."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("   \t\n   ")

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_get_embedding_with_numbers_only(self):
        """WHEN text is numbers only, THEN should generate embedding."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("1234567890")

        assert len(embedding) == 384

    def test_get_embedding_with_special_chars_only(self):
        """WHEN text has special chars only, THEN should generate embedding."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("!@#$%^&*()")

        assert len(embedding) == 384

    def test_get_embedding_with_very_long_text(self):
        """WHEN text is very long, THEN should still generate 384-dim embedding."""
        client = MockLLMClientForQueryGen()

        long_text = "a" * 10000
        embedding = client.get_embedding(long_text)

        assert len(embedding) == 384

    def test_get_embedding_with_mixed_unicode(self):
        """WHEN text has mixed Unicode, THEN should work correctly."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("Hello 世界 안녕하세요 🎉")

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_get_embedding_distribution(self):
        """WHEN generating embedding, THEN values should be distributed."""
        client = MockLLMClientForQueryGen()

        embedding = client.get_embedding("test text")

        # Check that we have varied values (not all the same)
        unique_values = set(embedding)
        assert len(unique_values) > 10  # Should have many unique values

    def test_get_embedding_same_input_same_output(self):
        """WHEN calling multiple times, THEN same input produces same output."""
        client = MockLLMClientForQueryGen()

        text = "consistent test"
        embeddings = [client.get_embedding(text) for _ in range(5)]

        # All should be identical
        assert all(e == embeddings[0] for e in embeddings)


class TestRealisticResponsePersonaCombinations:
    """Test suite for persona combinations in realistic mode."""

    def test_professor_with_all_difficulties(self):
        """WHEN professor persona with all difficulties, THEN should generate appropriate queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개), Medium (1개), Hard (1개) for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        assert len(queries) == 3
        # Check professor-specific content
        assert any("교권" in q["query"] or "규정" in q["query"] for q in queries)

    def test_student_with_all_difficulties(self):
        """WHEN student persona with all difficulties, THEN should generate appropriate queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개), Medium (1개), Hard (1개) for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        assert len(queries) == 3
        # Check student-specific content
        assert any("신청" in q["query"] or "장학금" in q["query"] for q in queries)

    def test_staff_with_all_difficulties(self):
        """WHEN staff persona (no keyword) with all difficulties, THEN should generate staff queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개), Medium (1개), Hard (1개) queries"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        assert len(queries) == 3  # 1 Easy + 1 Medium + 1 Hard (explicit counts)
        # Check staff-specific content
        assert any("절차" in q["query"] or "근무" in q["query"] for q in queries)

    def test_english_professor(self):
        """WHEN using English Professor keyword, THEN should generate professor queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) for Professor"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate with default counts (1 Easy, 2 Medium, 1 Hard)
        assert len(queries) == 4
        # Professor query for easy difficulty
        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert "교권" in easy_queries[0]["query"]

    def test_english_student(self):
        """WHEN using English Student keyword, THEN should generate student queries."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개) for Student"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should generate with default counts (1 Easy, 2 Medium, 1 Hard)
        assert len(queries) == 4
        # Student query for easy difficulty
        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert "신청" in easy_queries[0]["query"]

    def test_both_korean_and_english_persona(self):
        """WHEN both Korean and English persona keywords, THEN should prioritize."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        # Korean keyword comes first
        system_prompt = "Generate Easy (1개) for 교수 Professor Student"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Should use professor (first match in code) with default counts
        assert len(queries) == 4  # Default counts
        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert "교권" in easy_queries[0]["query"]


class TestRealisticResponseQueryTypes:
    """Test suite for query type variations in realistic mode."""

    def test_easy_professor_fact_check_type(self):
        """WHEN generating easy professor queries, THEN should be fact_check type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (2개) for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only easy queries
        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert all(q["type"] == "fact_check" for q in easy_queries)
        assert len(easy_queries) == 2

    def test_easy_student_procedural_type(self):
        """WHEN generating easy student queries, THEN should be procedural type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (2개) for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only easy queries
        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert all(q["type"] == "procedural" for q in easy_queries)
        assert len(easy_queries) == 2

    def test_easy_staff_procedural_type(self):
        """WHEN generating easy staff queries, THEN should be procedural type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (2개) for staff"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only easy queries
        easy_queries = [q for q in queries if q["difficulty"] == "easy"]
        assert all(q["type"] == "procedural" for q in easy_queries)
        assert len(easy_queries) == 2

    def test_medium_professor_eligibility_type(self):
        """WHEN generating medium professor queries, THEN should be eligibility type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Medium (2개) for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only medium queries
        medium_queries = [q for q in queries if q["difficulty"] == "medium"]
        assert all(q["type"] == "eligibility" for q in medium_queries)
        assert len(medium_queries) == 2

    def test_medium_student_eligibility_type(self):
        """WHEN generating medium student queries, THEN should be eligibility type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Medium (2개) for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only medium queries
        medium_queries = [q for q in queries if q["difficulty"] == "medium"]
        assert all(q["type"] == "eligibility" for q in medium_queries)
        assert len(medium_queries) == 2

    def test_hard_professor_emotional_type(self):
        """WHEN generating hard professor queries, THEN should be emotional type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Hard (2개) for 교수"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only hard queries
        hard_queries = [q for q in queries if q["difficulty"] == "hard"]
        assert all(q["type"] == "emotional" for q in hard_queries)
        assert len(hard_queries) == 2

    def test_hard_student_complex_type(self):
        """WHEN generating hard student queries, THEN should be complex type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Hard (2개) for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        # Filter only hard queries
        hard_queries = [q for q in queries if q["difficulty"] == "hard"]
        assert all(q["type"] == "complex" for q in hard_queries)
        assert len(hard_queries) == 2

    def test_hard_staff_eligibility_type(self):
        """WHEN generating hard staff queries, THEN should be eligibility type."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Hard (2개)"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        hard_queries = [q for q in queries if q["difficulty"] == "hard"]
        assert all(q["type"] == "eligibility" for q in hard_queries)


class TestQueryStructureValidation:
    """Test suite for query structure validation."""

    def test_all_queries_have_required_fields(self):
        """WHEN generating queries, THEN all should have required fields."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (2개), Medium (2개), Hard (2개) for 학생"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        required_fields = {"query", "type", "difficulty"}
        assert all(set(q.keys()) == required_fields for q in queries)

    def test_query_field_is_string(self):
        """WHEN generating queries, THEN query field should be string."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate Easy (1개)", "user")
        queries = json.loads(response)

        assert all(isinstance(q["query"], str) for q in queries)

    def test_type_field_is_string(self):
        """WHEN generating queries, THEN type field should be string."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate Easy (1개)", "user")
        queries = json.loads(response)

        assert all(isinstance(q["type"], str) for q in queries)

    def test_difficulty_field_is_string(self):
        """WHEN generating queries, THEN difficulty field should be string."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate Easy (1개)", "user")
        queries = json.loads(response)

        assert all(isinstance(q["difficulty"], str) for q in queries)

    def test_query_field_not_empty(self):
        """WHEN generating queries, THEN query field should not be empty."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate Easy (1개)", "user")
        queries = json.loads(response)

        assert all(len(q["query"]) > 0 for q in queries)

    def test_valid_difficulty_values(self):
        """WHEN generating queries, THEN difficulty should be valid."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        system_prompt = "Generate Easy (1개), Medium (1개), Hard (1개)"
        response = client.generate(system_prompt, "user")
        queries = json.loads(response)

        valid_difficulties = {"easy", "medium", "hard"}
        assert all(q["difficulty"] in valid_difficulties for q in queries)


class TestCallCountBehavior:
    """Test suite for call count tracking behavior."""

    def test_call_count_starts_at_zero(self):
        """WHEN client is created, THEN call_count should be 0."""
        client = MockLLMClientForQueryGen()

        assert client.call_count == 0

    def test_call_count_increments_with_each_generate(self):
        """WHEN generate is called multiple times, THEN count should increment."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        for i in range(1, 6):
            client.generate("system", "user")
            assert client.call_count == i

    def test_reset_to_zero_after_multiple_calls(self):
        """WHEN reset after multiple calls, THEN count should return to 0."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        client.generate("system", "user")
        client.generate("system", "user")
        client.generate("system", "user")
        assert client.call_count == 3

        client.reset_call_count()
        assert client.call_count == 0

    def test_reset_multiple_times(self):
        """WHEN reset is called multiple times, THEN should stay at 0."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        client.generate("system", "user")
        client.reset_call_count()
        client.reset_call_count()
        client.reset_call_count()

        assert client.call_count == 0

    def test_call_count_independent_of_mode(self):
        """WHEN using different modes, THEN call_count should increment."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        client.generate("system", "user")
        assert client.call_count == 1

        client.use_template_responses = False
        client.generate("system", "user")
        assert client.call_count == 2


class TestSimpleResponseMode:
    """Test suite for simple response mode behavior."""

    def test_simple_mode_returns_same_response(self):
        """WHEN simple mode is used, THEN should return identical response."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response1 = client.generate("any system", "any user")
        response2 = client.generate("different system", "different user")

        assert response1 == response2

    def test_simple_mode_structure(self):
        """WHEN using simple mode, THEN should have correct structure."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("system", "user")
        queries = json.loads(response)

        # Should have exactly 2 queries
        assert len(queries) == 2

        # First query should be procedural (easy)
        assert queries[0]["type"] == "procedural"
        assert queries[0]["difficulty"] == "easy"

        # Second query should be eligibility (medium)
        assert queries[1]["type"] == "eligibility"
        assert queries[1]["difficulty"] == "medium"

    def test_simple_mode_ignores_counts_in_prompt(self):
        """WHEN using simple mode with counts, THEN should ignore counts."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("Easy (10개), Medium (10개)", "user")
        queries = json.loads(response)

        # Should still return 2 queries
        assert len(queries) == 2

    def test_simple_mode_ignores_persona_in_prompt(self):
        """WHEN using simple mode with persona, THEN should ignore persona."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response = client.generate("Generate for 교수 Professor", "user")
        queries = json.loads(response)

        # Should still return default simple queries
        assert len(queries) == 2


class TestDefaultCounts:
    """Test suite for default count behavior."""

    def test_default_counts_when_no_pattern(self):
        """WHEN prompt has no count patterns, THEN should use defaults."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate some test queries", "user")
        queries = json.loads(response)

        # Default: 1 Easy, 2 Medium, 1 Hard = 4 total
        assert len(queries) == 4

    def test_default_easy_count(self):
        """WHEN no Easy pattern, THEN default to 1."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("no pattern here", "Easy")

        assert count == 1

    def test_default_medium_count(self):
        """WHEN no Medium pattern, THEN default to 2."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("no pattern here", "Medium")

        assert count == 2

    def test_default_hard_count(self):
        """WHEN no Hard pattern, THEN default to 1."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        count = client._extract_count("no pattern here", "Hard")

        assert count == 1

    def test_partial_counts_use_defaults_for_missing(self):
        """WHEN only some counts specified, THEN use defaults for others."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        # Only Easy count specified
        response = client.generate("Generate Easy (3개)", "user")
        queries = json.loads(response)

        # Should have 3 Easy + 2 Medium (default) + 1 Hard (default) = 6
        assert len(queries) == 6


class TestJSONOutputFormat:
    """Test suite for JSON output format validation."""

    def test_json_is_valid_array(self):
        """WHEN generating response, THEN should be valid JSON array."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate Easy (1개)", "user")

        # Should not raise
        queries = json.loads(response)
        assert isinstance(queries, list)

    def test_json_preserves_korean_characters(self):
        """WHEN generating Korean queries, THEN JSON should preserve Korean."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate("Generate Easy (1개) for 학생", "user")
        queries = json.loads(response)

        # Korean characters should be readable
        assert "신청" in queries[0]["query"] or "방법" in queries[0]["query"]

    def test_json_all_queries_parseable(self):
        """WHEN generating multiple queries, THEN all should be parseable."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response = client.generate(
            "Generate Easy (3개), Medium (3개), Hard (3개)", "user"
        )
        queries = json.loads(response)

        assert len(queries) == 9
        assert all(isinstance(q, dict) for q in queries)


class TestMultipleGenerations:
    """Test suite for multiple generation calls."""

    def test_multiple_generations_with_different_prompts(self):
        """WHEN generating with different prompts, THEN should get different results."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response1 = client.generate("Generate Easy (1개) for 학생", "user")
        response2 = client.generate("Generate Easy (1개) for 교수", "user")

        queries1 = json.loads(response1)
        queries2 = json.loads(response2)

        # Should have different content
        assert queries1[0]["query"] != queries2[0]["query"]

    def test_multiple_generations_with_same_prompts_template_mode(self):
        """WHEN generating same prompt in template mode, THEN should get same results."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        prompt = "Generate Easy (2개) for 학생"
        response1 = client.generate(prompt, "user")
        response2 = client.generate(prompt, "user")

        assert response1 == response2

    def test_multiple_generations_with_same_prompts_simple_mode(self):
        """WHEN generating in simple mode, THEN should always get same results."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        response1 = client.generate("any", "any")
        response2 = client.generate("different", "prompts")

        assert response1 == response2

    def test_switching_modes(self):
        """WHEN switching between modes, THEN should use appropriate mode."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        response_template = client.generate("Generate Easy (1개)", "user")

        client.use_template_responses = False
        response_simple = client.generate("Generate Easy (1개)", "user")

        # Responses should be different
        assert response_template != response_simple

        # Parse to verify
        queries_template = json.loads(response_template)
        queries_simple = json.loads(response_simple)

        # Template mode should respect count (plus default for missing difficulties)
        assert len(queries_template) == 4  # 1 Easy + 2 Medium + 1 Hard
        # Simple mode should ignore count
        assert len(queries_simple) == 2


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_full_workflow_simple_mode(self):
        """WHEN using simple mode workflow, THEN should work end-to-end."""
        client = MockLLMClientForQueryGen(use_template_responses=False)

        # Generate queries
        response = client.generate("system prompt", "user message")
        queries = json.loads(response)

        # Verify structure
        assert len(queries) == 2
        assert client.call_count == 1

        # Reset and generate again
        client.reset_call_count()
        response2 = client.generate("system", "user")
        queries2 = json.loads(response2)

        # Should be identical
        assert queries == queries2
        assert client.call_count == 1

    def test_full_workflow_template_mode(self):
        """WHEN using template mode workflow, THEN should work end-to-end."""
        client = MockLLMClientForQueryGen(use_template_responses=True)

        # Generate for different personas
        student_response = client.generate(
            "Generate Easy (1개), Medium (1개) for 학생", "user"
        )
        professor_response = client.generate(
            "Generate Easy (1개), Medium (1개) for 교수", "user"
        )

        student_queries = json.loads(student_response)
        professor_queries = json.loads(professor_response)

        # Both should have 3 queries (1 Easy + 1 Medium + 1 Hard default)
        assert len(student_queries) == 3
        assert len(professor_queries) == 3

        # Content should differ
        assert student_queries[0]["query"] != professor_queries[0]["query"]

        # Call count should be 2
        assert client.call_count == 2

    def test_embedding_consistency_across_calls(self):
        """WHEN generating embeddings, THEN should be consistent."""
        client = MockLLMClientForQueryGen()

        texts = ["text1", "text2", "text1", "text3", "text2"]

        embeddings = [client.get_embedding(text) for text in texts]

        # Same text should produce same embedding
        assert embeddings[0] == embeddings[2]
        assert embeddings[1] == embeddings[4]

        # Different text should produce different embedding
        assert embeddings[0] != embeddings[1]
        assert embeddings[0] != embeddings[3]
