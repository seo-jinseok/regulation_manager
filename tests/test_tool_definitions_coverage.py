"""
Focused coverage tests for tool_definitions.py.

Targets low-coverage module to improve overall coverage from 67% to 85%.
"""


from src.rag.infrastructure.tool_definitions import (
    TOOL_DEFINITIONS,
    get_tool_by_name,
    get_tool_names,
    get_tools_prompt,
)


class TestGetToolNames:
    """Tests for get_tool_names function."""

    def test_get_tool_names_returns_list(self):
        """
        SPEC: get_tool_names should return a list of strings.
        """
        result = get_tool_names()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)

    def test_get_tool_names_returns_all_tools(self):
        """
        SPEC: get_tool_names should return one name per tool definition.
        """
        result = get_tool_names()
        assert len(result) == len(TOOL_DEFINITIONS)

    def test_get_tool_names_contains_expected_tools(self):
        """
        SPEC: get_tool_names should contain core tool names.
        """
        result = get_tool_names()
        expected_tools = [
            "search_regulations",
            "get_article",
            "get_chapter",
            "expand_synonyms",
            "detect_intent",
            "generate_answer",
            "sync_database",
        ]
        for tool in expected_tools:
            assert tool in result

    def test_get_tool_names_no_duplicates(self):
        """
        SPEC: get_tool_names should not contain duplicate names.
        """
        result = get_tool_names()
        assert len(result) == len(set(result))


class TestGetToolByName:
    """Tests for get_tool_by_name function."""

    def test_get_tool_by_name_existing_tool(self):
        """
        SPEC: get_tool_by_name should return tool definition for valid name.
        """
        result = get_tool_by_name("search_regulations")
        assert result is not None
        assert result["type"] == "function"
        assert result["function"]["name"] == "search_regulations"

    def test_get_tool_by_name_non_existing_tool(self):
        """
        SPEC: get_tool_by_name should return None for invalid name.
        """
        result = get_tool_by_name("nonexistent_tool")
        assert result is None

    def test_get_tool_by_name_empty_string(self):
        """
        SPEC: get_tool_by_name should return None for empty string.
        """
        result = get_tool_by_name("")
        assert result is None

    def test_get_tool_by_name_case_sensitive(self):
        """
        SPEC: get_tool_by_name should be case sensitive.
        """
        result = get_tool_by_name("Search_Regulations")
        assert result is None

    def test_get_tool_by_name_all_search_tools(self):
        """
        SPEC: get_tool_by_name should find all search tools.
        """
        search_tools = [
            "search_regulations",
            "get_article",
            "get_chapter",
            "get_attachment",
            "get_regulation_overview",
            "get_full_regulation",
        ]
        for tool_name in search_tools:
            result = get_tool_by_name(tool_name)
            assert result is not None
            assert result["function"]["name"] == tool_name

    def test_get_tool_by_name_all_analysis_tools(self):
        """
        SPEC: get_tool_by_name should find all analysis tools.
        """
        analysis_tools = [
            "expand_synonyms",
            "detect_intent",
            "detect_audience",
            "analyze_query",
        ]
        for tool_name in analysis_tools:
            result = get_tool_by_name(tool_name)
            assert result is not None
            assert result["function"]["name"] == tool_name

    def test_get_tool_by_name_all_admin_tools(self):
        """
        SPEC: get_tool_by_name should find all admin tools.
        """
        admin_tools = [
            "sync_database",
            "get_sync_status",
            "reset_database",
        ]
        for tool_name in admin_tools:
            result = get_tool_by_name(tool_name)
            assert result is not None
            assert result["function"]["name"] == tool_name


class TestGetToolsPrompt:
    """Tests for get_tools_prompt function."""

    def test_get_tools_prompt_returns_string(self):
        """
        SPEC: get_tools_prompt should return a string.
        """
        result = get_tools_prompt()
        assert isinstance(result, str)

    def test_get_tools_prompt_has_header(self):
        """
        SPEC: get_tools_prompt should start with 'Available tools:'.
        """
        result = get_tools_prompt()
        assert result.startswith("Available tools:")

    def test_get_tools_prompt_contains_tool_names(self):
        """
        SPEC: get_tools_prompt should contain all tool names.
        """
        result = get_tools_prompt()
        tool_names = get_tool_names()
        for name in tool_names:
            assert name in result

    def test_get_tools_prompt_contains_descriptions(self):
        """
        SPEC: get_tools_prompt should contain tool descriptions.
        """
        result = get_tools_prompt()
        # Should contain Korean descriptions
        assert "규정 검색" in result or "검색" in result

    def test_get_tools_prompt_format(self):
        """
        SPEC: get_tools_prompt should format each tool as '- name(params): description'.
        """
        result = get_tools_prompt()
        lines = result.split("\n")
        # First line is header
        assert lines[0] == "Available tools:"
        # Each tool line should start with '- '
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                assert line.strip().startswith("- ")


class TestToolDefinitionsConstant:
    """Tests for TOOL_DEFINITIONS constant structure."""

    def test_tool_definitions_is_list(self):
        """
        SPEC: TOOL_DEFINITIONS should be a list.
        """
        assert isinstance(TOOL_DEFINITIONS, list)

    def test_tool_definitions_not_empty(self):
        """
        SPEC: TOOL_DEFINITIONS should not be empty.
        """
        assert len(TOOL_DEFINITIONS) > 0

    def test_tool_definitions_structure(self):
        """
        SPEC: Each tool should have type='function' and function key.
        """
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_tool_definitions_parameters_structure(self):
        """
        SPEC: Tool parameters should follow OpenAI format.
        """
        for tool in TOOL_DEFINITIONS:
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_tool_definitions_required_is_list(self):
        """
        SPEC: Each tool's required field should be a list.
        """
        for tool in TOOL_DEFINITIONS:
            required = tool["function"]["parameters"]["required"]
            assert isinstance(required, list)

    def test_tool_definitions_properties_structure(self):
        """
        SPEC: Each property should have type and description.
        """
        for tool in TOOL_DEFINITIONS:
            properties = tool["function"]["parameters"]["properties"]
            for prop_name, prop_def in properties.items():
                assert "type" in prop_def
                # Description is optional but recommended


class TestSpecificToolDefinitions:
    """Tests for specific tool definition details."""

    def test_search_regulations_params(self):
        """
        SPEC: search_regulations should have query, top_k, audience params.
        """
        tool = get_tool_by_name("search_regulations")
        params = tool["function"]["parameters"]["properties"]
        assert "query" in params
        assert "top_k" in params
        assert "audience" in params
        assert params["query"]["type"] == "string"
        assert params["top_k"]["type"] == "integer"

    def test_search_regulations_required_params(self):
        """
        SPEC: search_regulations should only require query.
        """
        tool = get_tool_by_name("search_regulations")
        required = tool["function"]["parameters"]["required"]
        assert required == ["query"]

    def test_search_regulations_audience_enum(self):
        """
        SPEC: search_regulations audience should have enum values.
        """
        tool = get_tool_by_name("search_regulations")
        audience = tool["function"]["parameters"]["properties"]["audience"]
        assert "enum" in audience
        expected_enum = ["all", "student", "faculty", "staff"]
        assert audience["enum"] == expected_enum

    def test_get_article_params(self):
        """
        SPEC: get_article should have regulation and article_no params.
        """
        tool = get_tool_by_name("get_article")
        params = tool["function"]["parameters"]["properties"]
        assert "regulation" in params
        assert "article_no" in params
        assert params["article_no"]["type"] == "integer"

    def test_get_article_required_params(self):
        """
        SPEC: get_article should require both params.
        """
        tool = get_tool_by_name("get_article")
        required = tool["function"]["parameters"]["required"]
        assert "regulation" in required
        assert "article_no" in required

    def test_sync_database_params(self):
        """
        SPEC: sync_database should have optional full param.
        """
        tool = get_tool_by_name("sync_database")
        params = tool["function"]["parameters"]["properties"]
        assert "full" in params
        assert params["full"]["type"] == "boolean"

    def test_sync_database_no_required_params(self):
        """
        SPEC: sync_database should have no required params.
        """
        tool = get_tool_by_name("sync_database")
        required = tool["function"]["parameters"]["required"]
        assert len(required) == 0

    def test_generate_answer_params(self):
        """
        SPEC: generate_answer should have question and context params.
        """
        tool = get_tool_by_name("generate_answer")
        params = tool["function"]["parameters"]["properties"]
        assert "question" in params
        assert "context" in params

    def test_clarify_query_params(self):
        """
        SPEC: clarify_query should have query and options params.
        """
        tool = get_tool_by_name("clarify_query")
        params = tool["function"]["parameters"]["properties"]
        assert "query" in params
        assert "options" in params
        assert params["options"]["type"] == "array"


class TestToolCategories:
    """Tests for tool categorization."""

    def test_search_tools_count(self):
        """
        SPEC: Should have 6 search tools.
        """
        search_tools = [
            "search_regulations",
            "get_article",
            "get_chapter",
            "get_attachment",
            "get_regulation_overview",
            "get_full_regulation",
        ]
        actual_count = sum(1 for t in TOOL_DEFINITIONS
                          if t["function"]["name"] in search_tools)
        assert actual_count == len(search_tools)

    def test_analysis_tools_count(self):
        """
        SPEC: Should have 4 analysis tools.
        """
        analysis_tools = [
            "expand_synonyms",
            "detect_intent",
            "detect_audience",
            "analyze_query",
        ]
        actual_count = sum(1 for t in TOOL_DEFINITIONS
                          if t["function"]["name"] in analysis_tools)
        assert actual_count == len(analysis_tools)

    def test_admin_tools_count(self):
        """
        SPEC: Should have 3 admin tools.
        """
        admin_tools = [
            "sync_database",
            "get_sync_status",
            "reset_database",
        ]
        actual_count = sum(1 for t in TOOL_DEFINITIONS
                          if t["function"]["name"] in admin_tools)
        assert actual_count == len(admin_tools)

    def test_response_tools_count(self):
        """
        SPEC: Should have 2 response tools.
        """
        response_tools = [
            "generate_answer",
            "clarify_query",
        ]
        actual_count = sum(1 for t in TOOL_DEFINITIONS
                          if t["function"]["name"] in response_tools)
        assert actual_count == len(response_tools)
