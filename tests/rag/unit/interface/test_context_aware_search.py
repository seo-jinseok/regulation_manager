from src.rag.interface.chat_logic import expand_followup_query

def test_expand_followup_query_with_article_reference():
    """Test that context is applied when user queries for a specific article."""
    context = "교원인사규정"
    assert expand_followup_query("제7조", context) == "교원인사규정 제7조"

def test_expand_followup_query_without_prefix():
    """Test that context is applied even when '제' prefix is missing."""
    context = "교원인사규정"
    # Current behavior: returns "7조" (no match)
    # Desired behavior: returns "교원인사규정 7조"
    assert expand_followup_query("7조", context) == "교원인사규정 7조"

def test_expand_followup_query_with_full_regulation_reference():
    """Test that context is NOT applied when user explicitly names a regulation."""
    context = "교원인사규정"
    assert expand_followup_query("학칙 제7조", context) == "학칙 제7조"

def test_expand_followup_query_with_explicit_context_switch():
    """Test that context is NOT applied when user switches context."""
    context = "교원인사규정"
    assert expand_followup_query("직원복무규정", context) == "직원복무규정"
