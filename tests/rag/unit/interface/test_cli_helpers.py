from src.rag.interface.cli import _sanitize_query_input


def test_sanitize_query_input_removes_control_chars():
    assert _sanitize_query_input("\x7f") == ""
    assert _sanitize_query_input("휴학\x08 관련") == "휴 관련"
    assert _sanitize_query_input("교교\b원인사규정") == "교원인사규정"


def test_sanitize_query_input_trims_whitespace():
    assert _sanitize_query_input("   휴학 관련   ") == "휴학 관련"
    assert _sanitize_query_input("\n\t") == ""
