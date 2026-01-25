"""
Focused tests for formatters module to improve coverage from 86% toward 95%.
"""

from src.rag.interface.formatters import (
    _inject_tables,
    _is_repetitive_pattern,
    build_display_path,
    format_regulation_content,
    render_full_view_nodes,
    strip_path_prefix,
)


# Test build_display_path with text extraction (lines 181-185, 190-193)
def test_build_display_path_text_extraction():
    # Text path more detailed than parent
    result = build_display_path(
        chunk_parent_path=["규정명"],
        chunk_text="규정명 > 제1장 > 제1항: 내용",
        chunk_title="제목",
    )
    assert "제1항" in result

    # Text path with nested levels
    result = build_display_path(
        chunk_parent_path=["규정명"],
        chunk_text="규정명 > 부칙 > 별표 1: 내용",
        chunk_title="제목",
    )
    assert "별표 1" in result


# Test strip_path_prefix edge cases (lines 231, 237, 248, 255, 259)
def test_strip_path_prefix_edge_cases():
    # Empty text
    assert strip_path_prefix("", ["path"]) == ""

    # Empty parent_path
    assert strip_path_prefix("text", []) == "text"

    # Text without arrow
    assert strip_path_prefix("no arrow", ["path"]) == "no arrow"

    # Single part
    assert strip_path_prefix("single", ["path"]) == "single"

    # No match
    result = strip_path_prefix("text > with > arrows", ["nomatch"])
    assert "text > with > arrows" in result

    # Full match
    result = strip_path_prefix("A > B > C: content", ["A", "B", "C"])
    assert "C: content" in result

    # Partial match
    result = strip_path_prefix("A > B > C > D: content", ["A", "B"])
    assert "C > D: content" in result


# Test _inject_tables (lines 381, 383, 400, 402)
def test_inject_tables():
    # No tables
    assert _inject_tables("text", {}) == "text"
    assert _inject_tables("text", None) == "text"

    # Empty text
    assert _inject_tables("", {"tables": []}) == ""

    # Valid injection
    result = _inject_tables(
        "before [TABLE:1] after",
        {"tables": [{"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}]},
    )
    assert "[TABLE:1]" not in result
    assert "| A | B |" in result

    # Out of range
    result = _inject_tables("text [TABLE:5]", {"tables": [{"markdown": "| A |"}]})
    assert "[TABLE:5]" in result

    # No markdown field
    result = _inject_tables("text [TABLE:1]", {"tables": [{"format": "html"}]})
    assert "[TABLE:1]" in result

    # Empty markdown
    result = _inject_tables("text [TABLE:1]", {"tables": [{"markdown": ""}]})
    assert "[TABLE:1]" in result


# Test _is_repetitive_pattern
def test_is_repetitive_pattern():
    # Implementation date
    node = {"text": "이 규정은 2024년 1월 1일부터 시행한다."}
    assert _is_repetitive_pattern(node) is True

    # Implementation date with 변경
    node = {"text": "이 변경 규정은 2025년 3월 15일부터 시행한다."}
    assert _is_repetitive_pattern(node) is True

    # Article 시행일
    node = {"text": "제1조(시행일)"}
    assert _is_repetitive_pattern(node) is True

    # Non-repetitive
    node = {"text": "교원은 교육과 연구에 전념하여야 한다."}
    assert _is_repetitive_pattern(node) is False

    # Empty text
    node = {"text": ""}
    assert _is_repetitive_pattern(node) is False


# Test render_full_view_nodes (lines 437, 488, 503-508, 521, 547, 564-565, 575-583, 588, 591, 594, 597, 600)
def test_render_full_view_nodes_edge_cases():
    # Empty nodes
    assert render_full_view_nodes([]) == ""

    # Subitem type (inline format)
    nodes = [
        {
            "type": "subitem",
            "display_no": "(1)",
            "title": "",
            "text": "content",
            "children": [],
        }
    ]
    result = render_full_view_nodes(nodes)
    assert "(1) content" in result

    # Addendum item (inline format)
    nodes = [
        {
            "type": "addendum_item",
            "display_no": "1.",
            "title": "",
            "text": "content",
            "children": [],
        }
    ]
    result = render_full_view_nodes(nodes)
    assert "1. content" in result

    # Unknown type without title
    nodes = [
        {
            "type": "unknown",
            "display_no": "1.",
            "title": "",
            "text": "content",
            "children": [],
        }
    ]
    result = render_full_view_nodes(nodes)
    assert "1. content" in result

    # Unknown type with title
    nodes = [
        {
            "type": "unknown",
            "display_no": "제1조",
            "title": "목적",
            "text": "본문",
            "children": [],
        }
    ]
    result = render_full_view_nodes(nodes)
    assert "### 제1조 목적" in result

    # With children (max_items passed down)
    nodes = [
        {
            "type": "addendum",
            "display_no": "부칙",
            "title": "",
            "text": "",
            "children": [
                {
                    "type": "addendum_item",
                    "display_no": str(i),
                    "text": f"item {i}",
                    "children": [],
                }
                for i in range(20)
            ],
        }
    ]
    result = render_full_view_nodes(nodes, max_items=5)
    assert "중략" in result

    # Non-addendum node (max_items not passed)
    nodes = [
        {
            "type": "article",
            "display_no": "제1조",
            "title": "목적",
            "text": "본문",
            "children": [
                {
                    "type": "item",
                    "display_no": str(i),
                    "text": f"item {i}",
                    "children": [],
                }
                for i in range(20)
            ],
        }
    ]
    result = render_full_view_nodes(nodes, max_items=5)
    # Should NOT abbreviate children of non-addendum
    assert "item 1" in result
    assert "item 19" in result


# Test format_regulation_content (lines 564-565, 575-583, 588, 591, 594, 597, 600)
def test_format_regulation_content_indentation():
    # Empty
    assert format_regulation_content("") == ""

    # Paragraph (①) - 0 indent
    result = format_regulation_content("① 본문")
    lines = result.splitlines()
    assert not lines[0].startswith(" ")

    # Subparagraph (1.) - 2 spaces
    result = format_regulation_content("1. 본문")
    lines = result.splitlines()
    assert lines[0].startswith("  ")

    # Item (가.) - 5 spaces
    result = format_regulation_content("가. 본문")
    lines = result.splitlines()
    assert lines[0].startswith("     ")

    # Subitem number (1) - 8 spaces
    result = format_regulation_content("(1) 본문")
    lines = result.splitlines()
    assert lines[0].startswith("        ")

    # Subitem char (가) - 11 spaces
    result = format_regulation_content("(가) 본문")
    lines = result.splitlines()
    assert lines[0].startswith("           ")

    # Normalize "제 N조"
    result = format_regulation_content("제 1조 목적")
    assert "제1조" in result

    # Add dot to number
    result = format_regulation_content("1 본문")
    assert "1. 본문" in result

    # Empty lines preserved
    result = format_regulation_content("① 본문\n\n② 다음")
    assert "\n\n" in result
