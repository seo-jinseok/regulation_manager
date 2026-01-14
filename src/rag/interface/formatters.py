"""
Common formatting utilities for Regulation RAG System interfaces.

This module provides shared formatting functions used by CLI, Web UI, and MCP Server
to ensure consistent output formatting across all interfaces.

Functions:
    normalize_relevance_scores: Min-max scaling of relevance scores.
    filter_by_relevance: Filter results below threshold.
    get_relevance_label: Get emoji and label for relevance level.
    clean_path_segments: Remove duplicate segments from path.
    extract_display_text: Remove path prefix from text.
    get_confidence_info: Get icon, label, and description for confidence.
"""

import re
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from ..domain.entities import SearchResult


# ============================================================================
# Constants
# ============================================================================

# Default threshold for filtering low relevance results (10%)
DEFAULT_RELEVANCE_THRESHOLD = 0.10


# ============================================================================
# Score Normalization
# ============================================================================


def normalize_relevance_scores(sources: List["SearchResult"]) -> Dict[str, float]:
    """
    Normalize relevance scores using min-max scaling.

    Converts raw reranker scores to 0-1 range where:
    - 1.0 = highest score in the batch
    - 0.0 = lowest score in the batch

    Args:
        sources: List of SearchResult objects with score attributes.

    Returns:
        Dictionary mapping chunk.id to normalized score (0-1 range).
        Empty dict if sources is empty.
        All 1.0 if all scores are equal.
    """
    if not sources:
        return {}

    scores = [r.score for r in sources]
    max_s, min_s = max(scores), min(scores)

    if max_s == min_s:
        # All scores are equal -> treat as 100% relevance
        return {r.chunk.id: 1.0 for r in sources}

    return {r.chunk.id: (r.score - min_s) / (max_s - min_s) for r in sources}


def filter_by_relevance(
    sources: List["SearchResult"],
    norm_scores: Dict[str, float],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> List["SearchResult"]:
    """
    Filter results below the relevance threshold.

    Args:
        sources: List of SearchResult objects.
        norm_scores: Normalized scores from normalize_relevance_scores().
        threshold: Minimum normalized score to include (default: 0.10 = 10%).

    Returns:
        Filtered list of SearchResult objects.
    """
    return [r for r in sources if norm_scores.get(r.chunk.id, 0.0) >= threshold]


# ============================================================================
# Relevance Labels
# ============================================================================


def get_relevance_label(relevance_pct: int) -> Tuple[str, str]:
    """
    Get emoji icon and label for relevance percentage.

    Args:
        relevance_pct: Relevance percentage (0-100).

    Returns:
        Tuple of (icon, label). Example: ("🟢", "매우 높음")
    """
    if relevance_pct >= 80:
        return "🟢", "매우 높음"
    elif relevance_pct >= 50:
        return "🟡", "높음"
    elif relevance_pct >= 30:
        return "🟠", "보통"
    else:
        return "🔴", "낮음"


def get_relevance_label_combined(relevance_pct: int) -> str:
    """
    Get combined icon and label string for relevance percentage.

    Args:
        relevance_pct: Relevance percentage (0-100).

    Returns:
        Combined string. Example: "🟢 매우 높음"
    """
    icon, label = get_relevance_label(relevance_pct)
    return f"{icon} {label}"


# ============================================================================
# Path Formatting
# ============================================================================


def clean_path_segments(segments: List[str]) -> List[str]:
    """
    Remove duplicate path segments that differ only by whitespace.

    Handles cases like: ["부칙", "부 칙"] -> ["부칙"]

    Args:
        segments: List of path segment strings.

    Returns:
        Cleaned list with whitespace-only duplicates removed.
    """
    if not segments:
        return segments

    cleaned = [segments[0]]
    for seg in segments[1:]:
        # Normalize by removing all whitespace for comparison
        prev_normalized = cleaned[-1].replace(" ", "").replace("　", "")
        curr_normalized = seg.replace(" ", "").replace("　", "")

        # Skip if same as previous (only whitespace differs)
        if prev_normalized != curr_normalized:
            cleaned.append(seg)

    return cleaned


def build_display_path(
    chunk_parent_path: List[str],
    chunk_text: str,
    chunk_title: str,
) -> str:
    """
    Build a clean display path from chunk metadata.

    Combines parent_path with text-embedded path info, removing duplicates.

    Args:
        chunk_parent_path: Parent path from chunk metadata.
        chunk_text: Full text of the chunk (may contain path prefix).
        chunk_title: Title of the chunk.

    Returns:
        Cleaned path string like "규정명 > 조항 > 항목".
    """
    cleaned_segments = (
        clean_path_segments(chunk_parent_path) if chunk_parent_path else []
    )

    # Extract path from text if available (format: "path: content")
    text_path_match = re.match(r"^([^:]+):\s*", chunk_text)
    if text_path_match:
        text_path = text_path_match.group(1).strip()
        text_segments = [s.strip() for s in text_path.split(">")]
        # Use text path if it's more detailed than parent_path
        if len(text_segments) > len(cleaned_segments):
            cleaned_segments = clean_path_segments(text_segments)

    # Ensure regulation name is at the beginning
    reg_name = chunk_parent_path[0] if chunk_parent_path else chunk_title
    if cleaned_segments and reg_name and cleaned_segments[0] != reg_name:
        first_normalized = cleaned_segments[0].replace(" ", "")
        reg_normalized = reg_name.replace(" ", "")
        if first_normalized != reg_normalized:
            cleaned_segments = [reg_name] + cleaned_segments

    return " > ".join(cleaned_segments) if cleaned_segments else chunk_title


# ============================================================================
# Text Formatting
# ============================================================================


def extract_display_text(text: str) -> str:
    """
    Remove path prefix from text to avoid duplication in display.

    Handles text format: "규정명 > 조항 > 항목: 본문 내용"
    -> Returns: "본문 내용"

    Args:
        text: Full text including potential path prefix.

    Returns:
        Text with path prefix removed.
    """
    # Remove leading path pattern (e.g., "규정명 > 조항 > 항목: ")
    display_text = re.sub(r"^[^:]+:\s*", "", text)
    # Clean up remaining format (e.g., "1.:" -> "1.")
    display_text = re.sub(r"(\d+)\.\s*:", r"\1.", display_text)
    return display_text


def strip_path_prefix(text: str, parent_path: List[str]) -> str:
    """
    Strip a leading path prefix from text when it matches the parent path.

    Handles cases like:
    "규정명 > 부칙 > 부 칙 > 2. ..." -> "2. ..."
    """
    if not text or not parent_path:
        return text
    if ">" not in text:
        return text

    parts = [part.strip() for part in text.split(">")]
    if len(parts) <= 1:
        return text

    def normalize(segment: str) -> str:
        return segment.replace(" ", "").replace("　", "")

    normalized_parts = [normalize(part) for part in parts]
    candidates = [parent_path, clean_path_segments(parent_path)]
    best_match = 0

    for candidate in candidates:
        if not candidate:
            continue
        normalized_candidate = [normalize(part) for part in candidate]
        match_len = 0
        for part, target in zip(normalized_parts, normalized_candidate, strict=False):
            if part == target:
                match_len += 1
            else:
                break
        best_match = max(best_match, match_len)

    if best_match == 0:
        return text

    remainder = " > ".join(parts[best_match:]).strip()
    return remainder or text


def infer_regulation_title_from_tables(
    tables: List[object],
    fallback: str,
) -> str:
    """Infer regulation title from table paths, falling back when missing."""
    for table in tables or []:
        path = None
        if isinstance(table, dict):
            path = table.get("path")
        else:
            path = getattr(table, "path", None)
        if path:
            title = str(path[0]).strip()
            if title:
                return title
    return fallback


_ATTACHMENT_LABEL_PATTERN = re.compile(
    r"(별표|별첨|별지)\s*(?:제\s*)?(\d+)\s*(?:호|번)?"
)


def infer_attachment_label(match: object, fallback_label: str) -> str:
    """Infer attachment label with number from table metadata when available."""
    candidates: List[str] = []

    if isinstance(match, dict):
        path = match.get("path") or []
        candidates.append(" ".join(str(part) for part in path if part))
        for key in ("display_no", "title", "text"):
            value = match.get(key)
            if value:
                candidates.append(str(value))
        table_index = match.get("table_index")
    else:
        path = getattr(match, "path", None) or []
        candidates.append(" ".join(str(part) for part in path if part))
        for attr in ("display_no", "title", "text"):
            value = getattr(match, attr, None)
            if value:
                candidates.append(str(value))
        table_index = getattr(match, "table_index", None)

    for candidate in candidates:
        if not candidate:
            continue
        found = _ATTACHMENT_LABEL_PATTERN.search(candidate)
        if found:
            label = found.group(1)
            number = found.group(2)
            return f"{label} {number}"

    if isinstance(table_index, int) and table_index > 0:
        return f"{fallback_label} {table_index}"
    return fallback_label


# ============================================================================
# Full View Rendering
# ============================================================================

_INLINE_NODE_TYPES = {"paragraph", "item", "subitem", "addendum_item"}


def normalize_markdown_table(markdown: str) -> str:
    """Normalize markdown tables with blank header rows."""
    if not markdown:
        return markdown

    lines = [line.rstrip() for line in markdown.strip().splitlines()]
    if len(lines) < 2:
        return markdown

    if not _is_table_row(lines[0]) or not _is_table_separator_row(lines[1]):
        return markdown

    header_cells = _split_table_row(lines[0])
    if not header_cells or any(cell.strip() for cell in header_cells):
        return markdown

    for idx in range(2, len(lines)):
        if not _is_table_row(lines[idx]):
            continue
        data_cells = _split_table_row(lines[idx])
        if not data_cells or not any(cell.strip() for cell in data_cells):
            continue
        lines[0] = _format_table_row(data_cells, len(header_cells))
        del lines[idx]
        break

    return "\n".join(lines)


def _is_table_row(line: str) -> bool:
    return "|" in line


def _is_table_separator_row(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    if not stripped:
        return False
    parts = [part.strip() for part in stripped.split("|")]
    if not parts:
        return False
    return all(re.match(r"^:?-{3,}:?$", part) or part == "" for part in parts)


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _format_table_row(cells: List[str], width: int) -> str:
    padded = list(cells)
    if width > len(padded):
        padded.extend([""] * (width - len(padded)))
    if width < len(padded):
        padded = padded[:width]
    return "| " + " | ".join(padded) + " |"


def _inject_tables(text: str, metadata: Dict[str, object]) -> str:
    tables = metadata.get("tables") if isinstance(metadata, dict) else None
    if not text or not tables:
        return text

    def replace(match: re.Match) -> str:
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(tables):
            return match.group(0)
        table = tables[index]
        if isinstance(table, dict):
            markdown = table.get("markdown")
        else:
            markdown = None
        if not markdown:
            return match.group(0)
        normalized = normalize_markdown_table(markdown)
        return f"\n\n{normalized.strip()}\n\n"

    return re.sub(r"\[TABLE:(\d+)\]", replace, text)


def _is_repetitive_pattern(node: dict) -> bool:
    """Check if a node follows a repetitive implementation date pattern."""
    text = str(node.get("text") or "").strip()
    # Pattern: "이 (변경) 규정은 YYYY년 M월 D일부터 시행한다."
    patterns = [
        r"이\s*(?:변경\s*)?규정은\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일부터\s*시행한다",
        r"제\d+조\s*\(시행일\)",
    ]
    return any(re.search(p, text) for p in patterns)


def render_full_view_nodes(
    nodes: List[dict],
    depth: int = 0,
    max_items: int = 0,
) -> str:
    """
    Render regulation nodes for full-view display.

    Uses inline formatting for numbered 항/호/목 to avoid line breaks after
    display numbers (e.g., "1. 교수 : 정").

    Args:
        nodes: List of regulation nodes to render.
        depth: Current recursion depth for heading levels.
        max_items: If > 0 and len(nodes) > max_items, abbreviates the middle part.
    """
    if not nodes:
        return ""

    display_nodes = nodes
    total_count = len(nodes)

    # Smart Abbreviation: Detect repetitive implementation date patterns
    # If more than 5 items and many are implementation dates, we can be more aggressive
    repetitive_indices = [i for i, n in enumerate(nodes) if _is_repetitive_pattern(n)]

    # If we have many repetitive items (>= 3), abbreviate them specifically
    if len(repetitive_indices) >= 5 and max_items > 0:
        # Keep first 2 and last 1 of the whole list, OR respect max_items
        keep_top = 3
        keep_bottom = 1
        if total_count > (keep_top + keep_bottom + 1):
            display_nodes = (
                nodes[:keep_top]
                + [
                    {
                        "type": "abbreviation",
                        "count": total_count - (keep_top + keep_bottom),
                    }
                ]
                + nodes[-keep_bottom:]
            )

    # Fallback to simple count-based abbreviation if not already abbreviated
    elif max_items > 0 and total_count > max_items:
        keep = max(1, max_items // 2)
        display_nodes = (
            nodes[:keep]
            + [{"type": "abbreviation", "count": total_count - (keep + 1)}]
            + [nodes[-1]]
        )

    lines = []
    for node in display_nodes:
        if node.get("type") == "abbreviation":
            count = node.get("count", 0)
            lines.append(f"\n*... (중략: {count}개 항목) ...*\n")
            continue

        display_no = str(node.get("display_no") or "").strip()
        title = str(node.get("title") or "").strip()
        text = str(node.get("text") or "").strip()
        text = _inject_tables(text, node.get("metadata") or {})
        node_type = node.get("type")

        if display_no and text and (node_type in _INLINE_NODE_TYPES or not title):
            parts = [display_no]
            if title:
                parts.append(title)
            parts.append(text)
            lines.append(" ".join(parts))
        else:
            label = f"{display_no} {title}".strip() if display_no or title else ""
            if label:
                heading = "#" * min(6, depth + 3)
                lines.append(f"{heading} {label}")
            if text:
                lines.append(text)

        children = node.get("children", []) or []
        if children:
            # Pass max_items down but reduced or specific for sub-levels
            # For addenda, we typically want abbreviation at the first level of addenda_item
            sub_max = max_items if node_type == "addendum" else 0
            rendered_children = render_full_view_nodes(
                children, depth + 1, max_items=sub_max
            )
            if rendered_children:
                lines.append(rendered_children)

    return "\n\n".join([line for line in lines if line])


# ============================================================================
# Markdown Normalization
# ============================================================================


def normalize_markdown_emphasis(text: str) -> str:
    """Normalize emphasis to render bold correctly when wrapped in quotes."""
    if not text:
        return text

    replacements = [
        (r'\*\*"([^"]+)"\*\*', r'"**\1**"'),
        (r"\*\*'([^']+)'\*\*", r"'**\1**'"),
        (r"\*\*“([^”]+)”\*\*", r"“**\1**”"),
        (r"\*\*‘([^’]+)’\*\*", r"‘**\1**’"),
    ]
    normalized = text
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized)
    return normalized


def format_regulation_content(text: str) -> str:
    """
    Format regulation content with proper indentation for CLI display.

    Applies hierarchical indentation:
    - Paragraph (①, ②...): 0 spaces
    - Subparagraph (1., 2....): 2 spaces
    - Item (가., 나....): 4 spaces
    - Subitem (1), (2)...: 6 spaces
    - Subitem (가), (나)...: 8 spaces
    """
    if not text:
        return text

    lines = text.splitlines()
    formatted = []

    # Regex patterns for hierarchy
    p_paragraph = re.compile(r"^\s*([①-⑮])")
    # Match "1." or "1 " (digit + dot OR space)
    p_subparagraph = re.compile(r"^\s*(\d+(?:\.|\s))")
    # Match "가." or "가 " (char + dot OR space)
    p_item = re.compile(r"^\s*([가-하](?:\.|\s))")
    p_subitem_num = re.compile(r"^\s*(\(\d+\))")
    p_subitem_char = re.compile(r"^\s*(\([가-하]\))")

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            formatted.append("")
            continue

        # Normalize "제 N조" -> "제N조"
        clean_line = re.sub(r"제\s+(\d+)조", r"제\1조", line)
        stripped = clean_line.lstrip()

        # Normalize "1 ???" -> "1. ???" (Add dot if missing for numbering)
        # Check if line starts with number followed by space (and NO dot)
        match_num = re.match(r"^(\s*)(\d+)\s+([^.])", clean_line)
        if match_num:
            prefix_space = match_num.group(1)
            number = match_num.group(2)
            rest = match_num.group(3)
            full_rest = clean_line[match_num.end(2) :]

            # Apply generally for regulation numbering (typically 1-3 digits)
            if len(number) <= 3:
                clean_line = f"{prefix_space}{number}. {full_rest.lstrip()}"
                stripped = clean_line.lstrip()

        # Re-check pattern matches on normalized line
        if p_paragraph.match(stripped):
            # Paragraph (①): No indent
            formatted.append(stripped)
        elif p_subparagraph.match(stripped):
            # Subparagraph (1.): 2 Spaces
            formatted.append("  " + stripped)
        elif p_item.match(stripped):
            # Item (가.): 5 Spaces (2 + 3)
            formatted.append("     " + stripped)
        elif p_subitem_num.match(stripped):
            # Subitem (1): 8 Spaces (5 + 3)
            formatted.append("        " + stripped)
        elif p_subitem_char.match(stripped):
            # Subitem (가): 11 Spaces (8 + 3)
            formatted.append("           " + stripped)
        else:
            # Continuation line
            formatted.append(clean_line)

    return "\n".join(formatted)


# ============================================================================
# Confidence Info
# ============================================================================


def get_confidence_info(confidence: float) -> Tuple[str, str, str]:
    """
    Get icon, label, and description for confidence level.

    Args:
        confidence: Confidence score (0.0 - 1.0).

    Returns:
        Tuple of (icon, label, description).
        Example: ("🟢", "높음", "검색된 규정이 질문과 높은 관련성을 보입니다.")
    """
    if confidence >= 0.7:
        return (
            "🟢",
            "높음",
            "검색된 규정이 질문과 높은 관련성을 보입니다. 답변을 신뢰할 수 있습니다.",
        )
    elif confidence >= 0.4:
        return (
            "🟡",
            "보통",
            "관련 규정을 찾았지만, 중요한 결정은 위 규정 원문을 직접 확인하세요.",
        )
    else:
        return (
            "🔴",
            "낮음",
            "관련 규정을 찾기 어렵습니다. 학교 행정실이나 규정집을 직접 확인하세요.",
        )


# ============================================================================
# Search Result Explanation
# ============================================================================


def format_search_result_with_explanation(
    result: "SearchResult",
    query: str,
    show_score: bool = False,
) -> Tuple[str, str]:
    """
    Generate explanation for why a search result matched.

    Args:
        result: SearchResult object with chunk and score.
        query: Original search query.
        show_score: Whether to include raw AI score (for debug).

    Returns:
        Tuple of (explanation_line, matched_keywords_str).
        - explanation_line: Single line with icons and metadata.
        - matched_keywords_str: Comma-separated matched keywords.

    Example:
        ("💡 매칭 키워드: 연구년, 신청 | 📄 제15조 | 교원인사규정 > 연구년제", "연구년, 신청")
    """
    parts = []
    matched_keywords = []

    # 1. Extract matched keywords from chunk.keywords that appear in query
    if hasattr(result.chunk, "keywords") and result.chunk.keywords:
        query_lower = query.lower()
        for kw in result.chunk.keywords:
            term = kw.term if hasattr(kw, "term") else str(kw)
            # Check if keyword or any part of it is in query
            if term.lower() in query_lower or any(
                q_word in term.lower() for q_word in query_lower.split()
            ):
                matched_keywords.append(term)

    matched_keywords_str = ", ".join(matched_keywords[:5])  # Limit to 5 keywords

    if matched_keywords_str:
        parts.append(f"💡 매칭 키워드: {matched_keywords_str}")

    # 2. Extract article number if ARTICLE level
    if hasattr(result.chunk, "level"):
        from ..domain.entities import ChunkLevel

        if result.chunk.level == ChunkLevel.ARTICLE:
            import re

            article_no = None

            # Priority 1: Use display_no if present (e.g., "제15조")
            display_no = getattr(result.chunk, "display_no", None)
            if display_no:
                display_match = re.search(r"제\s*(\d+)\s*조", display_no)
                if display_match:
                    article_no = display_match.group(1)

            # Priority 2: Fall back to parsing title
            if not article_no:
                title = getattr(result.chunk, "title", "") or ""
                title_match = re.search(r"제\s*(\d+)\s*조", title)
                if title_match:
                    article_no = title_match.group(1)

            # Priority 3: Fall back to parsing text
            if not article_no:
                text = getattr(result.chunk, "text", "") or ""
                text_match = re.search(r"제\s*(\d+)\s*조", text)
                if text_match:
                    article_no = text_match.group(1)

            if article_no:
                parts.append(f"📄 제{article_no}조")

    # 3. Build path info
    if hasattr(result.chunk, "parent_path") and result.chunk.parent_path:
        path_segments = clean_path_segments(result.chunk.parent_path)
        # Show last 2 segments for brevity
        path_short = (
            " > ".join(path_segments[-2:])
            if len(path_segments) > 1
            else (path_segments[0] if path_segments else "")
        )
        if path_short:
            parts.append(path_short)

    # 4. Add AI confidence score if debug mode
    if show_score:
        parts.append(f"🎯 AI 신뢰도: {result.score:.3f}")

    # Combine parts with separator
    explanation = " | ".join(parts) if parts else "관련 내용"

    return explanation, matched_keywords_str
