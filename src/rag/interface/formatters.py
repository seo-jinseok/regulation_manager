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
from typing import Dict, List, Tuple, TYPE_CHECKING

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
    
    return {
        r.chunk.id: (r.score - min_s) / (max_s - min_s) 
        for r in sources
    }


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
    return [
        r for r in sources 
        if norm_scores.get(r.chunk.id, 0.0) >= threshold
    ]


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
    cleaned_segments = clean_path_segments(chunk_parent_path) if chunk_parent_path else []
    
    # Extract path from text if available (format: "path: content")
    text_path_match = re.match(r'^([^:]+):\s*', chunk_text)
    if text_path_match:
        text_path = text_path_match.group(1).strip()
        text_segments = [s.strip() for s in text_path.split('>')]
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
    display_text = re.sub(r'^[^:]+:\s*', '', text)
    # Clean up remaining format (e.g., "1.:" -> "1.")
    display_text = re.sub(r'(\d+)\.\s*:', r'\1.', display_text)
    return display_text


# ============================================================================
# Full View Rendering
# ============================================================================

_INLINE_NODE_TYPES = {"paragraph", "item", "subitem", "addendum_item"}


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
        return f"\n\n{markdown.strip()}\n\n"

    return re.sub(r"\[TABLE:(\d+)\]", replace, text)


def render_full_view_nodes(nodes: List[dict], depth: int = 0) -> str:
    """
    Render regulation nodes for full-view display.

    Uses inline formatting for numbered 항/호/목 to avoid line breaks after
    display numbers (e.g., "1. 교수 : 정").
    """
    lines = []
    for node in nodes:
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
            lines.append(render_full_view_nodes(children, depth + 1))

    return "\n\n".join([line for line in lines if line])


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
            "검색된 규정이 질문과 높은 관련성을 보입니다. 답변을 신뢰할 수 있습니다."
        )
    elif confidence >= 0.4:
        return (
            "🟡",
            "보통",
            "관련 규정을 찾았지만, 중요한 결정은 위 규정 원문을 직접 확인하세요."
        )
    else:
        return (
            "🔴",
            "낮음",
            "관련 규정을 찾기 어렵습니다. 학교 행정실이나 규정집을 직접 확인하세요."
        )
