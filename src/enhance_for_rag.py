"""
RAG 최적화를 위한 규정집 JSON 후처리 스크립트.

이 스크립트는 기존 규정집 JSON 파일에 다음 필드들을 추가합니다:
- parent_path: 노드의 계층 경로 (breadcrumb)
- full_text: 벡터 임베딩용 self-contained 텍스트
- keywords: 본문에서 추출한 핵심 키워드
- status: 규정 상태 (active/abolished)
- amendment_history: 개정/신설/삭제 이력
- is_index_duplicate: 중복 인덱스 플래그

Usage:
    # 모듈로 실행 (권장)
    uv run python -m src.enhance_for_rag data/output/규정집.json -o data/output/규정집_rag.json
    uv run python -m src.enhance_for_rag data/output/규정집.json --sample 3

    # 스크립트로 직접 실행
    uv run python src/enhance_for_rag.py data/output/규정집.json -o data/output/규정집_rag.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

# Regex patterns for amendment history extraction
AMENDMENT_PATTERNS = [
    # (개정 2006.11.06.) or (개정 2006. 11. 06.)
    (r"\(개정\s*(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*\)", "개정"),
    # (신설 2018.02.08.)
    (r"\(신설\s*(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*\)", "신설"),
    # (삭제 2008.01.29.)
    (r"\(삭제\s*(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*\)", "삭제"),
    # [본조신설 2006.11.06.]
    (r"\[본조신설\s*(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*\]", "신설"),
]

# Common Korean legal keywords to extract with weights
KEYWORD_PATTERNS = [
    (r"(?:이사|감사|위원|위원장|의장)", 1.0),  # 고위직 용어
    (r"(?:교원|교수|교직원|직원|강사)", 0.9),  # 인사 관련
    (r"(?:학생|학과|학부|대학|대학원)", 0.9),  # 학사 관련
    (r"(?:규정|규칙|학칙|정관|세칙)", 0.8),  # 규정 유형
    (r"(?:심의|의결|승인|결정|허가)", 0.8),  # 의결 관련
    (r"(?:임명|선임|해임|임기|보선)", 0.7),  # 임명 관련
    (r"(?:장학금|등록금|수업료|입학금)", 0.7),  # 재정 관련
    (r"(?:전형|입학|졸업|휴학|복학|제적)", 0.8),  # 학적 관련
    (r"(?:시행일|시행|폐지|개정|신설)", 0.6),  # 법률 용어
    (r"(?:예산|결산|회계|자산|재산)", 0.7),  # 재정 관련
    (r"(?:연구|교육|강의|수업|학점)", 0.8),  # 교육 관련
]

# Chunk level mapping from node type
CHUNK_LEVEL_MAP = {
    "chapter": "chapter",
    "section": "section",
    "subsection": "subsection",
    "article": "article",
    "paragraph": "paragraph",
    "item": "item",
    "subitem": "subitem",
    "addendum": "addendum",
    "addendum_item": "addendum_item",
    "preamble": "preamble",
    "text": "text",
}

# Regex patterns for detecting chunk types (REQ-001, REQ-002, REQ-003)
CHUNK_PATTERNS = {
    "chapter": r"^제\s*(\d+)\s*장\s+(.+)$",
    "section": r"^제\s*(\d+)\s*절\s+(.+)$",
    "subsection": r"^제\s*(\d+)\s*관\s+(.+)$",
    "article": r"^제\s*(\d+)\s*조\s*(?:\(([^)]+)\))?\s*(.*)$",
    "paragraph": r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(.*)$",
    "item": r"^(\d+)\.\s*(.+)$",
    "subitem": r"^([가-힣])\.\s*(.+)$",
}

# Average characters per token (Korean text approximation)
CHARS_PER_TOKEN = 2.5

# Chunk splitting patterns for HWPX Direct Parser
PARAGRAPH_PATTERN = r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])([^①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]*)"

# Unicode circled numbers mapping
CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


# ============================================================================
# HWPX Direct Parser Chunk Splitting Functions
# ============================================================================


def detect_chunk_type(text: str) -> Optional[Dict[str, Any]]:
    """
    Detect chunk type from text using pattern matching.

    Supports Korean legal document patterns:
    - Chapter (장): 제1장 총칙
    - Section (절): 제1절 목적
    - Subsection (관): 제1관 총칙

    Args:
        text: The text to analyze for chunk type detection.

    Returns:
        Dictionary with 'type', 'display_no', and 'title' keys, or None if no match.
    """
    if not text:
        return None

    text = text.strip()

    # Check patterns in priority order: chapter > section > subsection
    for chunk_type, pattern in [
        ("chapter", CHUNK_PATTERNS["chapter"]),
        ("section", CHUNK_PATTERNS["section"]),
        ("subsection", CHUNK_PATTERNS["subsection"]),
    ]:
        match = re.match(pattern, text)
        if match:
            number = match.group(1)
            title = match.group(2).strip() if len(match.groups()) > 1 else ""
            suffix = {"chapter": "장", "section": "절", "subsection": "관"}[chunk_type]
            return {
                "type": chunk_type,
                "display_no": f"제{number}{suffix}",
                "title": title,
            }

    return None


def split_text_into_chunks(text: str) -> List[Dict[str, Any]]:
    """
    Split article text into hierarchical chunks (paragraphs, items, subitems).

    Args:
        text: The full text of an article.

    Returns:
        List of chunk dictionaries with type, display_no, and text.
    """
    if not text:
        return []

    chunks = []

    # Find all paragraph markers and their positions
    paragraph_positions = []
    for match in re.finditer(PARAGRAPH_PATTERN, text):
        marker = match.group(1)
        content = match.group(2).strip()
        start_pos = match.start()
        paragraph_positions.append((marker, content, start_pos))

    if paragraph_positions:
        # Extract preamble (text before first paragraph)
        first_para_pos = paragraph_positions[0][2]
        if first_para_pos > 0:
            preamble = text[:first_para_pos].strip()
            if preamble:
                chunks.append(
                    {
                        "type": "text",
                        "display_no": "",
                        "text": preamble,
                    }
                )

        # Process each paragraph
        for marker, content, _ in paragraph_positions:
            # Split content into items if present
            items = extract_items_from_text(content)

            if items:
                # Paragraph has items - add paragraph without full content
                chunks.append(
                    {
                        "type": "paragraph",
                        "display_no": marker,
                        "text": "",  # Will be filled with non-item content
                        "children": items,
                    }
                )
            else:
                # Standalone paragraph
                chunks.append(
                    {
                        "type": "paragraph",
                        "display_no": marker,
                        "text": content,
                    }
                )
    else:
        # No paragraph markers - try to extract items directly
        items = extract_items_from_text(text)
        if items:
            chunks.extend(items)
        else:
            # No structure found - return as single text chunk
            chunks.append(
                {
                    "type": "text",
                    "display_no": "",
                    "text": text,
                }
            )

    return chunks


def extract_items_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract items (1., 2., etc.) and subitems (가., 나., etc.) from text.

    This function processes text line by line and groups subitems under their
    parent items based on line sequence.

    Args:
        text: The text to extract items from.

    Returns:
        List of item dictionaries with potential subitem children.
    """
    items = []
    current_item = None
    current_subitems = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with number followed by dot (e.g., "1.", "2.", etc.)
        if re.match(r"^\d+\.", line):
            # Save previous item with its subitems
            if current_item:
                if current_subitems:
                    current_item["children"] = current_subitems
                items.append(current_item)

            # Start new item
            parts = line.split(".", 1)
            item_no = parts[0].strip()
            item_content = parts[1].strip() if len(parts) > 1 else ""

            current_item = {
                "type": "item",
                "display_no": f"{item_no}.",
                "text": item_content,
            }
            current_subitems = []

        # Check if line starts with Korean letter followed by dot (e.g., "가.", "나.", etc.)
        elif re.match(r"^[가-힣]\.", line) and current_item:
            parts = line.split(".", 1)
            if len(parts) == 2:
                marker = parts[0].strip()
                content = parts[1].strip()
                current_subitems.append(
                    {
                        "type": "subitem",
                        "display_no": f"{marker}.",
                        "text": content,
                    }
                )

    # Don't forget the last item
    if current_item:
        if current_subitems:
            current_item["children"] = current_subitems
        items.append(current_item)

    return items


def extract_subitems_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract subitems (가., 나., etc.) from text.

    Args:
        text: The text to extract subitems from.

    Returns:
        List of subitem dictionaries.
    """
    subitems = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with Korean letter followed by dot (e.g., "가.", "나.", etc.)
        if re.match(r"^[가-힣]\.", line):
            parts = line.split(".", 1)
            if len(parts) == 2:
                marker = parts[0].strip()
                content = parts[1].strip()
                subitems.append(
                    {
                        "type": "subitem",
                        "display_no": f"{marker}.",
                        "text": content,
                    }
                )

    return subitems


def convert_article_to_children_structure(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a flat article to a children-structured article.

    Args:
        article: The article dictionary with 'text' field.

    Returns:
        Article with children structure for RAG optimization.
    """
    text = article.get("text", "")
    display_no = article.get("display_no", "")
    title = article.get("title", "")
    article_type = article.get("type", "article")

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    if not chunks:
        return article

    # If only one chunk with no structure, keep original
    if len(chunks) == 1 and chunks[0].get("type") == "text":
        return article

    # Create new article structure
    new_article = {
        "type": article_type,
        "display_no": display_no,
        "title": title,
        "text": "",  # Main article text is now distributed to children
        "children": chunks,
    }

    # Copy over other fields
    for key in article:
        if key not in ("type", "display_no", "title", "text"):
            new_article[key] = article[key]

    return new_article


def enhance_node_for_hwpx(
    node: Dict[str, Any],
    parent_path: List[str],
    doc_title: str,
) -> None:
    """
    Recursively enhance a HWPX node with RAG optimization fields.

    This function is similar to enhance_node but handles the chunked structure.

    Args:
        node: The node to enhance.
        parent_path: List of ancestor path labels.
        doc_title: The document title for the root path.
    """
    # Build current path with doc title at root
    current_path = [doc_title] + parent_path if doc_title else parent_path.copy()
    current_path = _dedupe_path_segments(current_path)

    # Add current node's label to path for children
    node_label = build_path_label(node)
    text = node.get("text", "")

    # 1. parent_path
    node["parent_path"] = current_path.copy()

    # 2. full_text (for display, includes path)
    embedding_text = ""
    if text:
        node["full_text"] = build_full_text(current_path, node)

    # 3. embedding_text (for vector embedding, with path context)
    if text:
        embedding_text = build_embedding_text(current_path, node)
        if embedding_text:
            node["embedding_text"] = embedding_text

    # 4. chunk_level
    node["chunk_level"] = determine_chunk_level(node)

    # 5. is_searchable
    node["is_searchable"] = is_node_searchable(node)

    # 6. token_count (based on embedding_text)
    if embedding_text:
        node["token_count"] = calculate_token_count(embedding_text)

    # 7. keywords (combine title and text)
    combined_text = f"{node.get('title', '')} {text}"
    keywords = extract_keywords(combined_text)
    if keywords:
        node["keywords"] = keywords

    # 8. amendment_history (from text only, not title)
    if text:
        history = extract_amendment_history(text)
        if history:
            node["amendment_history"] = history

    # 9. effective_date (for addendum nodes)
    node_type = node.get("type", "")
    if node_type in ("addendum", "addendum_item") and text:
        effective_date = extract_effective_date(text)
        if effective_date:
            node["effective_date"] = effective_date

    # Recursively process children
    children = node.get("children", [])
    child_parent_path = parent_path + [node_label] if node_label else parent_path
    child_parent_path = _dedupe_path_segments(child_parent_path)

    for child in children:
        enhance_node_for_hwpx(child, child_parent_path, doc_title)


def enhance_document_for_hwpx(doc: Dict[str, Any]) -> None:
    """
    Enhance a HWPX Direct Parser document with chunk splitting.

    This function:
    1. Converts flat articles to children structure
    2. Applies RAG enhancement to all nodes

    Args:
        doc: The document to enhance.
    """
    doc_title = doc.get("title", "")
    doc_type = doc.get("doc_type", "")

    # 1. status for document level
    status, abolished_date = determine_status(doc_title)
    doc["status"] = status
    if abolished_date:
        doc["abolished_date"] = abolished_date

    # 2. is_index_duplicate flag
    if doc_type in ("toc", "index_alpha", "index_dept", "index"):
        doc["is_index_duplicate"] = True

    # 3. Process content nodes - convert to children structure
    content = doc.get("content", [])
    new_content = []

    for node in content:
        # Convert flat article to children structure
        converted = convert_article_to_children_structure(node)
        new_content.append(converted)

    doc["content"] = new_content

    # 4. Enhance all content nodes with RAG fields
    for node in doc["content"]:
        enhance_node_for_hwpx(node, [], doc_title)

    # 5. Enhance addenda nodes
    addenda = doc.get("addenda", [])
    for node in addenda:
        enhance_node_for_hwpx(node, ["부칙"], doc_title)

    # 6. Extract last revision date from addenda
    last_revision = extract_last_revision_date(doc)
    if last_revision:
        metadata = doc.setdefault("metadata", {})
        metadata["last_revision_date"] = last_revision


# ============================================================================
# Core Enhancement Functions
# ============================================================================


def _normalize_path_segment(segment: str) -> str:
    return (segment or "").replace(" ", "").replace("　", "")


def _dedupe_path_segments(segments: List[str]) -> List[str]:
    if not segments:
        return segments
    deduped = [segments[0]]
    prev_norm = _normalize_path_segment(segments[0])
    for segment in segments[1:]:
        current_norm = _normalize_path_segment(segment)
        if current_norm != prev_norm:
            deduped.append(segment)
            prev_norm = current_norm
    return deduped


def extract_amendment_history(text: str) -> List[Dict[str, str]]:
    """
    Extract amendment history from text content.

    Args:
        text: The text to extract amendment history from.

    Returns:
        List of dicts with 'date' (YYYY-MM-DD) and 'type' keys.
    """
    history = []
    seen = set()

    for pattern, amend_type in AMENDMENT_PATTERNS:
        for match in re.finditer(pattern, text):
            year, month, day = match.groups()
            date_str = f"{year}-{int(month):02d}-{int(day):02d}"
            key = (date_str, amend_type)
            if key not in seen:
                seen.add(key)
                history.append({"date": date_str, "type": amend_type})

    # Sort by date
    history.sort(key=lambda x: x["date"])
    return history


def extract_keywords(text: str) -> List[Dict[str, Any]]:
    """
    Extract keywords from regulation text using pattern matching with weights.

    Args:
        text: The text to extract keywords from.

    Returns:
        List of dicts with 'term' and 'weight' keys, sorted by weight descending.
    """
    if not text:
        return []

    keywords = {}
    for pattern, weight in KEYWORD_PATTERNS:
        for match in re.finditer(pattern, text):
            term = match.group()
            # Keep highest weight if duplicate
            if term not in keywords or keywords[term] < weight:
                keywords[term] = weight

    # Sort by weight descending, then by term
    result = [{"term": k, "weight": v} for k, v in keywords.items()]
    result.sort(key=lambda x: (-x["weight"], x["term"]))
    return result


def extract_keywords_simple(text: str) -> List[str]:
    """
    Extract keywords as simple list (for backward compatibility in tests).

    Args:
        text: The text to extract keywords from.

    Returns:
        List of unique keywords found in the text.
    """
    keywords = extract_keywords(text)
    return [k["term"] for k in keywords]


def determine_chunk_level(node: Dict[str, Any]) -> str:
    """
    Determine the chunk level of a node based on its type.

    Args:
        node: The node dictionary.

    Returns:
        A string representing the chunk level.
    """
    node_type = node.get("type", "text")
    return CHUNK_LEVEL_MAP.get(node_type, "text")


def calculate_token_count(text: str) -> int:
    """
    Calculate approximate token count for text.

    Args:
        text: The text to count tokens for.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def calculate_hierarchy_depth(node: Dict[str, Any]) -> int:
    """
    Calculate the depth of hierarchy for a node.

    A leaf node has depth 1. Each level of children adds 1 to the depth.
    Supports hierarchy up to level 6 (REQ-004).

    Args:
        node: The node dictionary with optional 'children' key.

    Returns:
        The maximum depth of the node hierarchy (minimum 1).

    Example:
        >>> node = {"type": "article", "children": [{"type": "paragraph", "children": []}]}
        >>> calculate_hierarchy_depth(node)
        2
    """
    children = node.get("children", [])
    if not children:
        return 1
    return 1 + max(calculate_hierarchy_depth(child) for child in children)


def is_node_searchable(node: Dict[str, Any]) -> bool:
    """
    Determine if a node should be included in search results.

    A node is searchable if it has text content or is a leaf node.

    Args:
        node: The node dictionary.

    Returns:
        True if the node should be searchable.
    """
    text = node.get("text", "")
    children = node.get("children", [])
    # Searchable if has text content or is a leaf
    return bool(text) or len(children) == 0


def extract_effective_date(text: str) -> Optional[str]:
    """
    Extract effective date (시행일) from addendum text.

    Args:
        text: The addendum text to parse.

    Returns:
        Date string in YYYY-MM-DD format, or None if not found.
    """
    if not text:
        return None

    # Pattern: 이 규정은 YYYY년 MM월 DD일부터 시행한다
    patterns = [
        r"(\d{4})\s*[년\.]\s*(\d{1,2})\s*[월\.]\s*(\d{1,2})\s*일?\s*부터\s*시행",
        r"(\d{4})\s*[년\.]\s*(\d{1,2})\s*[월\.]\s*(\d{1,2})\s*\.?\s*[부로부터]*\s*시행",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

    return None


def determine_status(title: str) -> Tuple[str, Optional[str]]:
    """
    Determine regulation status based on title.

    Args:
        title: The regulation or node title.

    Returns:
        Tuple of (status, abolished_date). abolished_date is always None for now.
    """
    if "【폐지】" in title or "폐지" in title:
        return "abolished", None
    return "active", None


def build_path_label(node: Dict[str, Any]) -> str:
    """
    Build a human-readable label for a node in the path.

    Args:
        node: The node dictionary.

    Returns:
        A string label combining display_no and title (or text excerpt if title is empty).
    """
    display_no = node.get("display_no", "")
    title = node.get("title", "")

    # If title is empty, use the first part of text (extract key phrase)
    if not title:
        text = node.get("text", "")
        if text:
            # Extract key phrase: up to verb pattern (~은/는/이란/란) or max 30 chars
            # Allow Korean text, parentheses, and common punctuation
            match = re.match(r"^(.{1,30}?)(?:은|는|이란|란|의)\s", text)
            if match:
                title = match.group(1).strip()
            else:
                # Fallback: extract up to first major punctuation
                match = re.match(r"^([^.。\n]{1,25})", text)
                if match:
                    title = match.group(1).strip()

    if display_no and title:
        return f"{display_no} {title}"
    elif display_no:
        return display_no
    elif title:
        return title
    return ""


def build_full_text(parent_path: List[str], node: Dict[str, Any]) -> str:
    """
    Build self-contained full text for vector embedding.

    Args:
        parent_path: List of path segments.
        node: The current node.

    Returns:
        A string with path prefix and node text.
    """
    text = node.get("text", "")
    if not text:
        return ""

    if parent_path:
        path_segments = _dedupe_path_segments(parent_path)
        path_str = " > ".join(path_segments)
        label = build_path_label(node)
        if label:
            if path_segments and _normalize_path_segment(
                label
            ) == _normalize_path_segment(path_segments[-1]):
                return f"[{path_str}] {text}"
            return f"[{path_str} > {label}] {text}"
        return f"[{path_str}] {text}"
    return text


def build_embedding_text(parent_path: List[str], node: Dict[str, Any]) -> str:
    """
    Build context-aware embedding text for vector search.

    Unlike full_text (for display), embedding_text includes path context
    to help the embedding model understand the hierarchical position.

    Args:
        parent_path: List of path segments (최근 3단계만 사용).
        node: The current node.

    Returns:
        A string with path context prefix and node text.

    Example:
        Input: parent_path=["동의대학교학칙", "제3장 학사", "제1절 수업"]
               node={"display_no": "제15조", "title": "수업일수", "text": "..."}
        Output: "제3장 학사 > 제1절 수업 > 제15조 수업일수: ..."
    """
    text = node.get("text", "")
    if not text:
        return ""

    # Include the first segment (Regulation Title) and at most the last 3 segments for context
    if parent_path:
        # parent_path[0] is the document title (Regulation Name)
        doc_title = parent_path[0]

        # Get remaining segments from the end, excluding the first one if already included
        remaining = parent_path[1:]
        recent_path = remaining[-3:] if len(remaining) > 3 else remaining

        # Combine: [Doc Title] + [Last 3 Segments]
        context_segments = [doc_title] + recent_path
        path_segments = _dedupe_path_segments(context_segments)

        path_str = " > ".join(path_segments)
        label = build_path_label(node)
        if label:
            if path_segments and _normalize_path_segment(
                label
            ) == _normalize_path_segment(path_segments[-1]):
                return f"{path_str}: {text}"
            return f"{path_str} > {label}: {text}"
        return f"{path_str}: {text}"

    # If no path, include label if available
    label = build_path_label(node)
    if label:
        return f"{label}: {text}"
    return text


def enhance_node(
    node: Dict[str, Any],
    parent_path: List[str],
    doc_title: str,
) -> None:
    """
    Recursively enhance a node with RAG optimization fields.

    This function modifies the node in-place.

    Args:
        node: The node to enhance.
        parent_path: List of ancestor path labels.
        doc_title: The document title for the root path.
    """
    # Build current path with doc title at root
    current_path = [doc_title] + parent_path if doc_title else parent_path.copy()
    current_path = _dedupe_path_segments(current_path)

    # Add current node's label to path for children
    node_label = build_path_label(node)
    text = node.get("text", "")

    # 1. parent_path
    node["parent_path"] = current_path.copy()

    # 2. full_text (for display, includes path)
    embedding_text = ""
    if text:
        node["full_text"] = build_full_text(current_path, node)

    # 3. embedding_text (for vector embedding, with path context)
    if text:
        embedding_text = build_embedding_text(current_path, node)
        if embedding_text:
            node["embedding_text"] = embedding_text

    # 4. chunk_level
    node["chunk_level"] = determine_chunk_level(node)

    # 5. is_searchable
    node["is_searchable"] = is_node_searchable(node)

    # 6. token_count (based on embedding_text)
    if embedding_text:
        node["token_count"] = calculate_token_count(embedding_text)

    # 7. keywords (combine title and text)
    combined_text = f"{node.get('title', '')} {text}"
    keywords = extract_keywords(combined_text)
    if keywords:
        node["keywords"] = keywords

    # 8. amendment_history (from text only, not title)
    if text:
        history = extract_amendment_history(text)
        if history:
            node["amendment_history"] = history

    # 9. effective_date (for addendum nodes)
    node_type = node.get("type", "")
    if node_type in ("addendum", "addendum_item") and text:
        effective_date = extract_effective_date(text)
        if effective_date:
            node["effective_date"] = effective_date

    # Recursively process children
    children = node.get("children", [])
    child_parent_path = parent_path + [node_label] if node_label else parent_path
    child_parent_path = _dedupe_path_segments(child_parent_path)

    for child in children:
        enhance_node(child, child_parent_path, doc_title)


def extract_last_revision_date(doc: Dict[str, Any]) -> Optional[str]:
    """
    규정 문서의 부칙에서 가장 최근 시행일(effective_date)을 추출합니다.

    부칙이 여러 개인 경우, 가장 마지막(최신) 날짜를 반환합니다.

    Args:
        doc: 규정 문서 딕셔너리

    Returns:
        YYYY-MM-DD 형식의 날짜 문자열, 또는 None
    """
    dates: List[str] = []

    def collect_dates(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            if "effective_date" in node:
                dates.append(node["effective_date"])
            collect_dates(node.get("children", []))

    collect_dates(doc.get("addenda", []))

    if dates:
        return max(dates)  # YYYY-MM-DD 형식이므로 문자열 비교로 최신 날짜 추출 가능
    return None


def enhance_document(doc: Dict[str, Any]) -> None:
    """
    Enhance a single document with RAG optimization fields.

    This function modifies the document in-place.

    Args:
        doc: The document to enhance.
    """
    doc_title = doc.get("title", "")
    doc_type = doc.get("doc_type", "")

    # 1. status for document level
    status, abolished_date = determine_status(doc_title)
    doc["status"] = status
    if abolished_date:
        doc["abolished_date"] = abolished_date

    # 2. is_index_duplicate flag
    if doc_type in ("toc", "index_alpha", "index_dept", "index"):
        doc["is_index_duplicate"] = True

    # 3. Enhance content nodes
    content = doc.get("content", [])
    for node in content:
        enhance_node(node, [], doc_title)

    # 4. Enhance addenda nodes
    addenda = doc.get("addenda", [])
    for node in addenda:
        enhance_node(node, ["부칙"], doc_title)

    # 5. Extract last revision date from addenda
    last_revision = extract_last_revision_date(doc)
    if last_revision:
        metadata = doc.setdefault("metadata", {})
        metadata["last_revision_date"] = last_revision


def enhance_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance the entire JSON structure with RAG optimization fields.

    Automatically detects HWPX Direct Parser output and applies chunk splitting.

    Args:
        data: The parsed JSON data.

    Returns:
        The enhanced JSON data (modified in-place and returned).
    """
    # Detect HWPX Direct Parser output
    is_hwpx_direct = data.get("parsing_method") == "hwpx_direct"

    # Process each document
    docs = data.get("docs", [])
    for doc in docs:
        if is_hwpx_direct:
            enhance_document_for_hwpx(doc)
        else:
            enhance_document(doc)

    # Add enhancement metadata
    data["rag_enhanced"] = True
    data["rag_schema_version"] = "2.1"  # Bumped for chunk splitting support
    if is_hwpx_direct:
        data["rag_chunk_splitting"] = True

    return data


# ============================================================================
# CLI and Output Functions
# ============================================================================


def print_sample(data: Dict[str, Any], count: int = 3) -> None:
    """
    Print sample enhanced nodes for verification.

    Args:
        data: The enhanced JSON data.
        count: Number of samples to print.
    """
    samples_found = 0
    docs = data.get("docs", [])

    for doc in docs:
        if samples_found >= count:
            break

        doc_title = doc.get("title", "Unknown")
        doc_type = doc.get("doc_type", "unknown")

        # Skip index documents for sampling
        if doc_type in ("toc", "index_alpha", "index_dept", "index"):
            continue

        # Find a node with text content
        def find_node_with_text(
            nodes: List[Dict[str, Any]], depth: int = 0
        ) -> Optional[Dict[str, Any]]:
            for node in nodes:
                if node.get("text") and node.get("full_text"):
                    return node
                children = node.get("children", [])
                if children:
                    result = find_node_with_text(children, depth + 1)
                    if result:
                        return result
            return None

        content = doc.get("content", [])
        sample_node = find_node_with_text(content)

        if sample_node:
            samples_found += 1
            print(f"\n{'=' * 60}")
            print(f"[Sample {samples_found}] Document: {doc_title}")
            print(f"{'=' * 60}")
            print(f"  type: {sample_node.get('type')}")
            print(f"  display_no: {sample_node.get('display_no')}")
            print(f"  title: {sample_node.get('title')}")
            print(f"  text: {sample_node.get('text', '')[:100]}...")
            print(f"  parent_path: {sample_node.get('parent_path')}")
            print(f"  full_text: {sample_node.get('full_text', '')[:150]}...")
            print(f"  keywords: {sample_node.get('keywords', [])}")
            print(f"  amendment_history: {sample_node.get('amendment_history', [])}")

    if samples_found == 0:
        print("[Warning] No suitable sample nodes found.")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance regulation JSON for Hybrid RAG optimization."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: input_rag.json)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Print N sample enhanced nodes instead of writing output",
    )

    args = parser.parse_args()

    # Read input
    print(f"[INFO] Reading {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Enhance
    print("[INFO] Enhancing JSON for RAG...")
    enhanced_data = enhance_json(data)

    # Sample mode
    if args.sample > 0:
        print_sample(enhanced_data, args.sample)
        return

    # Determine output path
    output_path = args.output
    if output_path is None:
        stem = args.input_file.stem
        output_path = args.input_file.parent / f"{stem}_rag.json"

    # Write output
    print(f"[INFO] Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

    # Summary
    doc_count = len(enhanced_data.get("docs", []))
    abolished_count = sum(
        1 for d in enhanced_data.get("docs", []) if d.get("status") == "abolished"
    )
    print(f"[INFO] Done! Enhanced {doc_count} documents ({abolished_count} abolished).")
    print(f"[INFO] Output: {output_path}")


if __name__ == "__main__":
    main()
