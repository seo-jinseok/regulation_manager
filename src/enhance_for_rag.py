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
    "article": "article",
    "paragraph": "paragraph",
    "item": "item",
    "subitem": "subitem",
    "addendum": "addendum",
    "addendum_item": "addendum_item",
    "preamble": "preamble",
    "text": "text",
}

# Average characters per token (Korean text approximation)
CHARS_PER_TOKEN = 2.5


# ============================================================================
# Core Enhancement Functions
# ============================================================================


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
        A string label combining display_no and title.
    """
    display_no = node.get("display_no", "")
    title = node.get("title", "")
    
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
        path_str = " > ".join(parent_path)
        label = build_path_label(node)
        if label:
            return f"[{path_str} > {label}] {text}"
        return f"[{path_str}] {text}"
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
    
    # Add current node's label to path for children
    node_label = build_path_label(node)
    text = node.get("text", "")
    
    # 1. parent_path
    node["parent_path"] = current_path.copy()
    
    # 2. full_text (for display, includes path)
    if text:
        node["full_text"] = build_full_text(current_path, node)
    
    # 3. embedding_text (for vector embedding, pure text only)
    if text:
        node["embedding_text"] = text
    
    # 4. chunk_level
    node["chunk_level"] = determine_chunk_level(node)
    
    # 5. is_searchable
    node["is_searchable"] = is_node_searchable(node)
    
    # 6. token_count (based on embedding_text)
    if text:
        node["token_count"] = calculate_token_count(text)
    
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
    
    for child in children:
        enhance_node(child, child_parent_path, doc_title)


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


def enhance_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance the entire JSON structure with RAG optimization fields.
    
    Args:
        data: The parsed JSON data.
        
    Returns:
        The enhanced JSON data (modified in-place and returned).
    """
    # Process each document
    docs = data.get("docs", [])
    for doc in docs:
        enhance_document(doc)
    
    # Add enhancement metadata
    data["rag_enhanced"] = True
    data["rag_schema_version"] = "2.0"
    
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
        def find_node_with_text(nodes: List[Dict[str, Any]], depth: int = 0) -> Optional[Dict[str, Any]]:
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
            print(f"\n{'='*60}")
            print(f"[Sample {samples_found}] Document: {doc_title}")
            print(f"{'='*60}")
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
        "-o", "--output",
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
        1 for d in enhanced_data.get("docs", [])
        if d.get("status") == "abolished"
    )
    print(f"[INFO] Done! Enhanced {doc_count} documents ({abolished_count} abolished).")
    print(f"[INFO] Output: {output_path}")


if __name__ == "__main__":
    main()
