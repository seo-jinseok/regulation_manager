"""Quality metrics analysis for HWPX parsing results.

This module provides functions for analyzing the quality of parsed HWPX documents,
including chunk statistics, hierarchy depth calculation, and quality report generation.
"""
from datetime import datetime
from typing import Any, Dict, List


def collect_chunk_statistics(
    content: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Collect chunk type distribution from document content.

    Recursively counts the number of each chunk type in the document,
    including nested children.

    Args:
        content: List of content nodes with 'type' field.

    Returns:
        Dictionary mapping chunk type names to counts.
        Includes a 'total' key with the sum of all chunks.
    """
    stats: Dict[str, int] = {}

    def count_chunks(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            chunk_type = node.get("type", "unknown")
            stats[chunk_type] = stats.get(chunk_type, 0) + 1

            # Recursively count children
            children = node.get("children", [])
            if children:
                count_chunks(children)

    count_chunks(content)

    # Add total count
    stats["total"] = sum(v for k, v in stats.items() if k != "total")

    return stats


def calculate_hierarchy_depth(node: Dict[str, Any]) -> int:
    """Calculate the maximum depth of hierarchy for a node.

    A leaf node has depth 1. Each level of children adds 1 to the depth.
    Supports hierarchy up to level 6.

    Args:
        node: The node dictionary with optional 'children' key.

    Returns:
        The maximum depth of the node hierarchy (minimum 1 for non-empty).
        Returns 0 if node has no type (empty/placeholder).
    """
    # Check if this is an empty/placeholder node
    if not node.get("type"):
        return 0

    children = node.get("children", [])
    if not children:
        return 1

    return 1 + max(calculate_hierarchy_depth(child) for child in children)


def generate_quality_report(
    doc: Dict[str, Any],
    processing_time: float,
) -> Dict[str, Any]:
    """Generate a comprehensive quality report for a parsed document.

    Creates a JSON-serializable report containing chunk statistics,
    hierarchy depth, processing time, and timestamp.

    Args:
        doc: The parsed document dictionary with 'title' and 'content'.
        processing_time: Time taken to process the document in seconds.

    Returns:
        Dictionary containing quality metrics suitable for JSON serialization.
    """
    content = doc.get("content", [])

    # Calculate max hierarchy depth across all content nodes
    max_depth = 0
    if content:
        max_depth = max(calculate_hierarchy_depth(node) for node in content)

    # Collect chunk statistics
    chunk_stats = collect_chunk_statistics(content)

    # Build report
    report = {
        "document_title": doc.get("title", "Unknown"),
        "chunk_statistics": chunk_stats,
        "max_hierarchy_depth": max_depth,
        "processing_time_seconds": processing_time,
        "generated_at": datetime.now().isoformat(),
    }

    return report
