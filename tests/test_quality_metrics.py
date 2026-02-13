"""Tests for quality metrics analysis.

This module tests the quality analysis functionality for HWPX parsing:
- Chunk type distribution collection
- Hierarchy depth calculation
- Quality report generation
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.analysis.quality_metrics import (
    calculate_hierarchy_depth,
    collect_chunk_statistics,
    generate_quality_report,
)


# ============================================================================
# Chunk Statistics Tests (REQ-004)
# ============================================================================


def test_collect_chunk_statistics_basic() -> None:
    """REQ-004: Should collect chunk type distribution."""
    # Arrange
    content = [
        {"type": "chapter", "text": "Chapter 1"},
        {"type": "section", "text": "Section 1"},
        {"type": "article", "text": "Article 1"},
        {"type": "article", "text": "Article 2"},
        {"type": "paragraph", "text": "Paragraph 1"},
        {"type": "paragraph", "text": "Paragraph 2"},
        {"type": "paragraph", "text": "Paragraph 3"},
    ]

    # Act
    stats = collect_chunk_statistics(content)

    # Assert
    assert stats["chapter"] == 1
    assert stats["section"] == 1
    assert stats["article"] == 2
    assert stats["paragraph"] == 3


def test_collect_chunk_statistics_empty() -> None:
    """REQ-004: Should handle empty content."""
    # Arrange
    content: List[Dict[str, Any]] = []

    # Act
    stats = collect_chunk_statistics(content)

    # Assert - empty content returns only total=0
    assert stats.get("total", 0) == 0


def test_collect_chunk_statistics_nested() -> None:
    """REQ-004: Should count nested children chunks."""
    # Arrange
    content = [
        {
            "type": "article",
            "text": "Article 1",
            "children": [
                {"type": "paragraph", "text": "Para 1"},
                {
                    "type": "paragraph",
                    "text": "Para 2",
                    "children": [
                        {"type": "item", "text": "Item 1"},
                        {"type": "item", "text": "Item 2"},
                    ],
                },
            ],
        }
    ]

    # Act
    stats = collect_chunk_statistics(content)

    # Assert
    assert stats["article"] == 1
    assert stats["paragraph"] == 2
    assert stats["item"] == 2


def test_collect_chunk_statistics_total_count() -> None:
    """REQ-004: Should include total count in statistics."""
    # Arrange
    content = [
        {"type": "chapter", "text": "Chapter 1"},
        {"type": "article", "text": "Article 1"},
        {"type": "article", "text": "Article 2"},
    ]

    # Act
    stats = collect_chunk_statistics(content)

    # Assert
    assert stats.get("total", 0) == 3


# ============================================================================
# Hierarchy Depth Tests (REQ-004)
# ============================================================================


def test_calculate_hierarchy_depth_flat() -> None:
    """REQ-004: Should return 1 for flat structure (no children)."""
    # Arrange
    node = {"type": "article", "text": "Simple article"}

    # Act
    depth = calculate_hierarchy_depth(node)

    # Assert
    assert depth == 1


def test_calculate_hierarchy_depth_nested() -> None:
    """REQ-004: Should calculate depth for nested structure."""
    # Arrange
    node = {
        "type": "article",
        "text": "Article",
        "children": [
            {
                "type": "paragraph",
                "text": "Para",
                "children": [
                    {"type": "item", "text": "Item"},
                ],
            }
        ],
    }

    # Act
    depth = calculate_hierarchy_depth(node)

    # Assert
    assert depth == 3  # article -> paragraph -> item


def test_calculate_hierarchy_depth_deep() -> None:
    """REQ-004: Should support hierarchy up to level 6."""
    # Arrange - 6 levels deep
    node = {
        "type": "chapter",
        "children": [
            {
                "type": "section",
                "children": [
                    {
                        "type": "article",
                        "children": [
                            {
                                "type": "paragraph",
                                "children": [
                                    {
                                        "type": "item",
                                        "children": [
                                            {"type": "subitem"}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    # Act
    depth = calculate_hierarchy_depth(node)

    # Assert
    assert depth == 6  # chapter -> section -> article -> paragraph -> item -> subitem


def test_calculate_hierarchy_depth_empty_children() -> None:
    """REQ-004: Should handle empty children list."""
    # Arrange
    node = {"type": "article", "children": []}

    # Act
    depth = calculate_hierarchy_depth(node)

    # Assert
    assert depth == 1


def test_calculate_hierarchy_depth_multiple_branches() -> None:
    """REQ-004: Should return maximum depth across branches."""
    # Arrange
    node = {
        "type": "article",
        "children": [
            {"type": "paragraph"},  # depth 2
            {
                "type": "paragraph",
                "children": [
                    {"type": "item"}  # depth 3
                ],
            },
            {
                "type": "paragraph",
                "children": [
                    {
                        "type": "item",
                        "children": [
                            {"type": "subitem"}  # depth 4
                        ],
                    }
                ],
            },
        ],
    }

    # Act
    depth = calculate_hierarchy_depth(node)

    # Assert
    assert depth == 4


# ============================================================================
# Quality Report Tests (REQ-004)
# ============================================================================


def test_generate_quality_report_basic() -> None:
    """REQ-004: Should generate JSON quality report."""
    # Arrange
    doc = {
        "title": "Test Regulation",
        "content": [
            {"type": "chapter", "text": "Chapter 1"},
            {"type": "article", "text": "Article 1"},
        ],
    }
    processing_time = 1.5

    # Act
    report = generate_quality_report(doc, processing_time)

    # Assert
    assert "document_title" in report
    assert report["document_title"] == "Test Regulation"
    assert "chunk_statistics" in report
    assert "max_hierarchy_depth" in report
    assert "processing_time_seconds" in report
    assert report["processing_time_seconds"] == 1.5


def test_generate_quality_report_includes_chunk_stats() -> None:
    """REQ-004: Should include chunk statistics in report."""
    # Arrange
    doc = {
        "title": "Test Regulation",
        "content": [
            {"type": "chapter"},
            {"type": "article"},
            {"type": "article"},
            {"type": "paragraph"},
        ],
    }

    # Act
    report = generate_quality_report(doc, 0.5)

    # Assert
    stats = report["chunk_statistics"]
    assert stats["chapter"] == 1
    assert stats["article"] == 2
    assert stats["paragraph"] == 1


def test_generate_quality_report_includes_hierarchy_depth() -> None:
    """REQ-004: Should include max hierarchy depth in report."""
    # Arrange
    doc = {
        "title": "Test Regulation",
        "content": [
            {
                "type": "article",
                "children": [
                    {
                        "type": "paragraph",
                        "children": [
                            {"type": "item"},
                        ],
                    }
                ],
            }
        ],
    }

    # Act
    report = generate_quality_report(doc, 0.5)

    # Assert
    assert report["max_hierarchy_depth"] == 3


def test_generate_quality_report_serializable() -> None:
    """REQ-004: Should be JSON serializable."""
    # Arrange
    doc = {
        "title": "Test Regulation",
        "content": [{"type": "article", "text": "Article 1"}],
    }

    # Act
    report = generate_quality_report(doc, 1.0)

    # Assert - Should not raise exception
    json_str = json.dumps(report, ensure_ascii=False)
    assert json_str is not None


def test_generate_quality_report_includes_timestamp() -> None:
    """REQ-004: Should include generation timestamp."""
    # Arrange
    doc = {"title": "Test", "content": []}

    # Act
    report = generate_quality_report(doc, 0.5)

    # Assert
    assert "generated_at" in report
    # Should be ISO format timestamp
    assert "T" in report["generated_at"]


def test_generate_quality_report_empty_document() -> None:
    """REQ-004: Should handle empty document gracefully."""
    # Arrange
    doc = {"title": "Empty Regulation", "content": []}

    # Act
    report = generate_quality_report(doc, 0.1)

    # Assert
    assert report["document_title"] == "Empty Regulation"
    assert report["max_hierarchy_depth"] == 0
    # Empty content still has total=0 in statistics
    assert report["chunk_statistics"].get("total", 0) == 0
