"""
Domain Metrics for HWPX Regulation Parsing Coverage Tracking.

This module defines the CoverageReport dataclass for tracking coverage metrics
during HWPX regulation parsing (SPEC-HWXP-002).

TDD Approach: GREEN Phase
- Implementation created to make failing tests pass
- Minimal implementation focused on test requirements

TDD Approach: REFACTOR Phase
- Improved documentation and type hints
- Better dataclass organization
"""
from dataclasses import dataclass
from typing import Dict, Any

from src.parsing.format.format_type import FormatType


@dataclass
class CoverageReport:
    """
    Coverage metrics report for HWPX regulation parsing.

    Tracks overall coverage statistics including total regulations,
    regulations with content, coverage percentage, format breakdown,
    average content length, and low-coverage regulation count.

    Example:
        >>> report = CoverageReport(
        ...     total_regulations=100,
        ...     regulations_with_content=85,
        ...     coverage_percentage=85.0,
        ...     format_breakdown={FormatType.ARTICLE: 50, FormatType.LIST: 35},
        ...     avg_content_length=800.0,
        ...     low_coverage_count=5
        ... )
        >>> print(f"Coverage: {report.coverage_percentage:.1f}%")
        Coverage: 85.0%
        >>> report_dict = report.to_dict()
        >>> json.dumps(report_dict)  # Can be serialized to JSON

    Attributes:
        total_regulations: Total number of regulations tracked
        regulations_with_content: Number of regulations with content
        coverage_percentage: Coverage rate as percentage (0-100)
        format_breakdown: Count of regulations by format type
        avg_content_length: Average content length across all regulations
        low_coverage_count: Number of regulations with <20% content coverage
    """

    total_regulations: int
    """Total number of regulations tracked."""

    regulations_with_content: int
    """Number of regulations that have content."""

    coverage_percentage: float
    """Coverage rate as percentage (0.0 to 100.0)."""

    format_breakdown: Dict[FormatType, int]
    """Count of regulations by FormatType."""

    avg_content_length: float
    """Average content length in characters across all regulations."""

    low_coverage_count: int
    """Number of regulations with less than 20% content coverage."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert coverage report to dictionary for JSON serialization.

        Returns:
            Dictionary with coverage metrics including:
            - total: Total regulations
            - with_content: Regulations with content
            - coverage_rate: Coverage percentage
            - by_format: Format breakdown with string keys (not enum)
            - avg_content_length: Average content length
            - low_coverage_count: Low coverage regulation count

        Example:
            >>> report = CoverageReport(
            ...     total_regulations=100,
            ...     regulations_with_content=85,
            ...     coverage_percentage=85.0,
            ...     format_breakdown={FormatType.ARTICLE: 50, FormatType.LIST: 35},
            ...     avg_content_length=800.0,
            ...     low_coverage_count=5
            ... )
            >>> report_dict = report.to_dict()
            >>> report_dict["by_format"]["article"]  # String key, not enum
            50
        """
        return {
            "total": self.total_regulations,
            "with_content": self.regulations_with_content,
            "coverage_rate": self.coverage_percentage,
            "by_format": {
                format_type.value: count
                for format_type, count in self.format_breakdown.items()
            },
            "avg_content_length": self.avg_content_length,
            "low_coverage_count": self.low_coverage_count,
        }
