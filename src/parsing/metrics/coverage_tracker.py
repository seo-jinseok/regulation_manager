"""
Coverage Tracker for HWPX Regulation Parsing.

This module implements CoverageTracker for real-time metrics tracking
during HWPX regulation parsing (SPEC-HWXP-002).

TDD Approach: GREEN Phase
- Implementation created to make failing tests pass
- Minimal implementation focused on test requirements

TDD Approach: REFACTOR Phase
- Improved type hints and documentation
- Better code organization and readability
- Optimized calculations
"""
from typing import Dict, Any, List


from src.parsing.format.format_type import FormatType
from src.parsing.domain.metrics import CoverageReport


# Low coverage threshold (20% of expected content)
_LOW_COVERAGE_THRESHOLD = 200  # characters
# Expected content length for full coverage
_EXPECTED_CONTENT_LENGTH = 1000  # characters


class CoverageTracker:
    """
    Track real-time coverage metrics during HWPX regulation parsing.

    Tracks regulations by format type, calculates coverage percentage,
    generates coverage reports, and identifies low-coverage regulations
    for LLM fallback analysis.

    Example:
        >>> tracker = CoverageTracker()
        >>> tracker.track_regulation(FormatType.ARTICLE, True, 1000)
        >>> tracker.track_regulation(FormatType.LIST, False, 0)
        >>> report = tracker.get_coverage_report()
        >>> print(f"Coverage: {report.coverage_percentage:.1f}%")

    Attributes:
        _total_regulations: Total count of regulations tracked
        _regulations_with_content: Count of regulations with content
        _format_breakdown: Count of regulations by format type
        _content_lengths: List of content lengths for average calculation
        _low_coverage_count: Regulations with <20% content coverage
    """

    def __init__(self) -> None:
        """Initialize an empty coverage tracker."""
        self._total_regulations: int = 0
        self._regulations_with_content: int = 0
        self._format_breakdown: Dict[FormatType, int] = {}
        self._content_lengths: List[int] = []
        self._low_coverage_count: int = 0

    def track_regulation(
        self,
        format_type: FormatType,
        has_content: bool,
        content_length: int = 0,
    ) -> None:
        """
        Track a single regulation for coverage metrics.

        Args:
            format_type: The format type of the regulation
            has_content: Whether the regulation has content
            content_length: Length of content in characters (optional)

        Example:
            >>> tracker = CoverageTracker()
            >>> tracker.track_regulation(FormatType.ARTICLE, True, 1000)
        """
        # Increment total regulations
        self._total_regulations += 1

        # Track regulations with content
        if has_content:
            self._regulations_with_content += 1

        # Track format breakdown
        if format_type not in self._format_breakdown:
            self._format_breakdown[format_type] = 0
        self._format_breakdown[format_type] += 1

        # Track content length for average calculation
        self._content_lengths.append(content_length)

        # Track low coverage regulations (<20% of expected content)
        if self._is_low_coverage(has_content, content_length):
            self._low_coverage_count += 1

    def _is_low_coverage(self, has_content: bool, content_length: int) -> bool:
        """
        Determine if a regulation has low coverage.

        Args:
            has_content: Whether the regulation has content
            content_length: Length of content in characters

        Returns:
            True if regulation has low coverage (<20% of expected content)

        Note:
            Low coverage is defined as:
            - No content (has_content=False)
            - Content length < 200 characters (<20% of 1000 char expected)
        """
        if not has_content:
            return True
        if content_length > 0 and content_length < _LOW_COVERAGE_THRESHOLD:
            return True
        return False

    def get_coverage_report(self) -> CoverageReport:
        """
        Generate current coverage report.

        Returns:
            CoverageReport with current metrics including total regulations,
            regulations with content, coverage percentage, format breakdown,
            average content length, and low coverage count.

        Example:
            >>> tracker = CoverageTracker()
            >>> tracker.track_regulation(FormatType.ARTICLE, True, 1000)
            >>> report = tracker.get_coverage_report()
            >>> print(f"Coverage: {report.coverage_percentage:.1f}%")
        """
        # Calculate coverage percentage
        coverage_percentage = self._calculate_coverage_percentage()

        # Calculate average content length
        avg_content_length = self._calculate_average_content_length()

        return CoverageReport(
            total_regulations=self._total_regulations,
            regulations_with_content=self._regulations_with_content,
            coverage_percentage=coverage_percentage,
            format_breakdown=self._format_breakdown.copy(),
            avg_content_length=avg_content_length,
            low_coverage_count=self._low_coverage_count,
        )

    def _calculate_coverage_percentage(self) -> float:
        """
        Calculate coverage percentage.

        Returns:
            Coverage percentage (0.0 to 100.0)

        Note:
            Returns 0.0 if no regulations tracked to avoid division by zero.
        """
        if self._total_regulations == 0:
            return 0.0
        return (self._regulations_with_content / self._total_regulations) * 100

    def _calculate_average_content_length(self) -> float:
        """
        Calculate average content length across all regulations.

        Returns:
            Average content length in characters

        Note:
            Returns 0.0 if no regulations tracked to avoid division by zero.
            Averages across ALL regulations, including those with 0 content.
        """
        if self._total_regulations == 0:
            return 0.0
        return sum(self._content_lengths) / self._total_regulations

    def get_low_coverage_regulations(
        self, threshold: float = 0.2
    ) -> List[str]:
        """
        Get list of low-coverage regulation IDs.

        Args:
            threshold: Coverage threshold (default 0.2 = 20%)

        Returns:
            List of regulation IDs with coverage below threshold

        Note:
            This is a placeholder for future implementation.
            Currently tracks counts but not individual regulation IDs.
            Future enhancement would store regulation IDs with metrics.
        """
        # TODO: Implement regulation ID tracking
        # This would require storing regulation IDs with their content lengths
        # and filtering based on the threshold parameter
        return []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tracker state to dictionary for serialization.

        Returns:
            Dictionary containing current coverage report and metadata

        Example:
            >>> tracker = CoverageTracker()
            >>> tracker.track_regulation(FormatType.ARTICLE, True, 1000)
            >>> tracker_dict = tracker.to_dict()
            >>> json.dumps(tracker_dict)  # Can be serialized to JSON
        """
        report = self.get_coverage_report()
        return {
            "report": report.to_dict(),
            "tracker_state": {
                "total_regulations": self._total_regulations,
                "regulations_with_content": self._regulations_with_content,
                "format_counts": self._convert_format_breakdown_to_dict(),
            },
        }

    def _convert_format_breakdown_to_dict(self) -> Dict[str, int]:
        """
        Convert format breakdown to dictionary with string keys.

        Returns:
            Dictionary mapping format type strings to counts
        """
        return {
            format_type.value: count
            for format_type, count in self._format_breakdown.items()
        }
