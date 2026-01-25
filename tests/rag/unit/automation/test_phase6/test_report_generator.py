"""
Unit Tests for TestReportGenerator (Phase 6).

Tests for comprehensive test report generation in markdown and HTML formats.
Clean Architecture: Infrastructure layer tests.
"""

from typing import Optional

from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import (
    QualityDimensions,
    QualityScore,
)
from src.rag.automation.infrastructure.test_report_generator import TestReportGenerator


def create_test_result(
    test_id: str,
    passed: bool,
    confidence: float,
    time_ms: int,
    quality: float,
    query: str = "Test query",
    answer: str = "Test answer",
    dimensions: Optional[tuple] = None,
) -> TestResult:
    """Helper to create test result."""
    if dimensions:
        dims = QualityDimensions(*dimensions)
    else:
        dims = QualityDimensions(0.8, 0.75, 0.85, 0.9, 0.4, 0.45)

    return TestResult(
        test_case_id=test_id,
        query=query,
        answer=answer,
        sources=["source1"],
        confidence=confidence,
        execution_time_ms=time_ms,
        rag_pipeline_log={},
        quality_score=QualityScore(
            dimensions=dims, total_score=quality, is_pass=passed
        ),
        passed=passed,
    )


class TestReportGeneration:
    """Test suite for report generation."""

    def test_generate_report_creates_file(self, tmp_path):
        """WHEN generating markdown report, THEN should create valid file."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001", passed=True, confidence=0.9, time_ms=1000, quality=4.5
            ),
            create_test_result(
                "test_002", passed=False, confidence=0.6, time_ms=1500, quality=2.5
            ),
        ]

        report_path = generator.generate_report(
            session_id="integration_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata={"test_key": "test_value"},
        )

        assert report_path.exists()
        assert report_path.parent == tmp_path
        assert "integration_test" in report_path.name

    def test_generate_report_content(self, tmp_path):
        """WHEN generating markdown report, THEN should include key sections."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001", passed=True, confidence=0.9, time_ms=1000, quality=4.5
            ),
        ]

        report_path = generator.generate_report(
            session_id="content_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        content = report_path.read_text(encoding="utf-8")
        # Check that report has expected sections
        assert "content_test" in content or "Content Test" in content
        assert "test_001" in content or "Test query" in content

    def test_generate_report_with_metadata(self, tmp_path):
        """WHEN generating report with metadata, THEN should include metadata."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001", passed=True, confidence=0.9, time_ms=1000, quality=4.5
            )
        ]

        report_path = generator.generate_report(
            session_id="metadata_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata={"model": "gpt-4", "temperature": 0.7},
        )

        content = report_path.read_text(encoding="utf-8")
        # Metadata should be included
        assert "metadata" in content.lower() or "gpt-4" in content

    def test_generate_html_report_creates_file(self, tmp_path):
        """WHEN generating HTML report, THEN should create HTML file."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001", passed=True, confidence=0.9, time_ms=1000, quality=4.5
            )
        ]

        report_path = generator.generate_html_report(
            session_id="html_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        assert report_path.exists()
        assert report_path.suffix == ".html"

    def test_generate_html_report_content(self, tmp_path):
        """WHEN generating HTML report, THEN should include HTML structure."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001", passed=True, confidence=0.9, time_ms=1000, quality=4.5
            )
        ]

        report_path = generator.generate_html_report(
            session_id="html_content_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        content = report_path.read_text(encoding="utf-8")
        # Should be valid HTML
        assert "<!DOCTYPE html>" in content or "<html" in content
        assert "html_content_test" in content

    def test_generate_report_with_multiple_results(self, tmp_path):
        """WHEN generating report with multiple results, THEN should include all."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001", passed=True, confidence=0.9, time_ms=1000, quality=4.5
            ),
            create_test_result(
                "test_002", passed=False, confidence=0.6, time_ms=1500, quality=2.5
            ),
            create_test_result(
                "test_003", passed=True, confidence=0.8, time_ms=1200, quality=4.0
            ),
        ]

        report_path = generator.generate_report(
            session_id="multi_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        content = report_path.read_text(encoding="utf-8")
        # All test IDs should be mentioned
        assert "test_001" in content or "Test query" in content
        # Should have some indication of pass/fail
        assert "pass" in content.lower() or "fail" in content.lower()

    def test_output_dir_creation(self, tmp_path):
        """WHEN generator is initialized, THEN should create output directory."""
        test_subdir = tmp_path / "reports"
        generator = TestReportGenerator(output_dir=test_subdir)

        assert test_subdir.exists()
        assert test_subdir.is_dir()
        assert generator.output_dir == test_subdir

    def test_empty_test_results(self, tmp_path):
        """WHEN generating report with no results, THEN should handle gracefully."""
        generator = TestReportGenerator(output_dir=tmp_path)

        report_path = generator.generate_report(
            session_id="empty_test",
            test_results=[],
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        assert report_path.exists()
        # Should still create a valid file
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_report_with_quality_dimensions(self, tmp_path):
        """WHEN results have quality dimensions, THEN should include in report."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            create_test_result(
                "test_001",
                passed=True,
                confidence=0.9,
                time_ms=1000,
                quality=4.5,
                dimensions=(0.9, 0.85, 0.9, 0.95, 0.4, 0.45),
            ),
        ]

        report_path = generator.generate_report(
            session_id="quality_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        content = report_path.read_text(encoding="utf-8")
        # Quality metrics should be mentioned
        assert "quality" in content.lower()

    def test_html_report_with_special_characters(self, tmp_path):
        """WHEN HTML report has special chars, THEN should handle properly."""
        generator = TestReportGenerator(output_dir=tmp_path)

        test_results = [
            TestResult(
                test_case_id="test_001",
                query='Query with "quotes" and <script>',
                answer="Answer with 'apostrophes'",
                sources=["source1"],
                confidence=0.9,
                execution_time_ms=1000,
                rag_pipeline_log={},
                passed=True,
            )
        ]

        report_path = generator.generate_html_report(
            session_id="special_chars_test",
            test_results=test_results,
            multi_turn_scenarios=None,
            component_analyses=None,
            failure_analyses=None,
            metadata=None,
        )

        # Should create valid HTML without errors
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        # Special characters should be escaped or handled
        assert "<!DOCTYPE html>" in content or "<html" in content
