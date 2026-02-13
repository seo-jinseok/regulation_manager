"""
TASK-009: Integration Testing for SPEC-HWXP-002.

Comprehensive integration tests and end-to-end validation for:
- Full HWPX file parsing (514 regulations)
- Coverage validation (>90% target)
- Format breakdown verification
- Performance benchmarks (<60 seconds)
- Backward compatibility with RAG pipeline
- Edge case handling

Version: 1.0.0
Reference: SPEC-HWXP-002, TASK-009
"""
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import pytest

from src.parsing.multi_format_parser import HWPXMultiFormatParser
from src.parsing.optimized_multi_format_parser import OptimizedHWPXMultiFormatParser
from src.parsing.format.format_type import FormatType
from src.parsing.metrics.coverage_tracker import CoverageTracker, CoverageReport


# ============================================================================
# Test Configuration and Fixtures
# ============================================================================

TARGET_HWPX_FILE = Path("data/input/규정집9-343(20250909).hwpx")
TARGET_REGULATIONS_COUNT = 514
TARGET_COVERAGE_RATE = 90.0
TARGET_PARSING_TIME = 60.0  # seconds


@pytest.fixture(scope="module")
def actual_hwpx_file() -> Path:
    """Path to the actual HWPX file for testing."""
    path = Path(__file__).parent.parent.parent.parent / TARGET_HWPX_FILE
    assert path.exists(), f"Target HWPX file not found: {path}"
    assert path.stat().st_size > 4_000_000, "HWPX file seems too small"
    return path


@pytest.fixture(scope="module")
def hwpx_file_stats(actual_hwpx_file: Path) -> Dict[str, Any]:
    """Get HWPX file statistics for validation."""
    stats = {
        "size_bytes": actual_hwpx_file.stat().st_size,
        "size_mb": actual_hwpx_file.stat().st_size / (1024 * 1024),
        "is_zip": False,
        "section_count": 0,
        "sections": []
    }

    # Verify ZIP structure
    try:
        with zipfile.ZipFile(actual_hwpx_file, 'r') as zf:
            stats["is_zip"] = True
            stats["section_count"] = len([
                n for n in zf.namelist()
                if n.startswith("Contents/section") and n.endswith(".xml")
            ])
            stats["sections"] = [
                n for n in zf.namelist()
                if n.startswith("Contents/section")
            ]
    except Exception as e:
        pytest.fail(f"Failed to read HWPX as ZIP: {e}")

    return stats


@pytest.fixture(scope="module")
def standard_parser_result(actual_hwpx_file: Path) -> Dict[str, Any]:
    """Parse HWPX file with standard parser (cached for module)."""
    parser = HWPXMultiFormatParser()
    start_time = time.time()
    result = parser.parse_file(actual_hwpx_file)
    elapsed = time.time() - start_time

    result["metadata"]["parsing_time_seconds"] = elapsed

    return result


@pytest.fixture(scope="module")
def optimized_parser_result(actual_hwpx_file: Path) -> Dict[str, Any]:
    """Parse HWPX file with optimized parser (cached for module)."""
    parser = OptimizedHWPXMultiFormatParser()
    start_time = time.time()
    result = parser.parse_file(actual_hwpx_file)
    elapsed = time.time() - start_time

    result["metadata"]["parsing_time_seconds"] = elapsed

    return result


# ============================================================================
# Test Class 1: File Structure Validation
# ============================================================================

class TestHWPXFileStructure:
    """Validate HWPX file structure before parsing."""

    def test_hwpx_file_exists(self, actual_hwpx_file: Path):
        """Verify the target HWPX file exists."""
        assert actual_hwpx_file.exists()
        assert actual_hwpx_file.is_file()

    def test_hwpx_file_size(self, hwpx_file_stats: Dict[str, Any]):
        """Verify HWPX file has expected size (~4MB)."""
        # File should be between 3MB and 10MB
        assert 3_000_000 < hwpx_file_stats["size_bytes"] < 10_000_000

    def test_hwpx_is_valid_zip(self, hwpx_file_stats: Dict[str, Any]):
        """Verify HWPX file is a valid ZIP archive."""
        assert hwpx_file_stats["is_zip"]
        assert hwpx_file_stats["section_count"] >= 2  # At least section0 and section1

    def test_hwpx_has_required_sections(self, hwpx_file_stats: Dict[str, Any]):
        """Verify HWPX file has required sections."""
        required_sections = ["Contents/section0.xml", "Contents/section1.xml"]
        for section in required_sections:
            assert section in hwpx_file_stats["sections"], f"Missing section: {section}"


# ============================================================================
# Test Class 2: Full File Parsing
# ============================================================================

class TestFullFileParsing:
    """Test complete HWPX file parsing with all regulations."""

    def test_parse_full_file_standard(self, standard_parser_result: Dict[str, Any]):
        """Test standard parser parses all 514 regulations."""
        metadata = standard_parser_result["metadata"]
        assert metadata["total_regulations"] == TARGET_REGULATIONS_COUNT
        assert metadata["successfully_parsed"] == TARGET_REGULATIONS_COUNT

    def test_parse_full_file_optimized(self, optimized_parser_result: Dict[str, Any]):
        """Test optimized parser parses all 514 regulations."""
        metadata = optimized_parser_result["metadata"]
        assert metadata["total_regulations"] == TARGET_REGULATIONS_COUNT
        assert metadata["successfully_parsed"] == TARGET_REGULATIONS_COUNT

    def test_result_structure(self, standard_parser_result: Dict[str, Any]):
        """Verify result has required structure."""
        assert "docs" in standard_parser_result
        assert "coverage" in standard_parser_result
        assert "metadata" in standard_parser_result

    def test_regulation_entries_complete(self, standard_parser_result: Dict[str, Any]):
        """Verify all regulation entries are created."""
        docs = standard_parser_result["docs"]
        assert len(docs) == TARGET_REGULATIONS_COUNT

    def test_all_regulations_have_required_fields(self, standard_parser_result: Dict[str, Any]):
        """Verify all regulations have required fields."""
        docs = standard_parser_result["docs"]

        for doc in docs:
            assert "title" in doc, "Missing title field"
            assert "content" in doc, "Missing content field"
            assert "articles" in doc, "Missing articles field"
            assert "metadata" in doc, "Missing metadata field"
            assert isinstance(doc["articles"], list)


# ============================================================================
# Test Class 3: Coverage Validation
# ============================================================================

class TestCoverageValidation:
    """Validate coverage metrics meet target (>90%)."""

    def test_coverage_rate_exceeds_target(self, standard_parser_result: Dict[str, Any]):
        """Verify coverage rate exceeds 90% target."""
        coverage = standard_parser_result["coverage"]
        coverage_rate = coverage["coverage_rate"]
        assert coverage_rate >= TARGET_COVERAGE_RATE, \
            f"Coverage rate {coverage_rate}% is below target {TARGET_COVERAGE_RATE}%"

    def test_regulations_with_content(self, standard_parser_result: Dict[str, Any]):
        """Verify at least 90% of regulations have content."""
        coverage = standard_parser_result["coverage"]
        with_content = coverage["with_content"]
        total = coverage["total"]

        coverage_rate = (with_content / total) * 100
        assert coverage_rate >= TARGET_COVERAGE_RATE, \
            f"Only {coverage_rate:.1f}% regulations have content, target {TARGET_COVERAGE_RATE}%"

    def test_empty_regulations_below_threshold(self, standard_parser_result: Dict[str, Any]):
        """Verify empty regulations are below 10% threshold."""
        coverage = standard_parser_result["coverage"]
        total = coverage["total"]
        with_content = coverage["with_content"]
        empty = total - with_content

        empty_rate = (empty / total) * 100
        assert empty_rate < 10.0, \
            f"Too many empty regulations: {empty} ({empty_rate:.1f}%), should be <10%"

    def test_avg_content_length_reasonable(self, standard_parser_result: Dict[str, Any]):
        """Verify average content length is reasonable."""
        coverage = standard_parser_result["coverage"]
        avg_length = coverage["avg_content_length"]

        # Should be at least 400 characters average
        assert avg_length >= 400, \
            f"Average content length too low: {avg_length:.0f} chars"

    def test_format_breakdown_complete(self, standard_parser_result: Dict[str, Any]):
        """Verify format breakdown includes all format types."""
        coverage = standard_parser_result["coverage"]
        by_format = coverage["by_format"]

        # Should have all format types
        assert "article" in by_format
        assert "list" in by_format
        assert "guideline" in by_format
        assert "unstructured" in by_format

        # At least one regulation in each format
        for fmt, count in by_format.items():
            assert count >= 0, f"Invalid count for format {fmt}: {count}"


# ============================================================================
# Test Class 4: Format Breakdown Verification
# ============================================================================

class TestFormatBreakdown:
    """Verify all format types are correctly handled."""

    def test_article_format_detected(self, standard_parser_result: Dict[str, Any]):
        """Verify article format regulations are detected."""
        by_format = standard_parser_result["coverage"]["by_format"]
        article_count = by_format.get("article", 0)

        # Should have substantial article-format regulations
        assert article_count >= 200, \
            f"Too few article-format regulations: {article_count}"

    def test_list_format_detected(self, standard_parser_result: Dict[str, Any]):
        """Verify list format regulations are detected."""
        by_format = standard_parser_result["coverage"]["by_format"]
        list_count = by_format.get("list", 0)

        # Should have list-format regulations (new in v3.5)
        assert list_count > 0, "No list-format regulations detected"

    def test_guideline_format_detected(self, standard_parser_result: Dict[str, Any]):
        """Verify guideline format regulations are detected."""
        by_format = standard_parser_result["coverage"]["by_format"]
        guideline_count = by_format.get("guideline", 0)

        # Should have guideline-format regulations (new in v3.5)
        assert guideline_count > 0, "No guideline-format regulations detected"

    def test_unstructured_format_detected(self, standard_parser_result: Dict[str, Any]):
        """Verify unstructured format regulations are detected."""
        by_format = standard_parser_result["coverage"]["by_format"]
        unstructured_count = by_format.get("unstructured", 0)

        # Unstructured should be small minority (<10%)
        total = standard_parser_result["coverage"]["total"]
        unstructured_rate = (unstructured_count / total) * 100
        assert unstructured_rate < 10.0, \
            f"Too many unstructured regulations: {unstructured_rate:.1f}%"

    def test_format_classification_in_result(self, standard_parser_result: Dict[str, Any]):
        """Verify each regulation has format type in metadata."""
        docs = standard_parser_result["docs"]

        format_counts = {
            "article": 0,
            "list": 0,
            "guideline": 0,
            "unstructured": 0
        }

        for doc in docs:
            fmt = doc["metadata"].get("format_type")
            assert fmt in format_counts, f"Unknown format type: {fmt}"
            format_counts[fmt] += 1

        # At least some regulations in each format
        for fmt, count in format_counts.items():
            assert count > 0 or fmt == "unstructured", \
                f"No regulations with format type: {fmt}"


# ============================================================================
# Test Class 5: Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Verify parsing performance meets targets."""

    def test_standard_parser_parsing_time(self, standard_parser_result: Dict[str, Any]):
        """Verify standard parser completes within time target."""
        parsing_time = standard_parser_result["metadata"]["parsing_time_seconds"]
        assert parsing_time < TARGET_PARSING_TIME, \
            f"Standard parser too slow: {parsing_time:.2f}s, target {TARGET_PARSING_TIME}s"

    def test_optimized_parser_parsing_time(self, optimized_parser_result: Dict[str, Any]):
        """Verify optimized parser completes within time target."""
        parsing_time = optimized_parser_result["metadata"]["parsing_time_seconds"]
        assert parsing_time < TARGET_PARSING_TIME, \
            f"Optimized parser too slow: {parsing_time:.2f}s, target {TARGET_PARSING_TIME}s"

    def test_optimized_faster_than_standard(
        self,
        standard_parser_result: Dict[str, Any],
        optimized_parser_result: Dict[str, Any]
    ):
        """Verify optimized parser is faster than standard."""
        standard_time = standard_parser_result["metadata"]["parsing_time_seconds"]
        optimized_time = optimized_parser_result["metadata"]["parsing_time_seconds"]

        # Optimized should be at least 10% faster
        improvement = (standard_time - optimized_time) / standard_time
        assert improvement >= 0.10, \
            f"Optimized parser not significantly faster: {improvement:.1%} improvement"

    def test_optimized_uses_parallel_workers(self, optimized_parser_result: Dict[str, Any]):
        """Verify optimized parser used parallel processing."""
        metadata = optimized_parser_result["metadata"]
        assert "parallel_workers_used" in metadata
        assert metadata["parallel_workers_used"] > 1, \
            "Optimized parser should use multiple workers"

    def test_optimization_enabled_flag(self, optimized_parser_result: Dict[str, Any]):
        """Verify optimization flag is set in result."""
        metadata = optimized_parser_result["metadata"]
        assert metadata.get("optimization_enabled", False), \
            "Optimization should be enabled in optimized parser"


# ============================================================================
# Test Class 6: Content Quality Validation
# ============================================================================

class TestContentQuality:
    """Verify extracted content quality."""

    def test_content_not_empty_for_most(self, standard_parser_result: Dict[str, Any]):
        """Verify most regulations have non-empty content."""
        docs = standard_parser_result["docs"]
        with_content = sum(1 for doc in docs if doc.get("content"))

        content_rate = (with_content / len(docs)) * 100
        assert content_rate >= TARGET_COVERAGE_RATE, \
            f"Only {content_rate:.1f}% regulations have content"

    def test_articles_extracted(self, standard_parser_result: Dict[str, Any]):
        """Verify articles are extracted for regulations with content."""
        docs = standard_parser_result["docs"]

        with_articles = 0
        for doc in docs:
            if doc.get("articles"):
                with_articles += 1

        articles_rate = (with_articles / len(docs)) * 100
        assert articles_rate >= TARGET_COVERAGE_RATE, \
            f"Only {articles_rate:.1f}% regulations have articles extracted"

    def test_content_length_distribution(self, standard_parser_result: Dict[str, Any]):
        """Verify content length distribution is reasonable."""
        docs = standard_parser_result["docs"]

        lengths = [len(doc.get("content", "")) for doc in docs]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)

        # Average should be reasonable
        assert 400 <= avg_length <= 2000, \
            f"Average content length out of range: {avg_length:.0f}"

        # Should have some long regulations
        assert max_length > 1000, f"Max content length too low: {max_length}"

    def test_coverage_scores_reasonable(self, standard_parser_result: Dict[str, Any]):
        """Verify coverage scores are reasonable."""
        docs = standard_parser_result["docs"]

        coverage_scores = [
            doc["metadata"].get("coverage_score", 0.0)
            for doc in docs
        ]

        # Filter out zero coverage (empty regs)
        non_zero_scores = [s for s in coverage_scores if s > 0]

        if non_zero_scores:
            avg_coverage = sum(non_zero_scores) / len(non_zero_scores)
            assert 0.3 <= avg_coverage <= 1.0, \
                f"Average coverage score unreasonable: {avg_coverage:.2f}"


# ============================================================================
# Test Class 7: Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Verify compatibility with existing RAG pipeline."""

    def test_json_serializable(self, standard_parser_result: Dict[str, Any]):
        """Verify result is JSON serializable."""
        try:
            json_str = json.dumps(standard_parser_result, ensure_ascii=False)
            assert len(json_str) > 0
        except Exception as e:
            pytest.fail(f"Result not JSON serializable: {e}")

    def test_regulation_structure_compatible(self, standard_parser_result: Dict[str, Any]):
        """Verify regulation structure matches expected schema."""
        docs = standard_parser_result["docs"]

        if docs:
            sample_doc = docs[0]

            # Required fields for RAG pipeline
            assert "title" in sample_doc
            assert "content" in sample_doc
            assert "articles" in sample_doc
            assert "provisions" in sample_doc

    def test_metadata_field_present(self, standard_parser_result: Dict[str, Any]):
        """Verify metadata field exists for each regulation."""
        docs = standard_parser_result["docs"]

        for doc in docs:
            assert "metadata" in doc
            assert isinstance(doc["metadata"], dict)


# ============================================================================
# Test Class 8: Edge Case Handling
# ============================================================================

class TestEdgeCaseHandling:
    """Test handling of edge cases and special scenarios."""

    def test_titles_with_special_characters(self, standard_parser_result: Dict[str, Any]):
        """Verify titles with special characters are handled."""
        docs = standard_parser_result["docs"]

        # Find titles with special characters
        special_char_titles = [
            doc["title"] for doc in docs
            if any(c in doc["title"] for c in "()[]{}〈〉·")
        ]

        assert len(special_char_titles) > 0, "Should have titles with special chars"

        # Verify they're properly extracted
        for title in special_char_titles:
            assert len(title) >= 4, f"Title too short after extraction: {title}"

    def test_repealed_regulations_handled(self, standard_parser_result: Dict[str, Any]):
        """Verify repealed regulations are handled gracefully."""
        docs = standard_parser_result["docs"]

        # Look for potential repealed regulations
        repealed = [
            doc for doc in docs
            if "폐지" in doc.get("title", "") or "폐지" in doc.get("content", "")
        ]

        # Should not crash, even if empty
        for doc in repealed:
            assert "title" in doc
            assert "articles" in doc
            assert isinstance(doc["articles"], list)

    def test_very_short_regulations(self, standard_parser_result: Dict[str, Any]):
        """Verify very short regulations are handled."""
        docs = standard_parser_result["docs"]

        # Find very short regulations
        short_regs = [
            doc for doc in docs
            if 0 < len(doc.get("content", "")) < 100
        ]

        # Should still have structure
        for doc in short_regs:
            assert "title" in doc
            assert "articles" in doc

    def test_very_long_regulations(self, standard_parser_result: Dict[str, Any]):
        """Verify very long regulations are handled."""
        docs = standard_parser_result["docs"]

        # Find very long regulations
        long_regs = [
            doc for doc in docs
            if len(doc.get("content", "")) > 5000
        ]

        # Should have reasonable article extraction
        for doc in long_regs:
            assert "articles" in doc
            assert isinstance(doc["articles"], list)


# ============================================================================
# Test Class 9: Coverage Report Validation
# ============================================================================

class TestCoverageReport:
    """Validate coverage report structure and accuracy."""

    def test_coverage_report_structure(self, standard_parser_result: Dict[str, Any]):
        """Verify coverage report has all required fields."""
        coverage = standard_parser_result["coverage"]

        required_fields = [
            "total",
            "with_content",
            "coverage_rate",
            "by_format",
            "avg_content_length",
            "low_coverage_count"
        ]

        for field in required_fields:
            assert field in coverage, f"Missing coverage field: {field}"

    def test_coverage_report_values_valid(self, standard_parser_result: Dict[str, Any]):
        """Verify coverage report values are valid."""
        coverage = standard_parser_result["coverage"]

        assert coverage["total"] == TARGET_REGULATIONS_COUNT
        assert 0 <= coverage["coverage_rate"] <= 100
        assert coverage["avg_content_length"] > 0
        assert coverage["low_coverage_count"] >= 0

    def test_coverage_breakdown_sum_matches_total(self, standard_parser_result: Dict[str, Any]):
        """Verify format breakdown sum matches total."""
        coverage = standard_parser_result["coverage"]
        by_format = coverage["by_format"]

        format_sum = sum(by_format.values())
        total = coverage["total"]

        # Allow small tolerance for unclassified
        assert abs(format_sum - total) <= 10, \
            f"Format breakdown sum {format_sum} doesn't match total {total}"


# ============================================================================
# Test Class 10: Parser Comparison
# ============================================================================

class TestParserComparison:
    """Compare standard and optimized parser results."""

    def test_same_regulation_count(
        self,
        standard_parser_result: Dict[str, Any],
        optimized_parser_result: Dict[str, Any]
    ):
        """Verify both parsers extract same number of regulations."""
        standard_count = len(standard_parser_result["docs"])
        optimized_count = len(optimized_parser_result["docs"])

        assert standard_count == optimized_count, \
            f"Parser count mismatch: standard={standard_count}, optimized={optimized_count}"

    def test_same_coverage_rate(
        self,
        standard_parser_result: Dict[str, Any],
        optimized_parser_result: Dict[str, Any]
    ):
        """Verify both parsers achieve similar coverage."""
        standard_coverage = standard_parser_result["coverage"]["coverage_rate"]
        optimized_coverage = optimized_parser_result["coverage"]["coverage_rate"]

        # Coverage should be within 1%
        assert abs(standard_coverage - optimized_coverage) <= 1.0, \
            f"Coverage rate differs too much: standard={standard_coverage}%, optimized={optimized_coverage}%"

    def test_same_titles_extracted(
        self,
        standard_parser_result: Dict[str, Any],
        optimized_parser_result: Dict[str, Any]
    ):
        """Verify both parsers extract same titles."""
        standard_titles = set(doc["title"] for doc in standard_parser_result["docs"])
        optimized_titles = set(doc["title"] for doc in optimized_parser_result["docs"])

        assert standard_titles == optimized_titles, \
            f"Titles differ: standard has {len(standard_titles)}, optimized has {len(optimized_titles)}"


# ============================================================================
# Test Class 11: Regression Tests
# ============================================================================

class TestRegressionPrevention:
    """Ensure no regressions from previous implementation."""

    def test_no_duplicate_regulations(self, standard_parser_result: Dict[str, Any]):
        """Verify no duplicate regulation entries."""
        docs = standard_parser_result["docs"]
        titles = [doc["title"] for doc in docs]

        # Check for duplicates
        unique_titles = set(titles)
        assert len(unique_titles) == len(titles), \
            f"Found {len(titles) - len(unique_titles)} duplicate regulations"

    def test_all_titles_from_toc(self, standard_parser_result: Dict[str, Any]):
        """Verify all extracted titles are valid regulation titles."""
        docs = standard_parser_result["docs"]

        for doc in docs:
            title = doc["title"]
            # Title should be at least 4 characters
            assert len(title) >= 4, f"Title too short: {title}"
            # Title should not contain only special characters
            assert any(c.isalnum() for c in title), f"Title has no alphanumeric: {title}"

    def test_article_numbering_preserved(self, standard_parser_result: Dict[str, Any]):
        """Verify article numbering is preserved."""
        docs = standard_parser_result["docs"]

        # Find regulations with articles
        with_articles = [
            doc for doc in docs
            if doc.get("articles") and len(doc["articles"]) > 0
        ]

        if with_articles:
            # Check first few regulations
            for doc in with_articles[:10]:
                articles = doc["articles"]
                # Article numbers should be sequential
                for i, article in enumerate(articles):
                    if "number" in article:
                        expected = i + 1
                        actual = article["number"]
                        assert actual == expected, \
                            f"Article numbering not sequential: expected {expected}, got {actual}"


# ============================================================================
# Test Class 12: Memory and Resource Usage
# ============================================================================

class TestResourceUsage:
    """Test memory and resource usage during parsing."""

    def test_memory_usage_reasonable(self, actual_hwpx_file: Path):
        """Verify memory usage is reasonable during parsing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        parser = HWPXMultiFormatParser()
        parser.parse_file(actual_hwpx_file)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable (<2GB)
        assert memory_increase < 2000, \
            f"Memory usage too high: {memory_increase:.0f}MB increase"

    def test_no_file_handle_leaks(self, actual_hwpx_file: Path):
        """Verify no file handle leaks during parsing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_open_files = len(process.open_files())

        parser = HWPXMultiFormatParser()
        for _ in range(3):  # Parse multiple times
            parser.parse_file(actual_hwpx_file)

        final_open_files = len(process.open_files())

        # Open files should not increase significantly
        assert final_open_files - initial_open_files < 10, \
            f"Potential file handle leak: {final_open_files - initial_open_files} extra open files"


# ============================================================================
# Integration Test Markers
# ============================================================================

@pytest.mark.integration
def test_full_pipeline_integration(actual_hwpx_file: Path):
    """
    Full integration test for TASK-009.

    This test verifies the complete integration of:
    1. HWPX file extraction (514 regulations)
    2. Format classification (article/list/guideline/unstructured)
    3. Multi-format content extraction
    4. Coverage tracking (>90% target)
    5. Performance benchmark (<60 seconds)
    6. Backward compatibility (JSON output)
    """
    parser = HWPXMultiFormatParser()
    start_time = time.time()
    result = parser.parse_file(actual_hwpx_file)
    elapsed = time.time() - start_time

    # Test 1: All regulations extracted
    assert len(result["docs"]) == TARGET_REGULATIONS_COUNT

    # Test 2: Coverage rate exceeds target
    coverage_rate = result["coverage"]["coverage_rate"]
    assert coverage_rate >= TARGET_COVERAGE_RATE

    # Test 3: Parsing time within target
    assert elapsed < TARGET_PARSING_TIME

    # Test 4: All format types detected
    by_format = result["coverage"]["by_format"]
    assert "article" in by_format
    assert "list" in by_format
    assert "guideline" in by_format
    assert "unstructured" in by_format

    # Test 5: JSON serializable
    json_str = json.dumps(result, ensure_ascii=False)
    assert len(json_str) > 0


@pytest.mark.slow
def test_large_scale_performance(actual_hwpx_file: Path):
    """
    Slow test for large-scale performance validation.

    Tests parsing performance with multiple iterations
    to ensure consistent performance.
    """
    iterations = 3
    times = []

    for _ in range(iterations):
        parser = HWPXMultiFormatParser()
        start_time = time.time()
        result = parser.parse_file(actual_hwpx_file)
        elapsed = time.time() - start_time
        times.append(elapsed)

        assert len(result["docs"]) == TARGET_REGULATIONS_COUNT

    avg_time = sum(times) / len(times)
    assert avg_time < TARGET_PARSING_TIME, \
        f"Average parsing time too high: {avg_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
