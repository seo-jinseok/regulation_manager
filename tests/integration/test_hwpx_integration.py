"""
Integration test for HWPX Direct Parser with actual HWPX file.

Tests the complete parsing pipeline with the target 규정집9-343(20250909).hwpx file.
"""
import json
import zipfile
from pathlib import Path

import pytest
import xml.etree.ElementTree as ET

from src.parsing.hwpx_direct_parser import HWPXDirectParser, HWPXTableParser
from src.parsing.regulation_article_extractor import (
    ParsingReportGenerator,
    RegulationArticleExtractor,
)


@pytest.fixture
def actual_hwpx_file():
    """Path to the actual HWPX file."""
    return Path("/Users/truestone/Dropbox/repo/University/regulation_manager/data/input/규정집9-343(20250909).hwpx")


@pytest.fixture
def hwpx_sections(actual_hwpx_file):
    """Extract all sections from the actual HWPX file."""
    parser = HWPXDirectParser()
    sections = parser._extract_sections_from_zip(actual_hwpx_file)
    return sections


class TestHWPXFileExtraction:
    """Test extraction from actual HWPX file."""

    def test_hwpx_file_exists(self, actual_hwpx_file):
        """Verify the HWPX file exists."""
        assert actual_hwpx_file.exists()
        assert actual_hwpx_file.stat().st_size > 4_000_000  # ~4MB

    def test_hwpx_is_valid_zip(self, actual_hwpx_file):
        """Verify the HWPX file is a valid ZIP archive."""
        with zipfile.ZipFile(actual_hwpx_file, 'r') as zf:
            assert len(zf.namelist()) > 0

    def test_extract_sections(self, hwpx_sections):
        """Test section extraction from HWPX file."""
        assert len(hwpx_sections) > 0

        # Check section names
        for section_name in hwpx_sections.keys():
            assert "Contents/section" in section_name
            assert section_name.endswith(".xml")

    def test_section_xml_validity(self, hwpx_sections):
        """Test that extracted sections contain valid XML."""
        for section_name, xml_content in hwpx_sections.items():
            try:
                root = ET.fromstring(xml_content)
                assert root is not None
            except ET.ParseError as e:
                pytest.fail(f"Invalid XML in {section_name}: {e}")

    def test_xml_has_namespace(self, hwpx_sections):
        """Test that XML sections have the expected namespace."""
        for section_name, xml_content in list(hwpx_sections.items())[:3]:  # Check first 3
            root = ET.fromstring(xml_content)
            # Check for HWPX namespace
            assert root.tag.startswith("{http://www.hancom.co.kr/hwpml/2011/")


class TestHWPXStructureParsing:
    """Test parsing of HWPX XML structure."""

    def test_extract_paragraph_text(self, hwpx_sections):
        """Test paragraph text extraction from actual XML."""
        parser = HWPXDirectParser()

        # Get first section
        section_content = next(iter(hwpx_sections.values()))
        root = ET.fromstring(section_content)

        # Find first paragraph
        p_elem = root.find('.//{http://www.hancom.co.kr/hwpml/2011/paragraph}p')
        if p_elem is not None:
            text = parser._extract_paragraph_text(p_elem)
            assert isinstance(text, str)

    def test_detect_regulation_titles(self, hwpx_sections):
        """Test regulation title detection from actual content."""
        parser = HWPXDirectParser()

        regulation_count = 0
        for section_content in hwpx_sections.values():
            root = ET.fromstring(section_content)

            for p_elem in root.iter():
                if p_elem.tag == f'{{{parser.ns["hp"]}}}p':
                    text = parser._extract_paragraph_text(p_elem)
                    if text and parser._is_regulation_title(text):
                        regulation_count += 1

        # Should find at least some regulations
        assert regulation_count > 0

    def test_detect_article_markers(self, hwpx_sections):
        """Test article marker detection from actual content."""
        parser = HWPXDirectParser()

        article_count = 0
        for section_content in hwpx_sections.values():
            root = ET.fromstring(section_content)

            for p_elem in root.iter():
                if p_elem.tag == f'{{{parser.ns["hp"]}}}p':
                    text = parser._extract_paragraph_text(p_elem)
                    if text and parser._is_article_marker(text):
                        article_count += 1

        # Should find many articles
        assert article_count > 100  # At least 100 articles expected


class TestHWPXFullParsing:
    """Test full HWPX file parsing."""

    def test_parse_full_file(self, actual_hwpx_file):
        """Test parsing the entire HWPX file."""
        parser = HWPXDirectParser()
        result = parser.parse_file(actual_hwpx_file)

        # Verify structure
        assert "metadata" in result
        assert "toc" in result
        assert "docs" in result

        # Verify metadata
        assert result["metadata"]["source_file"] == actual_hwpx_file.name
        assert result["metadata"]["parser_version"] == "2.0.0"
        assert "parsed_at" in result["metadata"]

        # Verify regulations were extracted
        assert len(result["docs"]) > 0

    def test_regulation_structure(self, actual_hwpx_file):
        """Test that parsed regulations have correct structure."""
        parser = HWPXDirectParser()
        result = parser.parse_file(actual_hwpx_file)

        if result["docs"]:
            first_reg = result["docs"][0]

            # Check required fields
            assert "id" in first_reg
            assert "kind" in first_reg
            assert "title" in first_reg
            assert first_reg["kind"] == "regulation"

    def test_toc_generation(self, actual_hwpx_file):
        """Test TOC generation from parsed documents."""
        parser = HWPXDirectParser()
        result = parser.parse_file(actual_hwpx_file)

        toc = result["toc"]
        assert len(toc) > 0

        # Check TOC structure
        first_toc = toc[0]
        assert "id" in first_toc
        assert "title" in first_toc
        assert "page" in first_toc

    def test_parsing_statistics(self, actual_hwpx_file):
        """Test parsing statistics tracking."""
        parser = HWPXDirectParser()
        result = parser.parse_file(actual_hwpx_file)

        metadata = result["metadata"]

        # Check statistics
        assert "total_regulations" in metadata
        assert "successfully_parsed" in metadata
        assert "success_rate" in metadata

        # Success rate should be between 0 and 100
        assert 0 <= metadata["success_rate"] <= 100


class TestArticleExtraction:
    """Test article extraction with real content."""

    def test_extract_article_from_real_content(self, hwpx_sections):
        """Test article extraction from actual HWPX content."""
        extractor = RegulationArticleExtractor()
        parser = HWPXDirectParser()

        # Get first section
        section_content = next(iter(hwpx_sections.values()))
        root = ET.fromstring(section_content)

        # Find first article
        for p_elem in root.iter():
            if p_elem.tag == f'{{{parser.ns["hp"]}}}p':
                text = parser._extract_paragraph_text(p_elem)
                if text and parser._is_article_marker(text):
                    # Try to extract article
                    article = extractor.extract_article(text)
                    if article:
                        assert article["article_no"]
                        break


class TestParsingReport:
    """Test parsing report generation."""

    def test_generate_comprehensive_report(self, actual_hwpx_file, tmp_path):
        """Test generating a comprehensive parsing report."""
        parser = HWPXDirectParser()
        result = parser.parse_file(actual_hwpx_file)

        # Generate report
        report = {
            "metadata": result["metadata"],
            "statistics": {
                "total_regulations": len(result["docs"]),
                "total_toc_items": len(result["toc"]),
                "avg_articles_per_regulation": sum(
                    len(doc.get("articles", [])) for doc in result["docs"]
                ) / max(len(result["docs"]), 1),
            },
            "sample_regulations": result["docs"][:5],  # First 5 regulations
        }

        # Verify report structure
        assert "metadata" in report
        assert "statistics" in report
        assert "sample_regulations" in report

        # Save report
        report_path = tmp_path / "parsing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        assert report_path.exists()


class TestPerformance:
    """Test parsing performance."""

    def test_parsing_speed(self, actual_hwpx_file):
        """Test that parsing completes in reasonable time."""
        import time

        parser = HWPXDirectParser()
        start_time = time.time()

        result = parser.parse_file(actual_hwpx_file)

        elapsed = time.time() - start_time

        # Should complete in under 60 seconds
        assert elapsed < 60, f"Parsing took {elapsed:.2f}s, expected < 60s"

        # Also verify we got results
        assert len(result["docs"]) > 0


@pytest.mark.integration
def test_full_pipeline_integration(actual_hwpx_file):
    """
    Full integration test for HWPX parsing pipeline.

    This test verifies:
    1. File extraction
    2. XML parsing
    3. Regulation detection
    4. Article extraction
    5. JSON output generation
    """
    # Parse file
    parser = HWPXDirectParser()
    result = parser.parse_file(actual_hwpx_file)

    # Validate output structure
    assert "metadata" in result
    assert "toc" in result
    assert "docs" in result

    # Validate metadata
    assert result["metadata"]["parser_version"] == "2.0.0"
    assert result["metadata"]["source_file"].endswith(".hwpx")

    # Validate TOC
    assert len(result["toc"]) > 0
    for toc_entry in result["toc"]:
        assert "id" in toc_entry
        assert "title" in toc_entry

    # Validate documents
    assert len(result["docs"]) > 0
    for doc in result["docs"]:
        assert "id" in doc
        assert "kind" in doc
        assert doc["kind"] == "regulation"
        assert "title" in doc

    # Statistics
    metadata = result["metadata"]
    assert metadata["total_regulations"] > 0
    assert metadata["success_rate"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
