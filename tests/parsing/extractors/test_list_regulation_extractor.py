"""
Test Suite for ListRegulationExtractor (TDD RED Phase)

This test suite follows TDD methodology - tests are written BEFORE implementation.
These tests will FAIL initially, driving the implementation in GREEN phase.

Reference: SPEC-HWXP-002, TASK-003
TDD Cycle: RED (write failing tests) -> GREEN (minimal implementation) -> REFACTOR
"""
import pytest
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# These imports will fail until we implement the classes
# This is expected in TDD RED phase


@dataclass
class TestListPattern:
    """Test data for list pattern detection."""
    name: str
    content: str
    expected_pattern: str
    expected_items: List[str]
    hierarchy_level: int = 0


class TestListPatternDetection:
    """
    Test list pattern detection (RED Phase).

    Tests will fail until ListRegulationExtractor is implemented.
    """

    @pytest.fixture
    def sample_list_contents(self) -> Dict[str, TestListPattern]:
        """Provide sample list contents for testing."""
        return {
            "numeric": TestListPattern(
                name="numeric_list",
                content="""1. 첫 번째 항목입니다.
2. 두 번째 항목입니다.
3. 세 번째 항목입니다.""",
                expected_pattern="numeric",
                expected_items=["첫 번째 항목입니다.", "두 번째 항목입니다.", "세 번째 항목입니다."]
            ),
            "korean_alphabet": TestListPattern(
                name="korean_alphabet_list",
                content="""가. 첫 번째 항목
나. 두 번째 항목
다. 세 번째 항목""",
                expected_pattern="korean",
                expected_items=["첫 번째 항목", "두 번째 항목", "세 번째 항목"]
            ),
            "circled_number": TestListPattern(
                name="circled_number_list",
                content="""① 첫 번째 사항
② 두 번째 사항
③ 세 번째 사항""",
                expected_pattern="circled",
                expected_items=["첫 번째 사항", "두 번째 사항", "세 번째 사항"]
            ),
            "mixed": TestListPattern(
                name="mixed_list",
                content="""1. 첫 번째 항목
① 첫 번째 하위 항목
② 두 번째 하위 항목
2. 두 번째 항목""",
                expected_pattern="mixed",
                expected_items=["첫 번째 항목", "첫 번째 하위 항목", "두 번째 하위 항목", "두 번째 항목"]
            )
        }

    def test_list_regulation_extractor_class_exists(self):
        """
        Test that ListRegulationExtractor class can be imported.
        This will FAIL until the class is created in GREEN phase.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor
        assert ListRegulationExtractor is not None

    def test_list_pattern_detection_numeric(self, sample_list_contents):
        """
        Test numeric list pattern detection (1., 2., 3.).
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        extractor = ListRegulationExtractor()
        test_data = sample_list_contents["numeric"]

        result = extractor.detect_pattern(test_data.content)

        assert result is not None
        assert result["pattern"] == test_data.expected_pattern

    def test_list_pattern_detection_korean_alphabet(self, sample_list_contents):
        """
        Test Korean alphabet list pattern detection (가., 나., 다.).
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        extractor = ListRegulationExtractor()
        test_data = sample_list_contents["korean_alphabet"]

        result = extractor.detect_pattern(test_data.content)

        assert result is not None
        assert result["pattern"] == test_data.expected_pattern

    def test_list_pattern_detection_circled_number(self, sample_list_contents):
        """
        Test circled number list pattern detection (①, ②, ③).
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        extractor = ListRegulationExtractor()
        test_data = sample_list_contents["circled_number"]

        result = extractor.detect_pattern(test_data.content)

        assert result is not None
        assert result["pattern"] == test_data.expected_pattern

    def test_list_pattern_detection_mixed(self, sample_list_contents):
        """
        Test mixed list pattern detection.
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        extractor = ListRegulationExtractor()
        test_data = sample_list_contents["mixed"]

        result = extractor.detect_pattern(test_data.content)

        assert result is not None
        assert result["pattern"] == test_data.expected_pattern


class TestNestedListExtraction:
    """
    Test nested list extraction with hierarchy preservation (RED Phase).

    Tests will fail until ListRegulationExtractor is implemented.
    """

    def test_extract_nested_list_hierarchy(self):
        """
        Test extracting nested lists with proper hierarchy (1. → ① → 가.).
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 첫 번째 항목
  ① 첫 번째 하위 항목
  ② 두 번째 하위 항목
2. 두 번째 항목
  ① 세 번째 하위 항목"""

        extractor = ListRegulationExtractor()
        result = extractor.extract_nested(content)

        assert result is not None
        assert "items" in result
        assert len(result["items"]) == 2  # Two top-level items

        # Check first item has nested items
        first_item = result["items"][0]
        assert "children" in first_item
        assert len(first_item["children"]) == 2

    def test_extract_nested_with_korean_alphabet(self):
        """
        Test extracting nested lists with Korean alphabet (가., 나., 다.).
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 첫 번째 항목
  ① 첫 번째 하위 항목
    가. 세부 항목 1
    나. 세부 항목 2
2. 두 번째 항목"""

        extractor = ListRegulationExtractor()
        result = extractor.extract_nested(content)

        assert result is not None
        assert len(result["items"]) == 2

        # Check deeply nested structure
        first_item = result["items"][0]
        assert "children" in first_item
        assert len(first_item["children"]) == 1

        first_child = first_item["children"][0]
        assert "children" in first_child
        assert len(first_child["children"]) == 2


class TestListToArticleConversion:
    """
    Test list-to-article conversion for RAG compatibility (RED Phase).

    Tests will fail until ListRegulationExtractor is implemented.
    """

    def test_convert_list_to_article_format(self):
        """
        Test converting list items to article-like format.
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 첫 번째 항목입니다. 이것은 중요한 내용을 담고 있습니다.
2. 두 번째 항목입니다. 이것도 중요한 내용입니다.
3. 세 번째 항목입니다. 마지막 항목입니다."""

        extractor = ListRegulationExtractor()
        result = extractor.to_article_format(content)

        assert result is not None
        assert "articles" in result

        # Should create article-like entries
        articles = result["articles"]
        assert len(articles) == 3

        # Check first article structure
        first_article = articles[0]
        assert "number" in first_article
        assert "content" in first_article
        assert first_article["number"] == 1

    def test_convert_nested_to_article_format(self):
        """
        Test converting nested lists to article format with hierarchy.
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 첫 번째 항목
  ① 첫 번째 하위 항목
  ② 두 번째 하위 항목
2. 두 번째 항목"""

        extractor = ListRegulationExtractor()
        result = extractor.to_article_format(content)

        assert result is not None
        articles = result["articles"]

        # Should preserve hierarchy in article numbers
        assert len(articles) >= 2

        # Check for hierarchical numbering
        article_numbers = [a["number"] for a in articles]
        # Should have main items like 1 and potentially sub-items
        assert 1 in article_numbers or any("1-" in str(n) for n in article_numbers)


class TestExtractionRate:
    """
    Test extraction rate for list-format regulations (RED Phase).

    Target: 90%+ extraction rate on test data
    """

    def test_extraction_rate_on_sample_data(self):
        """
        Test that extraction rate meets 90%+ target on pure list format.
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        # Sample list regulation content (pure list format)
        content = """1. 목적
사무관리의 효율화를 위한 사항을 정함을 목적으로 한다.
2. 정의
용어의 정의는 다음과 같다.
  ① 사무는 업무처리 과정에서 작성 또는 접수하는 문서를 말한다.
  ② 전자문서는 전자적 방식으로 작성하는 문서를 말한다.
3. 사무처리원칙
사무는 신속정확하고 효율적으로 처리하여야 한다.
4. 문서작성원칙
문서는 작성목적에 따라 적절한 형식을 갖추어 작성한다."""

        extractor = ListRegulationExtractor()
        result = extractor.extract(content)

        assert result is not None
        assert "extraction_rate" in result

        # Target: 90%+ extraction rate on pure list format
        assert result["extraction_rate"] >= 0.9

    def test_coverage_on_various_patterns(self):
        """
        Test coverage across various list patterns.
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        patterns = [
            "1. First item\n2. Second item",
            "가. 첫 번째\n나. 두 번째",
            "① First\n② Second",
            "1. Main\n  ① Sub\n  ② Sub"
        ]

        extractor = ListRegulationExtractor()

        for pattern in patterns:
            result = extractor.extract(pattern)
            assert result is not None, f"Failed to extract pattern: {pattern[:20]}..."
            assert "items" in result


class TestHierarchyPreservation:
    """
    Test hierarchy preservation in nested list extraction (RED Phase).
    """

    def test_parent_child_relationships(self):
        """
        Test that parent-child relationships are preserved.
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 상위 항목 1
  ① 하위 항목 1-1
  ② 하위 항목 1-2
2. 상위 항목 2
  ① 하위 항목 2-1"""

        extractor = ListRegulationExtractor()
        result = extractor.extract_nested(content)

        assert result is not None
        items = result["items"]

        # Verify parent-child structure
        assert len(items) == 2

        # First parent should have 2 children
        parent1 = items[0]
        assert parent1["level"] == 0
        assert len(parent1["children"]) == 2

        # Children should have proper level
        child1 = parent1["children"][0]
        assert child1["level"] == 1

    def test_deeply_nested_hierarchy(self):
        """
        Test extraction of deeply nested lists (3+ levels).
        Will FAIL until implemented.
        """
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 레벨 1 항목
  ① 레벨 2 항목
    가. 레벨 3 항목
      1) 레벨 4 항목"""

        extractor = ListRegulationExtractor()
        result = extractor.extract_nested(content)

        assert result is not None
        items = result["items"]

        # Should handle at least 3 levels
        assert len(items) == 1

        current = items[0]
        level_count = 0
        while current and "children" in current and len(current["children"]) > 0:
            level_count += 1
            current = current["children"][0]

        assert level_count >= 3, "Should extract at least 3 levels of hierarchy"


# Test data for edge cases
class TestEdgeCases:
    """
    Test edge cases in list extraction (RED Phase).
    """

    def test_empty_content(self):
        """Test handling of empty content."""
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        extractor = ListRegulationExtractor()
        result = extractor.extract("")

        assert result is not None
        assert "items" in result
        assert len(result["items"]) == 0

    def test_mixed_content_with_prose(self):
        """Test handling of content mixing lists and prose."""
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """이 규정의 목적은 다음과 같다.

1. 첫 번째 목적
2. 두 번째 목적

위와 같은 목적을 달성하기 위하여 다음과 같이 정한다.

가. 시행일자
나. 주요 내용"""

        extractor = ListRegulationExtractor()
        result = extractor.extract(content)

        assert result is not None
        # Should extract list items while ignoring prose
        assert len(result["items"]) > 0

    def test_inconsistent_indentation(self):
        """Test handling of inconsistent indentation."""
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. First item
  ① Indented sub-item
     ① Double-indented sub-sub-item
2. Second item"""

        extractor = ListRegulationExtractor()
        result = extractor.extract_nested(content)

        assert result is not None
        # Should handle varying indentation
        assert len(result["items"]) > 0


class TestIntegrationWithFormatClassifier:
    """
    Test integration with FormatClassifier (RED Phase).
    """

    def test_classifier_result_to_extractor_input(self):
        """
        Test that extractor can work with FormatClassifier output.
        Will FAIL until implemented.
        """
        from src.parsing.format.format_classifier import FormatClassifier
        from src.parsing.format.format_type import FormatType
        from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

        content = """1. 첫 번째 항목
2. 두 번째 항목
3. 세 번째 항목"""

        # First classify the content
        classifier = FormatClassifier()
        classification = classifier.classify(content)

        # Should be LIST format
        assert classification.format_type == FormatType.LIST

        # Then extract using the list pattern info
        extractor = ListRegulationExtractor()
        result = extractor.extract_with_pattern(
            content,
            classification.list_pattern
        )

        assert result is not None
        assert "items" in result
        assert len(result["items"]) == 3
