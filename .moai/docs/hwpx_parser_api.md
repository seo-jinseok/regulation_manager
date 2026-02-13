# HWPX Parser API Documentation

**Version:** 3.5.0
**Last Updated:** 2026-02-11
**Reference:** SPEC-HWXP-002

---

## Overview

The HWPX Parser API provides multi-format parsing capabilities for Korean university regulation documents in HWPX format. The parser supports four regulation formats: Article, List, Guideline, and Unstructured, achieving 90%+ content coverage.

---

## Core Components

### 1. Format Classification

#### `FormatType` Enum

```python
from src.parsing.format.format_type import FormatType

class FormatType(Enum):
    ARTICLE = "article"        # Clear article markers (제N조)
    LIST = "list"              # Numbered/bulleted lists
    GUIDELINE = "guideline"    # Continuous prose
    UNSTRUCTURED = "unstructured"  # Ambiguous content
```

#### `FormatClassifier` Class

```python
from src.parsing.format.format_classifier import FormatClassifier, ClassificationResult

classifier = FormatClassifier()
result = classifier.classify(content)

# Access classification results
result.format_type        # FormatType enum
result.confidence         # float: 0.0-1.0
result.list_pattern       # ListPattern (for LIST format)
result.indicators         # Dict of detection metrics
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `classify(content: str)` | Classify content format | `ClassificationResult` |

**Classification Rules:**

1. **Article Format:** Contains `제N조` markers (e.g., "제1조", "제2조의3")
2. **List Format:** Contains 2+ list markers (1., 2., 가., 나., ①, ②)
3. **Guideline Format:** Continuous prose without clear markers
4. **Unstructured:** Ambiguous or empty content

---

### 2. Format-Specific Extractors

#### `ListRegulationExtractor` Class

```python
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

extractor = ListRegulationExtractor()
result = extractor.extract(content)

# Access extraction results
result["items"]            # List of list items
result["pattern"]          # Detected list pattern
result["total_items"]      # Total count
result["extraction_rate"]  # Success ratio (0.0-1.0)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `detect_pattern(content: str)` | Detect list pattern type | `Dict` with pattern info |
| `extract(content: str)` | Extract list items (flat) | `Dict` with items |
| `extract_nested(content: str)` | Extract nested list items | `Dict` with hierarchy |
| `to_article_format(content: str)` | Convert to article format | `Dict` with articles |

**Supported List Patterns:**

- `NUMERIC`: 1., 2., 3. (Western numerals)
- `KOREAN_ALPHABET`: 가., 나., 다. (Korean alphabet)
- `CIRCLED_NUMBER`: ①, ②, ③ (Circled numbers)
- `MIXED`: Combination of multiple patterns

---

#### `GuidelineStructureAnalyzer` Class

```python
from src.parsing.analyzers.guideline_structure_analyzer import GuidelineStructureAnalyzer

analyzer = GuidelineStructureAnalyzer()
result = analyzer.analyze(title, content)

# Access analysis results
result["provisions"]       # List of provision segments
result["key_requirements"] # Extracted requirements
result["articles"]         # Pseudo-article structure
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `analyze(title: str, content: str)` | Analyze guideline structure | `Dict` with provisions |
| `_segment_provisions(content: str)` | Segment into provisions | `List[str]` |
| `_extract_key_requirements(provision: str)` | Extract requirements | `List[str]` |

**Segmentation Strategy:**

- Split by paragraph breaks
- Further split long paragraphs by sentences
- Target: 200-500 characters per provision
- Maximum: 500 characters per provision

---

#### `UnstructuredRegulationAnalyzer` Class

```python
from src.parsing.analyzers.unstructured_regulation_analyzer import UnstructuredRegulationAnalyzer

analyzer = UnstructuredRegulationAnalyzer(llm_client=None)
result = analyzer.analyze(title, content)

# Access analysis results
result["articles"]         # Extracted articles
result["confidence"]       # Confidence score (0.0-1.0)
result["llm_enhanced"]     # Whether LLM was used
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `analyze(title: str, content: str)` | Analyze unstructured content | `Dict` with articles |

**Note:** LLM analysis is optional and controlled by `llm_client` parameter. Without LLM client, falls back to raw text extraction.

---

### 3. Coverage Tracking

#### `CoverageTracker` Class

```python
from src.parsing.metrics.coverage_tracker import CoverageTracker
from src.parsing.format.format_type import FormatType

tracker = CoverageTracker()
tracker.track_regulation(
    format_type=FormatType.ARTICLE,
    has_content=True,
    content_length=1000
)

# Generate coverage report
report = tracker.get_coverage_report()

# Access report metrics
report.total_regulations          # Total count
report.regulations_with_content   # Count with content
report.coverage_percentage        # Coverage rate (0-100)
report.format_breakdown           # Dict[FormatType, int]
report.avg_content_length         # Average characters
report.low_coverage_count         # Count with <20% coverage
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `track_regulation(format_type, has_content, content_length)` | Track one regulation | `None` |
| `get_coverage_report()` | Generate coverage report | `CoverageReport` |
| `get_low_coverage_regulations(threshold=0.2)` | Get low-coverage IDs | `List[str]` |
| `to_dict()` | Convert to dict for JSON | `Dict[str, Any]` |

---

#### `CoverageReport` Dataclass

```python
from src.parsing.domain.metrics import CoverageReport

report = CoverageReport(
    total_regulations=514,
    regulations_with_content=463,
    coverage_percentage=90.1,
    format_breakdown={FormatType.ARTICLE: 224, ...},
    avg_content_length=820.0,
    low_coverage_count=51
)

# Convert to dictionary
report_dict = report.to_dict()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `total_regulations` | `int` | Total regulations tracked |
| `regulations_with_content` | `int` | Regulations with content |
| `coverage_percentage` | `float` | Coverage rate (0-100) |
| `format_breakdown` | `Dict[FormatType, int]` | Count by format type |
| `avg_content_length` | `float` | Average content in characters |
| `low_coverage_count` | `int` | Regulations with <20% coverage |

---

### 4. Main Parser Coordinator

#### `HWPXMultiFormatParser` Class

```python
from src.parsing.multi_format_parser import HWPXMultiFormatParser

# Initialize parser
parser = HWPXMultiFormatParser(
    llm_client=None,  # Optional LLM client
    status_callback=None  # Optional progress callback
)

# Parse HWPX file
result = parser.parse_file(file_path)

# Access parsing results
result["docs"]          # List of regulation entries
result["coverage"]      # Coverage report dict
result["metadata"]      # Parsing metadata
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `parse_file(file_path: Path)` | Parse HWPX file | `Dict[str, Any]` |

**Parsing Workflow:**

```
1. Extract TOC from section1.xml
2. Aggregate content from all sections
3. Validate TOC completeness
4. For each TOC entry:
   a. Extract relevant content
   b. Classify format
   c. Delegate to appropriate extractor
   d. Track coverage
5. Generate coverage report
6. Return parsing result
```

---

## Usage Examples

### Example 1: Classify Regulation Format

```python
from src.parsing.format.format_classifier import FormatClassifier

classifier = FormatClassifier()

# Article format
content1 = "제1조 (목적) 이 규정은..."
result1 = classifier.classify(content1)
print(result1.format_type)  # FormatType.ARTICLE
print(result1.confidence)   # 0.85+

# List format
content2 = "1. 학생은 매학기...\n2. 교수는..."
result2 = classifier.classify(content2)
print(result2.format_type)  # FormatType.LIST
```

### Example 2: Extract List-Format Regulation

```python
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

extractor = ListRegulationExtractor()
content = """
1. 학칙 개정은 다음 절차에 따른다.
2. 학칙 제정은 교무회의 심의를 거친다.
"""

result = extractor.extract(content)
print(result["pattern"])  # "numeric"
print(result["total_items"])  # 2
```

### Example 3: Track Coverage During Parsing

```python
from src.parsing.metrics.coverage_tracker import CoverageTracker
from src.parsing.format.format_type import FormatType

tracker = CoverageTracker()

# Track regulations
tracker.track_regulation(FormatType.ARTICLE, True, 1000)
tracker.track_regulation(FormatType.LIST, False, 0)
tracker.track_regulation(FormatType.GUIDELINE, True, 500)

# Get coverage report
report = tracker.get_coverage_report()
print(f"Coverage: {report.coverage_percentage:.1f}%")
print(f"Average content: {report.avg_content_length:.0f} chars")
```

### Example 4: Parse HWPX File

```python
from pathlib import Path
from src.parsing.multi_format_parser import HWPXMultiFormatParser

# Initialize parser with status callback
def status_callback(message):
    print(f"[STATUS] {message}")

parser = HWPXMultiFormatParser(status_callback=status_callback)

# Parse file
result = parser.parse_file(Path("규정집.hwpx"))

# Access results
print(f"Parsed: {result['metadata']['successfully_parsed']} regulations")
print(f"Coverage: {result['coverage']['coverage_rate']:.1f}%")
```

---

## Output Format

### Regulation Entry Structure

```json
{
  "title": "규정명",
  "content": "원본 내용...",
  "articles": [
    {
      "number": 1,
      "content": "조항 내용..."
    }
  ],
  "provisions": ["조항 내용..."],
  "metadata": {
    "format_type": "article",
    "confidence": 0.95,
    "coverage_score": 0.90,
    "extraction_rate": 0.95
  }
}
```

### Coverage Report Structure

```json
{
  "total": 514,
  "with_content": 463,
  "coverage_rate": 90.1,
  "by_format": {
    "article": 224,
    "list": 150,
    "guideline": 80,
    "unstructured": 9
  },
  "avg_content_length": 820.0,
  "low_coverage_count": 51
}
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Coverage Rate** | 90%+ (vs 43.6% baseline) |
| **Parsing Time** | ~0.24s per file |
| **Test Coverage** | 87-97% per component |
| **Format Classification Accuracy** | >85% |

---

## Integration Points

### With Existing Parser

```python
# HWPXMultiFormatParser is backward compatible
# and can be used as a drop-in replacement
from src.parsing.multi_format_parser import HWPXMultiFormatParser

# Use instead of hwpx_direct_parser_v2
parser = HWPXMultiFormatParser()
result = parser.parse_file(hwpx_file)
```

### With RAG Pipeline

```python
# Parse HWPX and feed directly to RAG system
from src.parsing.multi_format_parser import HWPXMultiFormatParser
from src.rag.infrastructure.json_loader import JSONLoader

parser = HWPXMultiFormatParser()
result = parser.parse_file(hwpx_file)

# Use docs array for RAG indexing
loader = JSONLoader()
docs = loader.load_data(result["docs"])
```

---

## Error Handling

All components use Python's built-in `logging` module:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Parser will log:
# - TOC extraction issues
# - Content extraction failures
# - Classification low confidence
# - Coverage warnings
```

---

## Testing

Each component has comprehensive unit tests:

```bash
# Test format classification
pytest tests/parsing/format/test_format_classifier.py

# Test list extraction
pytest tests/parsing/extractors/test_list_regulation_extractor.py

# Test coverage tracking
pytest tests/parsing/metrics/test_coverage_tracker.py

# Test integration
pytest tests/parsing/integration/test_task009_integration.py
```

---

## License

MIT License - See LICENSE file for details.
