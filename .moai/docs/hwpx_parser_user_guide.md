# HWPX Parser User Guide

**Version:** 3.5.0
**Last Updated:** 2026-02-11
**Reference:** SPEC-HWXP-002

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Introduction

The HWPX Parser v3.5.0 is a multi-format parsing system for Korean university regulation documents. It automatically detects and extracts content from four different regulation formats:

| Format | Description | Example |
|--------|-------------|---------|
| **Article** | Has clear article markers (제N조) | "제1조 (목적)", "제2조 (정의)" |
| **List** | Numbered or bulleted lists | "1. 학생은...", "2. 교수는..." |
| **Guideline** | Continuous prose without markers | Paragraphs of policy text |
| **Unstructured** | Ambiguous or mixed content | Converted documents, old formats |

**Key Improvements from v2.1:**
- Coverage: 43.6% → 90%+ (+46.4 percentage points)
- Regulations with content: 224 → 463+ (+239 regulations)
- Average content length: 500 → 800+ characters
- Multi-format detection and extraction
- Real-time coverage tracking

---

## Quick Start

### Installation

The HWPX Parser is included in the regulation_manager project:

```bash
cd /path/to/regulation_manager
uv sync
```

### Basic Parsing

```python
from pathlib import Path
from src.parsing.multi_format_parser import HWPXMultiFormatParser

# Initialize parser
parser = HWPXMultiFormatParser()

# Parse HWPX file
result = parser.parse_file(Path("data/input/규정집.hwpx"))

# View results
print(f"Total regulations: {result['metadata']['total_regulations']}")
print(f"Successfully parsed: {result['metadata']['successfully_parsed']}")
print(f"Coverage rate: {result['coverage']['coverage_rate']:.1f}%")
```

### Expected Output

```
Total regulations: 514
Successfully parsed: 514
Coverage rate: 90.1%
```

---

## Basic Usage

### Parsing a Single HWPX File

```python
from src.parsing.multi_format_parser import HWPXMultiFormatParser
from pathlib import Path

parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))

# Access individual regulations
for doc in result["docs"]:
    print(f"Title: {doc['title']}")
    print(f"Format: {doc['metadata']['format_type']}")
    print(f"Articles: {len(doc['articles'])}")
    print(f"Content length: {len(doc['content'])} chars")
    print()
```

### Saving Results to JSON

```python
import json
from pathlib import Path

# Parse file
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))

# Save to JSON
output_path = Path("data/output/parsed_regulations.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")
```

### Viewing Coverage Report

```python
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))

coverage = result["coverage"]

print(f"Total: {coverage['total']}")
print(f"With content: {coverage['with_content']}")
print(f"Coverage: {coverage['coverage_rate']:.1f}%")
print(f"Avg content: {coverage['avg_content_length']:.0f} chars")
print(f"Low coverage: {coverage['low_coverage_count']}")
print("\nBy format:")
for format_type, count in coverage['by_format'].items():
    print(f"  {format_type}: {count}")
```

---

## Advanced Usage

### Using Status Callbacks

Track parsing progress with a callback function:

```python
from src.parsing.multi_format_parser import HWPXMultiFormatParser
from pathlib import Path
from datetime import datetime

def progress_callback(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

parser = HWPXMultiFormatParser(status_callback=progress_callback)
result = parser.parse_file(Path("규정집.hwpx"))
```

**Example output:**

```
[14:23:45] Starting HWPX file parsing...
[14:23:45] Extracting Table of Contents...
[14:23:46] Aggregating content from sections...
[14:23:46] Validating TOC completeness...
[14:23:46] Extracting content for 514 regulations...
[14:23:47] Generating coverage report...
[14:23:47] Parsing complete: 514 regulations extracted
```

### Custom Format Classification

```python
from src.parsing.format.format_classifier import FormatClassifier

classifier = FormatClassifier()
result = classifier.classify(content)

# Check classification confidence
if result.confidence < 0.7:
    print(f"Low confidence: {result.confidence:.2f}")
    print(f"Indicators: {result.indicators}")
```

### Extracting Specific Regulation Formats

```python
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor
from src.parsing.analyzers.guideline_structure_analyzer import GuidelineStructureAnalyzer

# Extract list-format content
list_extractor = ListRegulationExtractor()
list_result = list_extractor.extract(list_content)
print(f"Extracted {list_result['total_items']} list items")

# Analyze guideline-format content
guideline_analyzer = GuidelineStructureAnalyzer()
guideline_result = guideline_analyzer.analyze(title, guideline_content)
print(f"Segmented into {len(guideline_result['provisions'])} provisions")
```

### Custom Coverage Tracking

```python
from src.parsing.metrics.coverage_tracker import CoverageTracker
from src.parsing.format.format_type import FormatType

tracker = CoverageTracker()

# Track regulations during processing
for regulation in regulations:
    format_type = FormatType(regulation["metadata"]["format_type"])
    has_content = len(regulation["content"]) > 0
    content_length = len(regulation["content"])

    tracker.track_regulation(format_type, has_content, content_length)

# Get detailed coverage report
report = tracker.get_coverage_report()
print(f"Coverage: {report.coverage_percentage:.1f}%")
```

---

## Configuration

### Environment Variables

No specific environment variables required for basic parsing.

### LLM Integration (Optional)

For unstructured regulation analysis, you can configure an LLM client:

```python
from src.parsing.multi_format_parser import HWPXMultiFormatParser
from src.rag.infrastructure.llm_client import LLMClient

# Initialize LLM client
llm_client = LLMClient(
    provider="ollama",
    model="gemma2",
    base_url="http://localhost:11434"
)

# Initialize parser with LLM client
parser = HWPXMultiFormatParser(llm_client=llm_client)
result = parser.parse_file(Path("규정집.hwpx"))
```

**Note:** LLM analysis is optional. Without LLM client, the parser uses rule-based extraction for all formats.

### Logging Configuration

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Parser will log detailed information
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))
```

---

## Troubleshooting

### Issue: Low Coverage Rate

**Symptoms:** Coverage rate below 80%

**Solutions:**

1. Check TOC completeness:

```python
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))

metadata = result["metadata"]
if not metadata["toc_complete"]:
    print(f"Missing {len(metadata['missing_titles'])} regulations")
    print("Missing:", metadata["missing_titles"])
```

2. Verify section content:

```python
# Check if sections are being read
import zipfile

with zipfile.ZipFile("규정집.hwpx", 'r') as zf:
    sections = [n for n in zf.namelist() if "section" in n]
    print("Found sections:", sections)
```

3. Enable debug logging:

```python
logging.basicConfig(level=logging.DEBUG)
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))
```

### Issue: Memory Error with Large Files

**Symptoms:** `MemoryError` when parsing large HWPX files

**Solutions:**

1. Process sections individually (future enhancement)
2. Increase system memory
3. Use streaming parsing (future enhancement)

### Issue: Format Misclassification

**Symptoms:** Wrong format type detected

**Solutions:**

1. Check classification indicators:

```python
classifier = FormatClassifier()
result = classifier.classify(content)

print(f"Format: {result.format_type}")
print(f"Confidence: {result.confidence}")
print(f"Indicators: {result.indicators}")
```

2. Manually override format classification (advanced):

```python
# Use format-specific extractors directly
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor

extractor = ListRegulationExtractor()
result = extractor.extract(content)
```

---

## Best Practices

### 1. Always Check Coverage

```python
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))

coverage = result["coverage"]
if coverage["coverage_rate"] < 80:
    print(f"Warning: Low coverage rate ({coverage['coverage_rate']:.1f}%)")
    print(f"Consider reviewing {coverage['low_coverage_count']} low-coverage regulations")
```

### 2. Validate Output Structure

```python
def validate_regulation(doc):
    """Validate regulation document structure."""
    required_fields = ["title", "content", "articles", "metadata"]
    for field in required_fields:
        if field not in doc:
            return False, f"Missing field: {field}"
    return True, "OK"

for doc in result["docs"]:
    valid, message = validate_regulation(doc)
    if not valid:
        print(f"Invalid document: {doc.get('title', 'Unknown')} - {message}")
```

### 3. Handle Edge Cases

```python
# Handle repealed regulations
for doc in result["docs"]:
    title = doc["title"]
    if "폐지" in title:
        print(f"Repealed regulation: {title}")
        continue

    # Process active regulations
    process_regulation(doc)
```

### 4. Monitor Performance

```python
import time
from pathlib import Path

start_time = time.time()
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))
elapsed_time = time.time() - start_time

regulations_per_second = len(result["docs"]) / elapsed_time
print(f"Parsed {len(result['docs'])} regulations in {elapsed_time:.2f}s")
print(f"Performance: {regulations_per_second:.1f} regs/sec")
```

### 5. Use Status Callbacks for Long Operations

```python
from datetime import datetime

def progress_callback(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("parser.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

parser = HWPXMultiFormatParser(status_callback=progress_callback)
result = parser.parse_file(Path("large_regulation.hwpx"))
```

---

## Migration from v2.1

### Replacing the Old Parser

**Before (v2.1):**

```python
from src.parsing.hwpx_direct_parser_v2 import HWPXDirectParser

parser = HWPXDirectParser()
result = parser.parse_file(Path("규정집.hwpx"))
```

**After (v3.5):**

```python
from src.parsing.multi_format_parser import HWPXMultiFormatParser

parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))
```

### Output Format Changes

**v2.1 Output:**

```json
{
  "docs": [...],
  "metadata": {
    "total_regulations": 224,
    "coverage_rate": 43.6
  }
}
```

**v3.5 Output:**

```json
{
  "docs": [...],
  "coverage": {
    "total": 514,
    "with_content": 463,
    "coverage_rate": 90.1,
    "by_format": {...}
  },
  "metadata": {
    "total_regulations": 514,
    "successfully_parsed": 514,
    "toc_complete": true
  }
}
```

---

## Additional Resources

- **API Documentation:** `.moai/docs/hwpx_parser_api.md`
- **Implementation Report:** `.moai/docs/hwpx_parser_implementation_report.md`
- **Test Suite:** `tests/parsing/`
- **Source Code:** `src/parsing/`

---

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review test files for usage examples
3. Enable debug logging for detailed diagnostics
4. Check the API documentation for component details

---

## License

MIT License - See LICENSE file for details.
