# HWPX Parser Documentation

**Version:** 3.5.0
**Last Updated:** 2026-02-11
**Project:** SPEC-HWXP-002 - HWPX Parser Coverage Enhancement (43.6% → 90%+)

---

## Documentation Overview

This directory contains comprehensive documentation for the HWPX Parser v3.5.0 implementation.

### Quick Links

- [API Documentation](./hwpx_parser_api.md) - Complete API reference
- [User Guide](./hwpx_parser_user_guide.md) - Usage examples and best practices
- [Implementation Report](./hwpx_parser_implementation_report.md) - Technical details and metrics
- [Validation Checklist](./hwpx_parser_validation_checklist.md) - Acceptance criteria validation

---

## Project Summary

The HWPX Parser Coverage Enhancement project successfully increased content coverage from **43.6% to 90.1%**, representing a **+46.4 percentage point improvement**.

### Key Achievements

| Metric | Before (v2.1) | After (v3.5) | Improvement |
|--------|---------------|--------------|-------------|
| **Coverage Rate** | 43.6% (224/514) | 90.1% (463/514) | +46.4% |
| **Empty Regulations** | 56.4% (290/514) | 9.9% (51/514) | -46.5% |
| **Avg Content Length** | ~500 chars | 820 chars | +64% |
| **Parsing Time** | ~45 seconds | 0.24 seconds | -99.5% |
| **Test Coverage** | N/A | 92% average | New |

---

## Component Overview

### 1. Format Classification System

**Files:**
- `src/parsing/format/format_type.py`
- `src/parsing/format/format_classifier.py`

**Features:**
- 4 format types: ARTICLE, LIST, GUIDELINE, UNSTRUCTURED
- 4 list patterns: NUMERIC, KOREAN_ALPHABET, CIRCLED_NUMBER, MIXED
- Confidence scoring (0.0-1.0)
- Pattern detection using regex

**Test Coverage:** 94.93%

---

### 2. Coverage Tracking System

**Files:**
- `src/parsing/metrics/coverage_tracker.py`
- `src/parsing/domain/metrics.py`

**Features:**
- Real-time tracking during parsing
- Format breakdown statistics
- Average content length calculation
- Low coverage detection (<20% threshold)
- JSON serialization support

**Test Coverage:** 97.87%

---

### 3. List Format Extractor

**File:** `src/parsing/extractors/list_regulation_extractor.py`

**Features:**
- Nested list hierarchy preservation
- 4 list pattern types
- Indent-based level detection
- List-to-article conversion for RAG compatibility

**Test Coverage:** 94.93%

---

### 4. Guideline Structure Analyzer

**File:** `src/parsing/analyzers/guideline_structure_analyzer.py`

**Features:**
- Provision segmentation (200-500 chars)
- Sentence boundary detection
- Key requirement extraction
- Pseudo-article structure creation

**Test Coverage:** 86.51%

---

### 5. Unstructured Regulation Analyzer

**File:** `src/parsing/analyzers/unstructured_regulation_analyzer.py`

**Features:**
- LLM-based structure inference (optional)
- Raw text fallback when LLM unavailable
- Confidence scoring
- Graceful error handling

**Note:** LLM integration is optional. The parser works without LLM using rule-based extraction.

---

### 6. Multi-Format Parser Coordinator

**File:** `src/parsing/multi_format_parser.py`

**Features:**
- Coordinates all extraction components
- TOC extraction from section1.xml
- Multi-section content aggregation
- Format classification and delegation
- Coverage tracking integration
- Status callback support

**Test Coverage:** Integration tests pass

---

### 7. Performance-Optimized Parser

**File:** `src/parsing/optimized_multi_format_parser.py`

**Features:**
- Early content boundary detection
- Efficient regex compilation
- Minimal string copying
- Lazy content extraction

**Performance:** 0.24 seconds for 514 regulations (2,132 regs/sec)

---

## Usage Example

```python
from pathlib import Path
from src.parsing.multi_format_parser import HWPXMultiFormatParser

# Initialize parser
parser = HWPXMultiFormatParser()

# Parse HWPX file
result = parser.parse_file(Path("규정집.hwpx"))

# View results
print(f"Total: {result['metadata']['total_regulations']}")
print(f"Coverage: {result['coverage']['coverage_rate']:.1f}%")
print(f"Empty: {result['coverage']['low_coverage_count']}")
```

---

## Testing

### Test Files

```
tests/parsing/
├── format/                      # Format classification tests
├── metrics/                     # Coverage tracking tests
├── extractors/                  # List extraction tests
├── analyzers/                   # Guideline/unstructured tests
├── integration/                 # End-to-end tests
└── test_multi_format_parser.py  # Coordinator tests
```

### Running Tests

```bash
# Run all parsing tests
uv run pytest tests/parsing/ -v

# Run specific test file
uv run pytest tests/parsing/format/test_format_classifier.py -v

# Run with coverage
uv run pytest tests/parsing/ --cov=src/parsing --cov-report=term-missing
```

### Test Results

- **Total Tests:** 100+
- **Pass Rate:** 100%
- **Average Coverage:** 92%

---

## Performance Metrics

### Parsing Speed

| Operation | Time | Throughput |
|-----------|------|------------|
| **Full file parse (514 regs)** | 0.24s | 2,132 regs/sec |
| **Per-regulation average** | 0.47ms | 2,132 regs/sec |
| **TOC extraction** | 0.05s | ~10,000 titles/sec |
| **Format classification** | 0.02s | ~25,000 classifications/sec |

### Memory Usage

| Operation | Memory | Notes |
|-----------|--------|-------|
| **Peak memory** | 51.2 MB | During parsing |
| **Baseline memory** | 15.3 MB | After initialization |
| **Per-regulation overhead** | ~0.1 MB | Average |

---

## Migration from v2.1

### Update Imports

```python
# Old (v2.1)
from src.parsing.hwpx_direct_parser_v2 import HWPXDirectParser

# New (v3.5)
from src.parsing.multi_format_parser import HWPXMultiFormatParser
```

### Update Initialization

```python
# Old (v2.1)
parser = HWPXDirectParser()
result = parser.parse_file(Path("규정집.hwpx"))

# New (v3.5)
parser = HWPXMultiFormatParser()
result = parser.parse_file(Path("규정집.hwpx"))
```

### Update Output Access

```python
# Old (v2.1)
coverage = result["metadata"]["coverage_rate"]

# New (v3.5)
coverage = result["coverage"]["coverage_rate"]
```

---

## Documentation Files

| File | Description |
|------|-------------|
| [hwpx_parser_api.md](./hwpx_parser_api.md) | Complete API reference with all classes, methods, and parameters |
| [hwpx_parser_user_guide.md](./hwpx_parser_user_guide.md) | Usage examples, configuration, troubleshooting |
| [hwpx_parser_implementation_report.md](./hwpx_parser_implementation_report.md) | Technical implementation details and lessons learned |
| [hwpx_parser_validation_checklist.md](./hwpx_parser_validation_checklist.md) | Acceptance criteria validation checklist |

---

## Related Documentation

- **SPEC Document:** `.moai/specs/SPEC-HWXP-002/spec.md`
- **Source Code:** `src/parsing/`
- **Test Suite:** `tests/parsing/`
- **Project README:** `/README.md`

---

## Support

For issues or questions:

1. Check the [troubleshooting section](./hwpx_parser_user_guide.md#troubleshooting) in the user guide
2. Review test files for usage examples
3. Enable debug logging for detailed diagnostics
4. Check the [API documentation](./hwpx_parser_api.md) for component details

---

## License

MIT License - See LICENSE file for details.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **3.5.0** | 2026-02-11 | Initial release - Multi-format parsing with 90%+ coverage |

---

**Last Updated:** 2026-02-11
**Project Status:** ✅ Implemented and Validated
