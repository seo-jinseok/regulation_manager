# Acceptance Criteria: SPEC-HWXP-002

**TAG BLOCK**
```
SPEC: HWXP-002
Title: HWPX Parser Coverage Enhancement (43.6% → 90%+)
Related: spec.md, plan.md
Implementation Version: 3.5.0
Created: 2026-02-11
```

---

## Overview

This document defines the acceptance criteria for HWPX Parser Coverage Enhancement. Each criterion is specified in Given-When-Then format for clear verification and testing.

---

## Format Classification Acceptance Criteria

### AC-001: Article Format Classification

**Given** a regulation with clear article markers (`제1조`, `제2조`, etc.)
**When** the format classifier analyzes the content
**Then** the format shall be classified as `FormatType.ARTICLE`
**And** the classification confidence shall be >0.9

**Example:**
```python
# Given
content = "제1조 (목적)\n이 규정은...\n제2조 (정의)\n이 규정에서..."

# When
format_type = classifier.classify(content)

# Then
assert format_type == FormatType.ARTICLE
```

---

### AC-002: List Format Classification

**Given** a regulation with bullet/numbered lists but no article markers
**When** the format classifier analyzes the content
**Then** the format shall be classified as `FormatType.LIST`
**And** the classification confidence shall be >0.8

**Example:**
```python
# Given
content = "1. 목적\n이 규정은...\n2. 정의\n이 규정에서..."

# When
format_type = classifier.classify(content)

# Then
assert format_type == FormatType.LIST
```

---

### AC-003: Guideline Format Classification

**Given** a regulation with continuous prose and no clear structure
**When** the format classifier analyzes the content
**Then** the format shall be classified as `FormatType.GUIDELINE`
**And** the classification confidence shall be >0.7

**Example:**
```python
# Given
content = "이 지침은 대학의 연구윤리를 확립하기 위한 것으로..."

# When
format_type = classifier.classify(content)

# Then
assert format_type == FormatType.GUIDELINE
```

---

### AC-004: Classification Accuracy Threshold

**Given** a sample set of 100+ regulations with known formats
**When** the format classifier processes all samples
**Then** the classification accuracy shall be >85%
**And** misclassifications shall be logged for review

---

## List Format Extraction Acceptance Criteria

### AC-005: Numeric List Extraction

**Given** a list-format regulation with numeric items (1., 2., 3.)
**When** the ListRegulationExtractor processes the content
**Then** all numeric items shall be extracted
**And** item numbering shall be preserved
**And** item content shall be complete

**Example:**
```python
# Given
content = "겸임교원규정\n1. 겸임 범위\n대학교원은...\n2. 겸임 절차\n..."

# When
regulation = extractor.extract(title="겸임교원규정", content=content)

# Then
assert len(regulation.articles) == 2
assert regulation.articles[0]['article_no'] == "제1조"
assert regulation.articles[0]['title'] == "겸임 범위"
```

---

### AC-006: Korean Alphabet List Extraction

**Given** a list-format regulation with Korean alphabet items (가., 나., 다.)
**When** the ListRegulationExtractor processes the content
**Then** all alphabet items shall be extracted
**And** item hierarchy shall be preserved

**Example:**
```python
# Given
content = "연구윤리지침\n가) 목적\n연구윤리를...\n나) 적용 범위\n..."

# When
regulation = extractor.extract(title="연구윤리지침", content=content)

# Then
assert len(regulation.articles) >= 2
assert regulation.articles[0]['title'] == "목적"
```

---

### AC-007: Nested List Hierarchy Preservation

**Given** a list-format regulation with nested items (1. → ① → 가.)
**When** the ListRegulationExtractor processes the content
**Then** the full hierarchy shall be preserved
**And** parent-child relationships shall be maintained

**Example:**
```python
# Given
content = """1. 연구자의 의무
① 연구윤리 준수
가) 진실성 원칙
나) 투명성 원칙
② 연구 부정행위 금지"""

# When
regulation = extractor.extract(title="연구윤리지침", content=content)

# Then
assert regulation.articles[0]['items'][0]['number'] == "①"
assert regulation.articles[0]['items'][0]['subitems'][0]['number'] == "가"
```

---

### AC-008: List to Article Conversion

**Given** extracted list items
**When** the converter generates article structures
**Then** each list item shall become a pseudo-article
**And** article numbers shall follow sequence (제1조, 제2조, ...)
**And** list item text shall become article title

---

## Guideline Format Analysis Acceptance Criteria

### AC-009: Provision Segmentation

**Given** a guideline-format regulation with continuous prose
**When** the GuidelineStructureAnalyzer segments the content
**Then** provisions shall be segmented at logical boundaries
**And** each provision shall be <500 characters
**And** content semantics shall be preserved

**Example:**
```python
# Given
content = """이 지침은 대학의 연구윤리를 확립하기 위한 것으로,
연구자가 준수해야 할 기본 원칙과 절차를 정한다.
연구윤리위원회는 이 지침의 해석과 운영에 관한 사항을 심의한다."""

# When
provisions = analyzer.segment_provisions(content)

# Then
assert len(provisions) >= 2
assert all(len(p) < 500 for p in provisions)
assert "연구윤리" in provisions[0]
```

---

### AC-010: Key Requirement Extraction

**Given** a segmented provision
**When** the analyzer extracts key requirements
**Then** actionable requirements shall be identified
**And** requirement verbs (shall, must, require) shall be detected
**And** extraction confidence shall be calculated

---

### AC-011: Pseudo-Article Generation

**Given** segmented provisions from guideline content
**When** the analyzer generates pseudo-articles
**Then** each provision shall become a pseudo-article
**And** pseudo-article titles shall summarize provision content
**And** original content shall be preserved in article body

---

## LLM Fallback Acceptance Criteria

### AC-012: Unstructured Format Detection

**Given** a regulation that doesn't match article/list/guideline patterns
**When** the format classifier analyzes the content
**Then** the format shall be classified as `FormatType.UNSTRUCTURED`
**And** LLM fallback shall be triggered

---

### AC-013: LLM Structure Inference

**Given** an unstructured regulation
**When** the UnstructuredRegulationAnalyzer processes the content
**Then** the LLM shall infer article/provision structure
**And** inference results shall be parsed into Regulation format
**And** confidence score shall be calculated

**Example:**
```python
# Given
content = "대학평의회규정\n(복잡한 구조의 규정 내용)"

# When
regulation, confidence = analyzer.analyze(title="대학평의회규정", content=content)

# Then
assert len(regulation.articles) > 0 or confidence < 0.5
assert 0 <= confidence <= 1
```

---

### AC-014: LLM Confidence Threshold

**Given** LLM inference results with confidence <0.5
**When** the confidence is below threshold
**Then** the regulation shall be flagged for manual review
**And** raw text shall be stored as fallback content

---

### AC-015: LLM Timeout Handling

**Given** an LLM inference request that exceeds 30 seconds
**When** the timeout is triggered
**Then** the request shall be cancelled
**And** raw text extraction shall be used as fallback
**And** timeout event shall be logged

---

## Coverage Tracking Acceptance Criteria

### AC-016: Real-Time Coverage Metrics

**Given** an active parsing operation
**When** regulations are processed
**Then** coverage metrics shall be updated in real-time
**And** format breakdown shall be tracked
**And** low-coverage regulations shall be identified

**Example:**
```python
# Given
tracker = CoverageTracker()

# When
tracker.track_regulation(FormatType.ARTICLE, has_content=True)
tracker.track_regulation(FormatType.LIST, has_content=False)

# Then
report = tracker.get_coverage_report()
assert report.total_regulations == 2
assert report.regulations_with_content == 1
assert report.format_breakdown[FormatType.ARTICLE] == 1
```

---

### AC-017: Coverage Report Generation

**Given** completed parsing of 514 regulations
**When** coverage report is generated
**Then** the report shall include:
- Total regulations: 514
- Regulations with content: >=463 (90%+)
- Coverage rate: >=90%
- Format breakdown by type
- Average content length: >=800 characters

---

### AC-018: Low-Coverage Identification

**Given** a parsed regulation with <20% content coverage
**When** the coverage tracker analyzes the regulation
**Then** the regulation shall be flagged as low-coverage
**And** LLM fallback shall be attempted
**And** flag shall be included in coverage report

---

## Multi-Format Parser Acceptance Criteria

### AC-019: TOC-Driven Parsing

**Given** an HWPX file with section1.xml containing 514 TOC entries
**When** the multi-format parser extracts the TOC
**Then** all 514 regulation titles shall be extracted
**And** each title shall have valid format classification
**And** TOC completeness shall be validated

---

### AC-020: Format Delegation

**Given** a TOC entry with classified format type
**When** the multi-format parser delegates to appropriate extractor
**Then** the correct extractor shall be invoked
**And** extraction results shall be returned
**And** extraction metrics shall be tracked

**Example:**
```python
# Given
toc_entry = {"title": "겸임교원규정", "format_type": FormatType.ARTICLE}

# When
regulation = parser._extract_with_format(toc_entry, content, FormatType.ARTICLE)

# Then
assert regulation is not None
assert regulation.format_type == FormatType.ARTICLE
```

---

### AC-021: Multi-Section Aggregation

**Given** an HWPX file with multiple sections (section0.xml, section1.xml, section2.xml)
**When** the parser aggregates content from all sections
**Then** content shall be merged correctly
**And** duplicate entries shall be eliminated
**And** section priority shall be respected (section0 > section1 > section2)

---

### AC-022: Empty Regulation Creation

**Given** a TOC entry with no matching content in any section
**When** the parser completes processing
**Then** an empty regulation entry shall still be created
**And** the regulation shall have `empty=True` flag
**And** the regulation title shall match TOC entry

---

## Performance Acceptance Criteria

### AC-023: Parsing Time Threshold

**Given** a target HWPX file with 514 regulations
**When** the multi-format parser processes the entire file
**Then** the total parsing time shall be <60 seconds
**And** time per regulation shall average <0.12 seconds

---

### AC-024: Memory Usage Threshold

**Given** an active parsing operation on the target file
**When** memory usage is monitored
**Then** peak memory usage shall be <2GB
**And** memory shall be released after each section is processed

---

### AC-025: LLM Inference Performance

**Given** a regulation requiring LLM fallback
**When** LLM inference is executed
**Then** the inference time shall be <5 seconds
**And** the result shall be returned within timeout period

---

## Quality Acceptance Criteria

### AC-026: Content Quality Threshold

**Given** a parsed regulation with extracted content
**When** content quality is assessed
**Then** the content shall be >100 characters (for non-empty regulations)
**And** Korean text shall be properly encoded
**And** special characters shall be preserved

---

### AC-027: Backward Compatibility

**Given** existing RAG pipeline processing Regulation objects
**When** the new multi-format parser output is consumed
**Then** all existing fields shall be present
**And** JSON schema shall be compatible
**And** no breaking changes shall be introduced

---

### AC-028: Human Validation Pass Rate

**Given** a sample of 50+ parsed regulations per format type
**When** human experts review the extracted content
**Then** >85% shall be rated as "acceptable" or better
**And** critical errors shall be <5%
**And** formatting issues shall be documented

---

## Integration Acceptance Criteria

### AC-029: End-to-End Parsing

**Given** the target HWPX file (규정집9-343(20250909).hwpx)
**When** the multi-format parser processes the entire file
**Then** all 514 TOC regulations shall be present in output
**And** >=463 regulations shall have content (90%+)
**And** coverage report shall show >90% rate

---

### AC-030: Error Handling

**Given** a parsing error during regulation extraction
**When** the error is caught
**Then** the error shall be logged with context
**And** parsing shall continue for remaining regulations
**And** failed regulations shall be marked in output

---

### AC-031: Logging and Debugging

**Given** an active parsing operation
**When** debug logging is enabled
**Then** format classifications shall be logged
**And** extraction progress shall be logged
**And** LLM fallback invocations shall be logged
**And** coverage metrics shall be logged

---

## Test Coverage Acceptance Criteria

### AC-032: Unit Test Coverage

**Given** all new components (ListRegulationExtractor, GuidelineStructureAnalyzer, etc.)
**When** unit tests are executed
**Then** code coverage shall be >85%
**And** all critical paths shall be tested
**And** edge cases shall be covered

---

### AC-033: Integration Test Coverage

**Given** the complete multi-format parsing pipeline
**When** integration tests are executed
**Then** all format types shall have integration tests
**And** multi-section aggregation shall be tested
**And** LLM fallback shall be tested (with mocked LLM)

---

## Definition of Done Checklist

**Implementation Complete:**
- [ ] All new components implemented
- [ ] All acceptance criteria (AC-001 to AC-033) verified
- [ ] Unit test coverage >85%
- [ ] Integration tests passing

**Quality Complete:**
- [ ] Coverage rate >90% on target file
- [ ] Format classification accuracy >85%
- [ ] Human validation of 50+ regulations per format
- [ ] Performance benchmarks met (<60 seconds)

**Integration Complete:**
- [ ] Backward compatibility verified
- [ ] Existing tests updated and passing
- [ ] Documentation complete
- [ ] Code review approved

**Deployment Complete:**
- [ ] Merged to main branch
- [ ] Release notes published
- [ ] User documentation updated
- [ ] Production deployment successful

---

## Success Metrics Summary

| Metric | Current (v2.1) | Target (v3.5) | Acceptance Threshold |
|--------|----------------|---------------|---------------------|
| Coverage Rate | 43.6% (224/514) | 90%+ (463+/514) | >=90% (AC-029) |
| List-Format Coverage | 0% | 90%+ of list-format | >=90% (AC-005-008) |
| Guideline-Format Coverage | 0% | 80%+ of guideline-format | >=80% (AC-009-011) |
| Format Classification Accuracy | N/A | >85% | >=85% (AC-004) |
| Parsing Time | ~45 seconds | <60 seconds | <=60s (AC-023) |
| Memory Usage | Unknown | <2GB | <2GB (AC-024) |
| Unit Test Coverage | Existing | >85% | >=85% (AC-032) |
| Human Validation Pass Rate | Unknown | >85% | >=85% (AC-028) |
