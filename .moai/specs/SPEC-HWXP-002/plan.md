# Implementation Plan: SPEC-HWXP-002

**TAG BLOCK**
```
SPEC: HWXP-002
Title: HWPX Parser Coverage Enhancement (43.6% → 90%+)
Related: spec.md, acceptance.md
Implementation Version: 3.5.0
Created: 2026-02-11
```

---

## Implementation Strategy

### Phased Approach

The implementation follows a progressive enhancement strategy with clear milestones and rollback capability:

```
Phase 1: Foundation (Priority High)
├─→ Format Classification Framework
├─→ Coverage Tracking System
└─→ Multi-Format Parser Architecture

Phase 2: List-Format Support (Priority High)
├─→ ListRegulationExtractor Implementation
├─→ List Pattern Detection
└─→ Hierarchy Preservation

Phase 3: Guideline-Format Support (Priority Medium)
├─→ GuidelineStructureAnalyzer Implementation
├─→ Provision Segmentation
└─→ Pseudo-Article Generation

Phase 4: LLM Fallback (Priority Medium)
├─→ UnstructuredRegulationAnalyzer Implementation
├─→ LLM Integration
└─→ Confidence Scoring

Phase 5: Integration & Optimization (Priority High)
├─→ Multi-Section Aggregation
├─→ Performance Optimization
└─→ Quality Validation
```

---

## Milestones by Priority

### Milestone 1: Foundation (Primary Goal)

**Objective:** Establish core multi-format parsing infrastructure

**Deliverables:**
- `FormatType` enum with classification logic
- `HWPXMultiFormatParser` skeleton with TOC integration
- `CoverageTracker` with real-time metrics
- Test infrastructure for format classification

**Success Criteria:**
- Format classification accuracy >80% on sample data
- Coverage tracking functional for all regulation types
- Integration with existing TOC extraction (section1.xml)

**Dependencies:** None (foundational work)

**Estimated Complexity:** Medium

---

### Milestone 2: List-Format Extraction (Primary Goal)

**Objective:** Enable extraction of list-format regulations without article markers

**Deliverables:**
- `ListRegulationExtractor` complete implementation
- List pattern detection (numeric, Korean alphabet, circled numbers)
- Hierarchical list structure preservation
- Conversion to article-like format for RAG compatibility

**Success Criteria:**
- Extract 90%+ of sample list-format regulations
- Preserve list hierarchy (1. → ① → 가.)
- Generate valid article structures

**Dependencies:** Milestone 1 (Format classification)

**Estimated Complexity:** Medium-High

**Technical Approach:**
```python
# List pattern detection strategy
patterns = {
    'numeric': r'^(\d+)\.\s+(.+)$',
    'korean_alphabet': r'^([가-하])\)\s+(.+)$',
    'circled_number': r'^([①-⑮])\s+(.+)$',
}

# Hierarchy extraction
def extract_nested_list(content):
    items = []
    current_level = 0
    for line in content.split('\n'):
        level = detect_indent_level(line)
        if level > current_level:
            # Child item
            items[-1]['children'].append(parse_item(line))
        else:
            items.append(parse_item(line))
    return items
```

---

### Milestone 3: Guideline-Format Analysis (Secondary Goal)

**Objective:** Extract and structure continuous prose regulations

**Deliverables:**
- `GuidelineStructureAnalyzer` implementation
- Provision segmentation by sentence/paragraph
- Key requirement extraction
- Pseudo-article generation

**Success Criteria:**
- Segment 80%+ of guideline-format regulations
- Generate meaningful provision boundaries
- Preserve content semantics

**Dependencies:** Milestone 1 (Format classification)

**Estimated Complexity:** High

**Technical Approach:**
```python
# Provision segmentation strategy
def segment_provisions(content):
    # Step 1: Split by paragraph
    paragraphs = content.split('\n\n')

    # Step 2: Detect transition words
    transitions = ['그러나', '따라서', '또한', '나아가']

    # Step 3: Segment by sentence boundaries
    # Step 4: Merge short segments (<50 chars)
    # Step 5: Split long segments (>500 chars)
    return provisions
```

---

### Milestone 4: LLM-Based Fallback (Secondary Goal)

**Objective:** Handle unstructured regulations using LLM inference

**Deliverables:**
- `UnstructuredRegulationAnalyzer` implementation
- LLM prompt engineering for structure inference
- Response parsing and validation
- Confidence scoring mechanism

**Success Criteria:**
- LLM inference success rate >75%
- Average confidence score >0.7
- Inference time <5 seconds per regulation

**Dependencies:** Milestone 1 (Format classification)

**Estimated Complexity:** High

**LLM Integration Strategy:**
```python
class UnstructuredRegulationAnalyzer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.max_retries = 2
        self.timeout = 30  # seconds

    def analyze(self, title: str, content: str):
        prompt = self._build_prompt(title, content)
        response = self.llm.generate(
            prompt,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000
        )
        regulation = self._parse_response(response)
        confidence = self._calculate_confidence(regulation)
        return regulation, confidence
```

---

### Milestone 5: Integration & Optimization (Final Goal)

**Objective:** Integrate all components and optimize performance

**Deliverables:**
- Multi-section content aggregation
- Performance optimization (parallel processing)
- Comprehensive testing suite
- Documentation updates

**Success Criteria:**
- End-to-end parsing time <60 seconds
- Coverage rate >90% on target file
- All integration tests passing
- Documentation complete

**Dependencies:** Milestones 1-4 (all components)

**Estimated Complexity:** Medium-High

---

## Technical Approach

### Architecture Design

**Component Interaction:**
```
HWPXMultiFormatParser (Coordinator)
    ├─→ TOC Extraction (section1.xml)
    │   └─→ RegulationTitleDetector
    ├─→ Content Aggregation (all sections)
    │   ├─→ section0.xml (main content)
    │   ├─→ section1.xml (TOC)
    │   └─→ section2+.xml (additional)
    ├─→ Format Classification
    │   └─→ _classify_format(content) -> FormatType
    ├─→ Format-Specific Extraction
    │   ├─→ ArticleFormatExtractor (existing)
    │   ├─→ ListRegulationExtractor (NEW)
    │   ├─→ GuidelineStructureAnalyzer (NEW)
    │   └─→ UnstructuredRegulationAnalyzer (NEW)
    ├─→ Coverage Tracking
    │   └─→ CoverageTracker
    └─→ LLM Fallback
        └─→ For low-coverage regulations
```

### Data Flow

**Parsing Pipeline:**
```
1. Load HWPX file (ZIP)
2. Extract TOC from section1.xml → 514 entries
3. For each TOC entry:
   a. Search for content in all sections
   b. Classify format type
   c. Delegate to appropriate extractor
   d. Track coverage metrics
   e. Attempt LLM fallback if coverage <20%
4. Generate coverage report
5. Output JSON with all regulations
```

### File Modifications

**New Files to Create:**
```
src/parsing/
├── format/
│   ├── __init__.py
│   ├── format_classifier.py      # Format classification logic
│   ├── list_regulation_extractor.py
│   ├── guideline_structure_analyzer.py
│   └── unstructured_regulation_analyzer.py
├── tracking/
│   ├── __init__.py
│   └── coverage_tracker.py
└── hwpx_multi_format_parser.py   # Main coordinator
```

**Files to Modify:**
```
src/parsing/
├── hwpx_direct_parser_v2.py      # Update to use multi-format
├── pipeline/hwpx_parsing_orchestrator.py  # Integrate new components
└── __init__.py                   # Export new classes

tests/parsing/
├── test_format_classifier.py
├── test_list_regulation_extractor.py
├── test_guideline_structure_analyzer.py
├── test_unstructured_regulation_analyzer.py
├── test_coverage_tracker.py
└── test_multi_format_parser.py
```

### Technology Stack

**Existing Dependencies:**
- `xml.etree.ElementTree`: XML parsing
- `zipfile`: HWPX archive extraction
- `re`: Pattern matching
- `dataclasses`: Data structures

**New Dependencies:**
- No new external dependencies required
- LLM integration uses existing `src/rag/infrastructure/llm_client.py`

---

## Risk Management

### Risk Mitigation Strategies

**Risk 1: Format Misclassification**
- Strategy: Hybrid extraction (try multiple extractors in parallel)
- Fallback: Use best-match extraction based on content yield
- Validation: Human review of 100+ classifications

**Risk 2: LLM Performance**
- Strategy: Limit LLM to <10% of regulations
- Fallback: Accept lower coverage without LLM
- Monitoring: Track LLM success rate and disable if <60%

**Risk 3: Performance Degradation**
- Strategy: Parallel processing for independent regulations
- Optimization: Cache LLM results for similar structures
- Validation: Benchmark every 50 regulations

**Risk 4: Quality Variance**
- Strategy: Content length thresholds (require >100 chars)
- Validation: Human evaluation of 50+ regulations per format
- Fallback: Flag low-quality extractions for review

### Rollback Plan

**If coverage <80% after implementation:**
1. Revert to v2.1 parser for production use
2. Debug new components in isolation
3. Incremental rollout with feature flags

**If performance >120 seconds:**
1. Disable LLM fallback temporarily
2. Optimize format classification
3. Parallel processing for section extraction

**If quality issues detected:**
1. Increase quality thresholds
2. Add manual review step
3. Flag suspicious extractions

---

## Quality Assurance

### Testing Strategy

**Unit Tests (Target: 85%+ coverage):**
- Format classification: 100+ test cases
- List extraction: 50+ test cases
- Guideline analysis: 30+ test cases
- LLM inference (mocked): 20+ test cases
- Coverage tracking: 20+ test cases

**Integration Tests:**
- Full HWPX file parsing (514 regulations)
- Multi-section aggregation
- Format delegation
- LLM fallback triggers

**Quality Tests:**
- Human evaluation of 50+ regulations per format
- Comparison with v2.1 on article-format regulations
- Performance benchmarks

### Continuous Integration

**Automated Checks:**
- Test coverage >85% for new code
- Parsing time <60 seconds for full file
- Coverage rate >90% on target file
- No regression in v2.1 functionality

**Manual Reviews:**
- Code review for new components
- Architecture review for multi-format design
- Quality review for extracted regulations

---

## Documentation Requirements

### Developer Documentation

**New Components:**
- API documentation for all new classes
- Format classification algorithm explanation
- LLM prompt engineering guide
- Performance tuning guidelines

**Updated Components:**
- v2.1 to v3.5 migration guide
- Integration examples for new parsers
- Troubleshooting guide for common issues

### User Documentation

**Release Notes:**
- Coverage improvement details (43.6% → 90%+)
- New format support announcement
- Breaking changes (if any)
- Performance metrics

**User Guide:**
- How to use multi-format parser
- How to interpret coverage reports
- How to handle low-quality extractions

---

## Success Metrics

### Quantitative Metrics

**Coverage Metrics:**
- Overall coverage: 43.6% → 90%+ (+46.4 percentage points)
- List-format coverage: 0% → 90%+
- Guideline-format coverage: 0% → 80%+
- Unstructured coverage: 0% → 50%+ (with LLM)

**Quality Metrics:**
- Format classification accuracy: >85%
- Average content length: 500 → 800+ characters
- LLM inference success rate: >75%

**Performance Metrics:**
- Parsing time: <60 seconds
- LLM inference time: <5 seconds per regulation
- Memory usage: <2GB

### Qualitative Metrics

**Maintainability:**
- Clean separation of format-specific logic
- Extensible design for new formats
- Comprehensive test coverage

**Usability:**
- Clear coverage reports
- Meaningful regulation structures
- Backward compatibility

---

## Next Steps

### Immediate Actions

1. **Review SPEC-HWXP-002** with team for feedback
2. **Create feature branch** `feature/hwxp-multi-format-parser`
3. **Set up development environment** with test data
4. **Implement Milestone 1** (Foundation)

### Implementation Order

**Week 1:**
- Day 1-2: Format classification framework
- Day 3-4: Coverage tracking system
- Day 5: Multi-format parser skeleton

**Week 2:**
- Day 1-3: List-format extraction
- Day 4-5: Unit tests for list extraction

**Week 3:**
- Day 1-3: Guideline-format analysis
- Day 4-5: Unit tests for guideline analysis

**Week 4:**
- Day 1-3: LLM fallback implementation
- Day 4-5: Integration testing

**Week 5:**
- Day 1-3: Performance optimization
- Day 4-5: Quality validation and documentation

### Validation Criteria

**Before merging:**
- [ ] All unit tests passing (>85% coverage)
- [ ] Integration tests passing (full HWPX file)
- [ ] Coverage rate >90% on target file
- [ ] Parsing time <60 seconds
- [ ] Human evaluation of 50+ regulations
- [ ] Documentation complete
- [ ] Code review approved

**After merging:**
- [ ] Monitor coverage in production
- [ ] Collect user feedback
- [ ] Track performance metrics
- [ ] Plan next iteration if needed
