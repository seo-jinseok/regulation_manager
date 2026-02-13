# SPEC-CHUNK-001: HWPX Direct Parser Chunk Enhancement

## Metadata

| Field       | Value                     |
| ----------- | ------------------------- |
| SPEC ID     | SPEC-CHUNK-001            |
| Title       | HWPX Direct Parser Chunk Enhancement |
| Created     | 2026-02-13                |
| Status      | Implemented               |
| Priority    | High                      |
| Assigned    | expert-backend            |
| Commit      | b07d355                   |
| Completed   | 2026-02-13                |

## TAG BLOCK

```yaml
tags:
  - CHUNK-001
  - hwpx-parser
  - chunk-splitting
  - hierarchy-depth
  - rag-optimization
dependencies: []
related_spec: []
```

---

## Environment

### System Context

The HWPX Direct Parser processes Korean university regulation documents (HWPX format) for RAG (Retrieval-Augmented Generation) systems. The parser converts structured documents into hierarchical chunks for vector embedding and semantic search.

### Current State Analysis

| Metric               | HWPX Direct Parser | Legacy Parser | Gap            |
| -------------------- | ------------------ | ------------- | -------------- |
| Chunk Count          | 11,755             | 24,387        | -12,632 (-52%) |
| Max Hierarchy Depth  | 3                  | 6             | -3 levels      |
| Content Coverage     | 100%               | ~62%          | +38%           |
| Chunk Types          | 4                  | 8             | -4 types       |

### Supported Chunk Types

| Type        | HWPX Direct | Legacy | Pattern Example         |
| ----------- | ----------- | ------ | ----------------------- |
| chapter     | No          | Yes    | "제1장 총칙"            |
| section     | No          | Yes    | "제1절 목적"            |
| article     | Yes         | Yes    | "제1조"                 |
| paragraph   | Yes         | Yes    | "①", "②"                |
| item        | Yes         | Yes    | "1.", "2."              |
| subitem     | Yes         | Yes    | "가.", "나."            |
| subsection  | No          | Yes    | "제1관"                 |
| text        | Yes         | Yes    | Plain text              |

### Target File

- `src/enhance_for_rag.py` - Contains chunk splitting logic in `split_text_into_chunks()` and related functions

---

## Assumptions

### Technical Assumptions

1. Rule-based parsing without LLM calls remains the preferred approach for deterministic, fast processing
2. Python standard library regex patterns can detect Korean legal document structures
3. 100% content coverage must be maintained while improving granularity
4. Existing test suite must continue to pass after modifications

### Business Assumptions

1. Improved chunk granularity leads to better RAG retrieval accuracy
2. Deeper hierarchy representation improves context preservation in embeddings
3. Compatibility with Legacy parser output format is desirable for migration purposes

### Constraint Assumptions

1. Processing time should not increase significantly (no LLM calls)
2. Memory usage should remain within reasonable bounds
3. Output JSON schema must remain backward compatible

---

## Requirements

### REQ-001: Chapter Type Detection (Ubiquitous)

The system **shall** detect and classify chapter-level nodes from Korean regulation text.

**EARS Pattern**: Ubiquitous (Always Active)

**Pattern Rules**:
- `제\d+장` - Standard chapter (e.g., "제1장", "제2장")
- `제\s*\d+\s*장` - With whitespace (e.g., "제 1 장")
- Title extraction after chapter marker

**Acceptance Criteria**:
- WHEN text contains "제1장 총칙" THEN system creates chunk with type="chapter", display_no="제1장", title="총칙"
- WHEN text contains "제 2 장 학사" THEN system creates chunk with type="chapter", display_no="제2장", title="학사"

### REQ-002: Section Type Detection (Ubiquitous)

The system **shall** detect and classify section-level nodes from Korean regulation text.

**EARS Pattern**: Ubiquitous (Always Active)

**Pattern Rules**:
- `제\d+절` - Standard section (e.g., "제1절", "제2절")
- `제\s*\d+\s*절` - With whitespace

**Acceptance Criteria**:
- WHEN text contains "제1절 목적" THEN system creates chunk with type="section", display_no="제1절", title="목적"
- WHEN text contains "제 3 절 시행" THEN system creates chunk with type="section", display_no="제3절", title="시행"

### REQ-003: Subsection Type Detection (Optional)

**Where** feature exists, the system **shall** detect subsection-level nodes (관) from Korean regulation text.

**EARS Pattern**: Optional

**Pattern Rules**:
- `제\d+관` - Standard subsection (e.g., "제1관", "제2관")
- `제\s*\d+\s*관` - With whitespace

**Acceptance Criteria**:
- WHEN text contains "제1관 학점" THEN system creates chunk with type="subsection", display_no="제1관", title="학점"

### REQ-004: Hierarchy Depth Support (State-Driven)

**IF** a node contains nested structures **THEN** the system **shall** support hierarchy depth up to level 6.

**EARS Pattern**: State-Driven (Conditional)

**Hierarchy Order** (from root to leaf):
1. chapter (장)
2. section (절)
3. subsection (관) - optional
4. article (조)
5. paragraph (항)
6. item (호) / subitem (목)

**Current Limit**: Max depth 3 (article > paragraph > item/subitem)

**Target Limit**: Max depth 6 (chapter > section > subsection > article > paragraph > item > subitem)

**Acceptance Criteria**:
- WHEN document has 6-level hierarchy THEN system creates nested children structure with correct depth
- WHEN document has 4-level hierarchy THEN system creates nested children structure matching actual depth

### REQ-005: Chunk Count Increase (Event-Driven)

**WHEN** processing a regulation document **THEN** the system **shall** generate chunks with increased granularity targeting approximately 20,000 chunks for equivalent content.

**EARS Pattern**: Event-Driven (Trigger-Response)

**Target Metrics**:
- Current: ~11,755 chunks
- Target: ~20,000 chunks (+70% increase)
- Constraint: Maintain 100% coverage

**Strategies**:
1. Detect chapter/section markers and create separate chunks
2. Split preamble text into smaller chunks where structure permits
3. Ensure each paragraph, item, and subitem becomes an independent chunk

**Acceptance Criteria**:
- WHEN processing test regulation set THEN chunk count increases by at least 50%
- WHEN processing test regulation set THEN no content is lost (coverage remains 100%)

### REQ-006: Backward Compatibility (Unwanted Behavior)

The system **shall not** break existing API contracts or test cases.

**EARS Pattern**: Unwanted Behavior (Prohibition)

**Protected Behaviors**:
- Existing function signatures must remain unchanged
- Output JSON structure must include all existing fields
- All existing tests must pass

**Acceptance Criteria**:
- WHEN running `pytest tests/test_enhance_for_rag.py` THEN all tests pass
- WHEN running `pytest tests/ -k enhance` THEN all tests pass

### REQ-007: No LLM Dependency (Unwanted Behavior)

The system **shall not** require LLM calls for chunk detection and classification.

**EARS Pattern**: Unwanted Behavior (Prohibition)

**Constraint**:
- All chunk detection must use regex pattern matching
- No external API calls during parsing
- Python standard library only

**Acceptance Criteria**:
- WHEN processing documents THEN no network requests are made
- WHEN processing documents THEN only regex and string operations are used

---

## Specifications

### Chunk Type Detection Patterns

```python
# Proposed patterns for CHUNK_LEVEL_MAP extension
CHUNK_PATTERNS = {
    "chapter": r"^제\s*(\d+)\s*장\s*(.*)$",
    "section": r"^제\s*(\d+)\s*절\s*(.*)$",
    "subsection": r"^제\s*(\d+)\s*관\s*(.*)$",
    "article": r"^제\s*(\d+)\s*조\s*(?:\(([^)]+)\))?\s*(.*)$",
    "paragraph": r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])(.*)$",
    "item": r"^(\d+)\.\s*(.*)$",
    "subitem": r"^([가-힣])\.\s*(.*)$",
}
```

### Hierarchy Depth Implementation

```python
# Proposed depth calculation
def calculate_hierarchy_depth(node: Dict[str, Any]) -> int:
    """
    Calculate the depth of hierarchy for a node.
    Returns 1 for leaf nodes, increases by 1 for each level.
    """
    if not node.get("children"):
        return 1
    return 1 + max(calculate_hierarchy_depth(child) for child in node["children"])
```

### CHUNK_LEVEL_MAP Extension

```python
# Current
CHUNK_LEVEL_MAP = {
    "chapter": "chapter",
    "section": "section",
    "article": "article",
    "paragraph": "paragraph",
    "item": "item",
    "subitem": "subitem",
    "addendum": "addendum",
    "addendum_item": "addendum_item",
    "preamble": "preamble",
    "text": "text",
}

# Proposed extension
CHUNK_LEVEL_MAP_EXTENDED = {
    **CHUNK_LEVEL_MAP,
    "subsection": "subsection",  # NEW: 관 level
}
```

---

## Success Metrics

| Metric                  | Current    | Target     | Measurement Method         |
| ----------------------- | ---------- | ---------- | -------------------------- |
| Total Chunk Count       | 11,755     | ~20,000    | `len(flattened_chunks)`    |
| Max Hierarchy Depth     | 3          | 5-6        | `calculate_hierarchy_depth`|
| Chunk Type Coverage     | 4 types    | 7-8 types  | Type frequency analysis    |
| Test Pass Rate          | 100%       | 100%       | `pytest` results           |
| Coverage Maintenance    | 100%       | 100%       | Content comparison         |
| Processing Time         | Baseline   | <1.5x      | Benchmark comparison       |

---

## Risks and Mitigations

| Risk                          | Probability | Impact | Mitigation                          |
| ----------------------------- | ----------- | ------ | ----------------------------------- |
| False positive type detection | Medium      | Medium | Add validation tests, adjust regex  |
| Performance regression        | Low         | Medium | Profile before/after, optimize regex|
| Breaking change in output     | Low         | High   | Comprehensive test coverage         |
| Edge case handling            | Medium      | Medium | Add test cases for edge cases       |

---

## References

- Legacy Parser: `src/parsing/regulation_parser.py`
- Current Implementation: `src/enhance_for_rag.py`
- Test Suite: `tests/test_enhance_for_rag.py`
- EARS Specification: MoAI-ADK Foundation Core
