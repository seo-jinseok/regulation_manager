# Implementation Plan: SPEC-CHUNK-001

## Overview

This plan outlines the implementation approach for enhancing the HWPX Direct Parser's chunk splitting capabilities to improve RAG retrieval accuracy through finer granularity and deeper hierarchy support.

---

## Milestones

### Milestone 1: Pattern Definition and Validation

**Priority**: High (Primary Goal)

**Objectives**:
- Define regex patterns for chapter, section, and subsection detection
- Create unit tests for pattern matching
- Validate patterns against sample regulation documents

**Tasks**:
1. Add `CHUNK_PATTERNS` constant with new pattern definitions
2. Create `detect_chunk_type()` function for automatic type detection
3. Write unit tests for pattern matching in `tests/test_enhance_for_rag.py`
4. Test patterns against existing regulation sample files

**Deliverables**:
- New regex patterns in `src/enhance_for_rag.py`
- Unit tests for pattern detection
- Pattern validation report

### Milestone 2: Chapter and Section Detection

**Priority**: High (Primary Goal)

**Objectives**:
- Implement chapter (장) level detection
- Implement section (절) level detection
- Integrate with existing chunk splitting pipeline

**Tasks**:
1. Create `detect_chapter()` function
2. Create `detect_section()` function
3. Modify `split_text_into_chunks()` to check for chapter/section markers
4. Add chapter/section nodes to output structure
5. Update `CHUNK_LEVEL_MAP` with new types

**Deliverables**:
- Chapter and section detection functions
- Modified `split_text_into_chunks()` function
- Updated `CHUNK_LEVEL_MAP`

### Milestone 3: Subsection Detection (Optional)

**Priority**: Medium (Secondary Goal)

**Objectives**:
- Implement subsection (관) level detection
- Add subsection as optional hierarchy level

**Tasks**:
1. Create `detect_subsection()` function
2. Integrate subsection detection into chunk splitting
3. Add tests for subsection detection

**Deliverables**:
- Subsection detection function
- Extended hierarchy support

### Milestone 4: Hierarchy Depth Support

**Priority**: High (Primary Goal)

**Objectives**:
- Support hierarchy depth up to 6 levels
- Implement proper parent-child relationships
- Ensure correct depth calculation

**Tasks**:
1. Implement `calculate_hierarchy_depth()` utility function
2. Modify `convert_article_to_children_structure()` for deeper nesting
3. Update `enhance_node_for_hwpx()` for depth-aware processing
4. Add depth tracking to node metadata

**Deliverables**:
- Depth calculation utility
- Deep hierarchy support in chunk structure
- Depth metadata in output

### Milestone 5: Chunk Count Optimization

**Priority**: Medium (Secondary Goal)

**Objectives**:
- Increase chunk count from ~11,755 to ~20,000
- Maintain 100% content coverage
- Ensure no content loss

**Tasks**:
1. Analyze current chunk distribution
2. Identify opportunities for finer splitting
3. Implement additional splitting rules for edge cases
4. Verify coverage with comparison tests

**Deliverables**:
- Optimized chunk splitting logic
- Coverage verification tests
- Before/after comparison report

### Milestone 6: Test Coverage and Validation

**Priority**: High (Primary Goal)

**Objectives**:
- Ensure all new functionality has test coverage
- Maintain 100% backward compatibility
- Pass all existing tests

**Tasks**:
1. Add unit tests for new chunk types
2. Add integration tests for hierarchy depth
3. Add regression tests for existing functionality
4. Run full test suite and fix any failures

**Deliverables**:
- Complete test coverage for new features
- All tests passing
- Test coverage report

---

## Technical Approach

### Architecture Changes

```
Current Flow:
  text -> split_text_into_chunks() -> [paragraph, item, subitem] chunks

Enhanced Flow:
  text -> detect_structure_level() -> split_text_into_chunks() -> [chapter, section, subsection, article, paragraph, item, subitem] chunks
```

### Key Functions to Modify

| Function                        | Change Type | Description                           |
| ------------------------------- | ----------- | ------------------------------------- |
| `split_text_into_chunks()`      | Enhance     | Add chapter/section detection         |
| `extract_items_from_text()`     | Minor       | Support deeper nesting                |
| `convert_article_to_children_structure()` | Enhance | Handle multi-level hierarchy          |
| `enhance_node_for_hwpx()`       | Minor       | Track depth information               |
| `determine_chunk_level()`       | Extend      | Add new chunk types                   |

### New Functions to Add

| Function                        | Purpose                                    |
| ------------------------------- | ------------------------------------------ |
| `detect_chunk_type()`           | Auto-detect chunk type from text           |
| `detect_chapter()`              | Detect chapter (장) markers                 |
| `detect_section()`              | Detect section (절) markers                 |
| `detect_subsection()`           | Detect subsection (관) markers              |
| `calculate_hierarchy_depth()`   | Calculate max depth of node tree           |
| `split_text_by_structure()`     | Split text at all structural levels        |

---

## File Modification Plan

### Primary File: `src/enhance_for_rag.py`

**Changes**:
1. Add `CHUNK_PATTERNS` constant (lines 75-85)
2. Add detection functions (lines 86-150)
3. Modify `split_text_into_chunks()` (lines 87-156)
4. Extend `CHUNK_LEVEL_MAP` (lines 59-71)
5. Add utility functions (lines 253-280)

### Secondary File: `tests/test_enhance_for_rag.py`

**Changes**:
1. Add test class `TestChapterDetection`
2. Add test class `TestSectionDetection`
3. Add test class `TestSubsectionDetection`
4. Add test class `TestHierarchyDepth`
5. Add test class `TestChunkCountIncrease`

---

## Constraints and Guidelines

### Performance Constraints

- No LLM calls: All detection must be regex-based
- Processing time: Should not exceed 1.5x current baseline
- Memory: Should remain within current bounds

### Code Quality Constraints

- Follow existing code style (ruff formatting)
- Add docstrings to all new functions
- Type hints for all function parameters and returns

### Testing Constraints

- Maintain 85%+ test coverage for modified code
- All existing tests must pass
- Add negative test cases (invalid patterns)

---

## Risk Mitigation Strategy

### Pattern False Positives

**Mitigation**: Use strict regex patterns with word boundaries
**Validation**: Test against diverse regulation document samples

### Breaking Changes

**Mitigation**: Use feature flag for new chunk types
**Validation**: Run full regression test suite

### Performance Regression

**Mitigation**: Profile code before and after changes
**Validation**: Benchmark with production-size documents

---

## Dependencies

### Internal Dependencies

- `src/enhance_for_rag.py` - Core implementation file
- `tests/test_enhance_for_rag.py` - Test file
- Sample regulation documents for testing

### External Dependencies

- Python 3.10+ (standard library only)
- pytest for testing
- No new external dependencies

---

## Rollout Plan

### Phase 1: Development

1. Implement pattern definitions
2. Add detection functions
3. Modify chunk splitting logic
4. Add unit tests

### Phase 2: Validation

1. Run full test suite
2. Test with sample documents
3. Compare output with Legacy parser
4. Performance benchmarking

### Phase 3: Deployment

1. Merge to main branch
2. Update documentation
3. Monitor for issues

---

## Success Criteria

- [ ] Chapter detection working with 95%+ accuracy
- [ ] Section detection working with 95%+ accuracy
- [ ] Hierarchy depth support up to level 6
- [ ] Chunk count increased by 50%+
- [ ] All existing tests passing
- [ ] No regression in content coverage (100%)
- [ ] Processing time within 1.5x baseline
