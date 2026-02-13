# Expert Backend Agent Memory

## HWPX Parsing System Implementation (2026-02-11)

### Project-Specific Patterns

**HWPX File Format:**
- HWPX files are ZIP archives containing XML documents
- Namespace: `http://www.hancom.co.kr/hwpml/2011/section`
- Content location: `Contents/section*.xml`
- section1.xml typically contains Table of Contents (TOC)
- section0.xml contains main body content

**Korean Regulation Structure:**
- Hierarchy: 편(Part) > 장(Chapter) > 절(Section) > 조(Article) > 항(Item) > 호(Subitem)
- Article numbering: 제N조, 제N조의M (variants)
- Paragraph markers: ①, ②, ③... (circled numbers)
- Item markers: 1., 2., 3...
- Subitem markers: 가.), 나.), 다)... (Korean alphabet)

**Text Artifacts to Handle:**
- Page headers: `\d+[—－]\d+[—－]\d+[~～]` pattern
- Unicode filler characters: U+F0800-U+F0FFF
- Duplicate titles from page contamination
- Horizontal rules: `[─＿]+`

### Key Design Decisions

**TOC-First Parsing Strategy:**
- Parse section1.xml (TOC) first to get complete regulation list
- Then parse section0.xml (body) to match content
- Create regulation entries even if no articles found (preamble-only)
- This ensures 100% coverage of TOC entries

**Context-Aware Title Detection:**
- `RegulationTitleDetector`: General-purpose, accepts standalone keywords
- `HWPXDirectParser._is_regulation_title`: Context-aware, filters short standalone keywords
- This two-tier approach balances reusability with context specificity

**Title Detection Rules:**
- Must end with regulation keyword (규정, 요령, 지침, 세칙, etc.)
- Skip patterns: 제N조 markers, TOC elements, numbered lists
- False positive patterns: "이 규정", "규정 관리", etc.
- Length validation: 4-200 characters (with special handling for 2-3 char keywords)

### Testing Strategy

**Unit Test Coverage Targets:**
- Core modules: 85%+ coverage
- Integration tests: Full HWPX file parsing
- Parametrized tests for pattern variations

**Test Organization:**
```
tests/parsing/
├── test_text_normalizer.py          (20 tests)
├── test_regulation_title_detector.py (32 tests)
├── test_completeness_checker.py      (18 tests)
├── test_hwpx_direct_parser.py        (31 tests)
└── test_regulation_article_extractor.py (27 tests)
```

### Common Pitfalls

1. **Encoding Issues:** HWPX files may use UTF-8 or CP949 encoding
   - Solution: Try UTF-8 first, fallback to CP949

2. **Standalone Short Keywords:** "지침", "규정" alone are ambiguous
   - Solution: Filter in parser context, accept in detector

3. **TOC vs Body Mismatches:** TOC may contain entries not in body
   - Solution: Create empty regulation entries for missing items

4. **Article Format Variations:** "제1조", "## 제1조", "제1조의2"
   - Solution: Multiple regex patterns with fallback

### Module Dependencies

```
hwpx_direct_parser.py
├── core/text_normalizer.py
├── detectors/regulation_title_detector.py
├── validators/completeness_checker.py
└── regulation_article_extractor.py
```

### File Paths Reference

- Parser: `/Users/truestone/Dropbox/repo/University/regulation_manager/src/parsing/hwpx_direct_parser.py`
- Tests: `/Users/truestone/Dropbox/repo/University/regulation_manager/tests/parsing/`
- Docs: `/Users/truestone/Dropbox/repo/University/regulation_manager/.moai/docs/backend-architecture-SPEC-HWXP-001.md`
