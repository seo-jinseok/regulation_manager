# Gap Analysis Report - SPEC-RAG-QUALITY-005

**Analysis Date**: 2026-02-17
**Analyst**: manager-ddd (DDD ANALYZE Phase)
**Source Evaluation**: rag_quality_local_20260215

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Staff Queries Tested | 25 |
| Failed Queries | 25 (100%) |
| Root Cause | API compatibility issue (use_rerank parameter) |
| Staff Document Coverage | 5.1% (19/372 docs) |
| Missing Topics | 급여, 사무용품, 입찰, 물품, 휴가 (direct docs) |

---

## 1. Staff Query Failures

### 1.1 Failure Overview

All 25 staff queries failed with the same technical error:
```
Error: SearchUseCase.ask() got an unexpected keyword argument 'use_rerank'
```

### 1.2 Query Categories Analyzed

| Category | Count | Example Queries |
|----------|-------|-----------------|
| leave (휴가) | 5 | 연차 휴가 신청 절차, 병가 사용 기준, 경조사 휴가 기간 |
| salary (급여) | 5 | 급여 지급 일자, 성과상여금 지급 기준, 초과근무 수당 |
| personnel (인사) | 5 | 전보 발령 기준, 승진 심사 항목, 징계 처분 종류 |
| training (연수) | 3 | 연수 참여 신청, 보안 교육 이수, 성희롱 예방 교육 |
| admin (행정) | 7 | 사무용품 구매 절차, 시설 사용 신청, 출장비 정산 |

### 1.3 Root Cause Analysis

**Primary Issue**: The `SearchUseCase.ask()` method does not accept `use_rerank` as a parameter.

**Evidence from Code Review**:
- `/src/rag/application/search_usecase.py` line 2260:
  ```python
  def ask(
      self,
      question: str,
      filter: Optional[SearchFilter] = None,
      top_k: int = 5,
      include_abolished: bool = False,
      audience_override: Optional["Audience"] = None,
      history_text: Optional[str] = None,
      search_query: Optional[str] = None,
      debug: bool = False,
      custom_prompt: Optional[str] = None,
  ) -> Answer:
  ```

- No `use_rerank` parameter exists in the `ask()` method signature
- The `use_rerank` parameter is only valid during `SearchUseCase.__init__()`, not in `ask()`

**Files with incorrect usage** (passing `use_rerank` to `ask()`):
- `/src/rag/domain/evaluation/parallel_evaluator.py` - line 366

### 1.4 Secondary Issue: Context Retrieval

The mock evaluation still showed high `contextual_recall` (0.87) even with errors, suggesting:
- The underlying retrieval system is functional
- The issue is purely at the API call level, not retrieval quality

---

## 2. Regulation Coverage Analysis

### 2.1 Current Document Coverage

| Topic | Documents Found | Coverage Status | Notes |
|-------|-----------------|-----------------|-------|
| 복무 | 1 (교직원복무규정) | Partial | Covers general service rules, may lack detail |
| 휴가 | 0 (via 복무) | Missing | No dedicated leave regulation document |
| 급여 | 0 (via 보수/수당) | Missing | No dedicated salary regulation |
| 연수 | 1 (한국어연수과정) | Partial | Only Korean language training covered |
| 사무용품 | 0 | Missing | No office supplies regulation |
| 시설 | 2 | Good | 시설관리규정, 연구시설·장비비통합관리규정 |
| 인사 | 4 | Good | 교원인사규정, 직원인사규정, 강사인사규정, JA교원인사규정 |
| 구매 | 3 | Good | 구매업무규정, 산학협력단구매및자산관리규정 |
| 차량 | 1 | Good | 차량관리규정 |
| 물품 | 0 (via 구매) | Partial | Covered indirectly via purchasing regulations |
| 입찰 | 0 | Missing | No bidding/tender regulation |

### 2.2 Coverage Statistics

- **Total active regulations**: 372
- **Staff-related regulations**: 19 (5.1%)
- **Missing critical topics**: 사무용품, 입찰, dedicated 급여/휴가

### 2.3 TOC Analysis (24 staff-related regulations found)

The TOC shows more staff regulations than the docs array suggests:
- Some may be in different JSON files
- Some may have been merged into compound regulations
- Recommended to verify document ingestion completeness

---

## 3. Citation Extraction Analysis

### 3.1 Current Implementation

**File**: `/src/rag/domain/citation/article_number_extractor.py`

**Supported Patterns**:
```python
PATTERNS = [
    (ArticleType.SUB_ARTICLE, r"제(\d+)조의(\d+)"),  # 제10조의2
    (ArticleType.CHAPTER, r"제(\d+)장"),            # 제2장
    (ArticleType.TABLE, r"별표(\d+)"),              # 별표1
    (ArticleType.FORM, r"서식(\d+)"),               # 서식1
    (ArticleType.ARTICLE, r"제(\d+)조"),            # 제26조
]
```

### 3.2 Strengths
- Comprehensive coverage of standard Korean regulation citation formats
- Support for sub-articles (제N조의M), tables (별표), forms (서식)
- Proper regex compilation for performance

### 3.3 Weaknesses Identified

1. **Missing Pattern: 항/호 references**
   - Current: Only extracts article-level (제N조)
   - Missing: Paragraph (제N항) and item (제N호) level citations
   - Impact: Less precise citation for detailed queries

2. **No Clause Reference Support**
   - Missing patterns for 전단, 후단, but/다만 clauses
   - Important for conditional provisions

3. **Regulation Name Extraction Dependency**
   - `CitationEnhancer` relies on `chunk.parent_path[0]` for regulation name
   - If parent_path is malformed, citations will be incomplete

4. **No Cross-Reference Detection**
   - Does not detect references like "제N조에 따라" within text
   - Only extracts from chunk titles, not content

### 3.4 Citation Formatting

**File**: `/src/rag/domain/citation/citation_enhancer.py`

**Current Format**: `「{regulation}」 {article_number}`
- Example: `「교원인사규정」 제26조`

**Quality Issues**:
- No validation that article number actually exists in regulation
- No handling for multiple regulations with same article numbers
- Confidence scoring not utilized in output formatting

---

## 4. Recommendations

### Priority 1: Fix API Compatibility (CRITICAL)

**Issue**: All staff queries failing due to `use_rerank` parameter error

**Action**:
1. Remove `use_rerank` parameter from `parallel_evaluator.py` line 366
2. Rerank is already configured at `SearchUseCase` initialization
3. Verify all callers of `ask()` method

**Files to Modify**:
- `/src/rag/domain/evaluation/parallel_evaluator.py` - line 366

### Priority 2: Add Missing Staff Regulations (HIGH)

**Issue**: Only 5.1% of documents are staff-related

**Recommended Additions**:
1. **휴가 관련**: 연차휴가규정, 병가규정, 경조사휴가규정
2. **급여 관련**: 직원급여규정, 수당지급기준세칙
3. **사무용품**: 사무용품구매및관리규정
4. **입찰**: 입찰참가규정, 계약관리규정
5. **연수**: 직원연수규정, 직무교육규정

**Source**: University administration office, regulation archives

### Priority 3: Enhance Citation Extraction (MEDIUM)

**Improvements**:
1. Add paragraph (항) and item (호) level patterns:
   ```python
   (ArticleType.PARAGRAPH, r"제(\d+)항"),
   (ArticleType.ITEM, r"제(\d+)호"),
   ```

2. Add cross-reference detection in content:
   - Pattern: `r"(제\d+조)(의\d+)?(제\d+항)?(제\d+호)?에?\s*(따르|의거|규정)"`

3. Add citation validation:
   - Verify article exists in source regulation
   - Flag low-confidence citations

**Files to Modify**:
- `/src/rag/domain/citation/article_number_extractor.py`
- `/src/rag/domain/citation/citation_enhancer.py`

### Priority 4: Edge Case Testing (MEDIUM)

**Recommended Test Categories**:
1. **Typo tolerance**: 연차휴가 -> 연차휴가, 연차회가
2. **Vague queries**: "휴가 관련해서요", "급여 문의"
3. **Multi-topic queries**: "휴가와 급여에 대해"
4. **Regulation-specific**: "제26조에 따르면" (no context)

### Priority 5: Metadata Enhancement (LOW)

**Improve chunk metadata**:
- Add `audience_tags` for staff-specific content
- Add `topic_category` for easier filtering
- Add `last_updated` for currency validation

---

## 5. Files to Modify

### Phase 1: Critical Fix

| File | Change | Risk |
|------|--------|------|
| `/src/rag/domain/evaluation/parallel_evaluator.py` | Remove `use_rerank` from `ask()` call | Low |

### Phase 2: Staff Regulation Enhancement

| File | Change | Risk |
|------|--------|------|
| `data/input/` | Add missing staff regulation documents | Low |
| `/src/rag/infrastructure/indexing/document_processor.py` | Process new documents | Medium |

### Phase 3: Citation Improvement

| File | Change | Risk |
|------|--------|------|
| `/src/rag/domain/citation/article_number_extractor.py` | Add 항/호 patterns | Medium |
| `/src/rag/domain/citation/citation_enhancer.py` | Add validation logic | Medium |
| `/src/rag/application/search_usecase.py` | Integrate enhanced citations | Medium |

---

## 6. Success Metrics Validation

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Staff Pass Rate | 0% (API error) | 80%+ | Fix API first |
| Staff Completeness | 0.760 (mock) | 0.85+ | +0.09 |
| Citation Score | 0.850 | 0.90+ | +0.05 |
| Staff Doc Coverage | 5.1% | 15%+ | +10% |

---

## 7. Conclusion

### Immediate Action Required
1. Fix the `use_rerank` parameter bug in `parallel_evaluator.py`
2. Re-run staff evaluation to get accurate baseline

### Medium-Term Improvements
1. Source and add 20+ staff-related regulation documents
2. Enhance citation extraction with paragraph/item support
3. Add edge case test scenarios

### Long-Term Enhancements
1. Implement citation validation
2. Add audience-specific metadata
3. Continuous monitoring dashboard

---

**Next Phase**: PRESERVE - Create characterization tests before implementing fixes
