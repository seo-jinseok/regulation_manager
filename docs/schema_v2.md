# JSON Schema V2 Documentation

This document describes the V2 JSON schema used for Regulation data, designed for robust database integration.

## Node Structure

All hierarchical nodes (`regulation` (root), `chapter`, `article`, `paragraph`, `item`, `subitem`) share a common structure:

```json
{
  "id": "UUID (v4 string)",
  "type": "article",         // Node type: chapter, section, article, paragraph, item, subitem
  "display_no": "제29조의2",  // Original display string
  "sort_no": {               // Integer keys for sorting
    "main": 29,              // Primary number
    "sub": 2                 // Branch/Sub number (default 0)
  },
  "title": "직원의 임용",     // Title (optional)
  "text": "직원은...",        // Text content (optional)
  "children": []             // List of child nodes
}
```

## Entity Details

### 1. Regulation (Root)
Top-level object representing a file/document.
- `metadata`: Contains `rule_code`, `page_range`, `scan_date`.
- `content`: List of top-level children (Chapters or Articles).

### 2. Article (`article`)
- **display_no**: e.g., "제29조", "제29조의2".
- **sort_no**:
  - `main`: The article number.
  - `sub`: The branch number (e.g., 2 for "의2"). 0 if none.

### 3. Paragraph (`paragraph`)
Represents generic paragraphs distinct by circled numbers (①, ②...).
- **display_no**: "①", "②".
- **sort_no**:
  - `main`: 1, 2, ... mapped from symbol.
  - `sub`: 0.

### 4. Item (`item`)
Represents items listed with numbers (1., 2....).
- **display_no**: "1.", "2.".
- **sort_no**:
  - `main`: 1, 2...
  - `sub`: 0.
- **Note**: Items are always nested under a Paragraph. If a source text has an Item directly under an Article, it is wrapped in an *implicit* Paragraph (empty display_no).

### 5. SubItem (`subitem`)
Represents sub-items listed with Hangul chars (가., 나....).
- **display_no**: "가.", "나.".
- **sort_no**:
  - `main`: 1 ('가'), 2 ('나')...
  - `sub`: 0.

## Database Mapping Recommendation

When migrating to a relational database (SQL), the `sort_no` fields allow composite sorting:
`ORDER BY sort_no_main ASC, sort_no_sub ASC`
