# Plan: Perfect JSON Extraction for DB & RAG

## Phase 1: Enhanced Core Extraction Logic
- [x] Task: Refine Title Detection Heuristics a5d7968
    -   **Description**: Improve the regex and logic in `src/preprocessor.py` (or equivalent) to strictly distinguish between actual regulation titles and noise (e.g., addenda start lines). Implement a validation step that checks for standard regulation title patterns.
- [x] Task: Implement HTML Table Extraction for Appendices e8c063b
    -   **Description**: Update the parser to identify `<table>` tags in the HWP-converted HTML. Extract these tables as raw HTML strings and store them in the `attached_files` or a new `appendices` field in the JSON structure.
- [x] Task: Deep Hierarchy Parsing 04ce5ad
    -   **Description**: Verify and fix the recursive parsing logic to ensure it captures all levels (Article down to Sub-item) without flattening or losing data. Add specific tests for deeply nested structures.

## Phase 2: Intelligent Error Handling & Metadata
- [x] Task: Integrate LLM Auto-Repair 62d9ce5
    -   **Description**: Create a new module `src/repair.py` that interacts with the configured LLM. When the regex parser fails or produces low-confidence results, send the snippet to the LLM to identify the structure.
- [x] Task: Implement Confidence Scoring 62d9ce5
    -   **Description**: Add a `confidence_score` field to each node in the JSON. Heuristic-based parsing gets high scores; ambiguous chunks get low scores.
- [x] Task: Extract Cross-References 62d9ce5
    -   **Description**: Implement a regex-based extractor to find patterns like "제X조" or "제X항" within the text. Store these as a `references` list within each node, pointing to the target (if resolvable).

## Phase 3: Validation and Verification
- [ ] Task: Update Schema Definition
    -   **Description**: Update `SCHEMA_REFERENCE.md` and any schema validation scripts to include the new fields (`html_content` for tables, `confidence_score`, `references`).
- [ ] Task: Run Full Regression Test
    -   **Description**: Process the reference HWP files and verify that the output JSON matches the "perfect" expectation (no garbage titles, all tables present).
- [ ] Task: Conductor - User Manual Verification 'Perfect Extraction' (Protocol in workflow.md)

