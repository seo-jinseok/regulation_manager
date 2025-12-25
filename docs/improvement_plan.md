# Improvement Plan

This plan addresses the full set of reviewed improvements. Each step includes
explicit verification before moving on.

## Step 1: Align schema docs with actual output
- Update `SCHEMA_REFERENCE.md` to reflect `keywords` as weighted objects and add
  missing RAG node fields (`embedding_text`, `chunk_level`, `is_searchable`,
  `token_count`, `effective_date`) plus schema version wording.
- Verification:
  - Cross-check `SCHEMA_REFERENCE.md` against `src/enhance_for_rag.py`.
  - Spot-check a real output file in `data/output/` for matching field shapes.

## Step 2: Stabilize incremental sync logic
- If the vector store is empty but state exists, force a full sync (or repair
  only missing rule codes).
- Verification:
  - Unit tests cover the new behavior.
  - Manual read-through confirms no regression for normal incremental sync.

## Step 3: Make content hashing resilient
- Exclude volatile metadata (e.g., `scan_date`, `generated_at`) from the hash
  used by incremental sync to reduce false "modified" detections.
- Verification:
  - Unit tests cover hash stability.
  - Compare hashes before/after for the same content with metadata-only changes.

## Step 4: Harden Chroma deletion/query handling
- Avoid fragile `include=[]` usage; use the most compatible call pattern.
- Verification:
  - Unit tests for delete path.
  - Confirm no behavior change for normal store operations.

## Step 5: Fix filter precedence in search
- Respect explicit `SearchFilter.status` over `include_abolished` defaults.
- Verification:
  - Unit tests cover filter precedence.

## Step 6: Improve CLI search display
- Show regulation title and rule code separately in CLI output.
- Verification:
  - Manual inspection of CLI output formatting.

## Step 7: Strengthen hybrid search scoring
- Incorporate `keywords` (weighted) into scoring when present.
- Verification:
  - Unit tests validate scoring changes without breaking ranking stability.

## Step 8: Packaging and dependency cleanup
- Replace placeholder description; align dependencies between
  `pyproject.toml` and `requirements.txt`.
- Verification:
  - `pyproject.toml` and `requirements.txt` are consistent.

## Step 9: Make cache writes atomic
- Write cache JSON to a temp file, then replace atomically to avoid corruption.
- Verification:
  - Unit tests for cache write path.
  - Manual read-through for correctness.

