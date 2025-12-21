# Track: Perfect JSON Extraction for DB & RAG

## Overview
This track aims to upgrade the current extraction pipeline to achieve "perfect" fidelity from HWP source files. The goal is to produce a structured JSON output that is 100% accurate, capturing not just the main body of regulations but also complex addenda (부칙) and appendices (별표/별지). The output must be ready for database insertion and optimized for future Hybrid RAG applications.

## Functional Requirements
*   **High-Fidelity Extraction**: Eliminate false positive title detections and ensure no valid regulations are missed.
*   **Complex Structure Handling**: Correctly parse and structure tables within appendices (별표/별지).
*   **Hierarchy Preservation**: Maintain the full depth of the regulation structure (Regulation -> Chapter -> Article -> Paragraph -> Item -> Sub-item).
*   **Table Representation**: Store tables as raw HTML strings within the JSON to preserve exact visual layout and content fidelity, facilitating accurate rendering and reference.
*   **Error Handling**: Implement an auto-repair mechanism using LLM heuristics for ambiguous content, supplemented by a confidence scoring system to flag low-confidence extractions for review.
*   **RAG Optimization**: Ensure the JSON structure is granular enough for effective chunking and explicitly extracts cross-references between regulations to support future Graph/Hybrid RAG implementations.

## Non-Functional Requirements
*   **Accuracy**: Target 100% concordance with the source HWP text.
*   **Schema**: The JSON output must strictly follow a defined schema that supports relational mapping (though the primary target mentioned is RAG-readiness).

## Out of Scope
*   Actual database insertion (this track focuses on generating the *ready-to-insert* JSON).
*   Generation of vector embeddings (this is a downstream task).

## Acceptance Criteria
*   The system correctly identifies and extracts all valid regulation titles without garbage.
*   Tables in appendices are accurately captured as HTML strings.
*   The JSON output passes strict schema validation.
*   Cross-references are identified and structured in the output.

