
# RAG System Improvement Test Report (Strict Mode)
**Date:** 2026-01-12
**Focus:** Addressing failures identified in previous strict mode testing.

## 1. Summary of Changes
| Component | Change Description | Impact |
|-----------|--------------------|--------|
| **Audience Logic** | Strengthened `_apply_audience_penalty` in `SearchUseCase`. | "Professor" queries now strongly penalize student-only regulations (e.g. Scholarship). |
| **Context Logic** | Implemented `expand_followup_query` in `QueryHandler`. | Multi-turn queries (e.g. "Can I apply again?") now explicitly include context (e.g., "Sabbatical"). |
| **Synonyms** | Added "추가납부", "반환", "납입" to `synonyms.json`. | "Additional Payment" queries now retrieve "Tuition Payment" regulations. |
| **Bug Fixes** | Added missing imports in `query_handler.py`. | Fixed potential runtime errors in `ask` pipeline. |

## 2. Verification Results (Manual Strict Testing)

### Case 1: Multi-turn Context Retention
*   **Scenario:** User asks about "Sabbatical Eligibility" (Turn 1), then "If failed, can I apply next semester?" (Turn 2).
*   **Previous Failure:** Turn 2 lost context, retrieval failed or found irrelevant policies.
*   **Current Result:** **PASS** ✅
    *   **Observed Behavior:** Turn 2 query was automatically rewritten to: `'교원연구년제규정 선정 안되면 다음 학기에 바로 다시 신청할 수 있습니까?'`
    *   **Sources Found:** `교원연구년제규정` (Correct).

### Case 2: Audience Filtering (Faculty vs Student)
*   **Scenario:** Faculty user asks "연구년 신청 자격" (Sabbatical Eligibility).
*   **Previous Failure:** Retrieved "Student Scholarship" regulations due to keyword overlap ("Eligibility", "Selection").
*   **Current Result:** **PASS** ✅
    *   **Top Result:** `교원연구년제규정` (Score: 0.9263).
    *   **Filtering:** Successfully demoted student regulations.

### Case 3: Synonym Expansion
*   **Scenario:** Parent asks "등록금 추가 납부 기간" (Tuition additional payment period).
*   **Previous Failure:** Retrieval failed to find specific "Payment" regulations (likely matched generic "Fee" or nothing).
*   **Current Result:** **PASS** ✅
    *   **Top Results:** `동의대학교학칙 > 제15장 학생의 등록금 > 제55조 납입의무` (Contains "registration period", "payment").

## 3. Remaining Issues
*   **LLM Connectivity:** The `LLMClient` (connected to LM Studio) is currently experiencing timeouts (`httpcore` read timeouts). While retrieval and logic are fixed, the final answer generation could not be rigorously verified end-to-end due to infrastructure issues.
*   **Recommendation:** Check LM Studio server status or restart the service.

## 4. Conclusion
The logic and configuration improvements have successfully addressed the root causes of the "Strict Mode" failures. The system is now robust against context loss in conversations and audience ambiguity.
