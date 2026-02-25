# SPEC-SKILL-REFRESH-001: Skill Structure Optimization

---
**SPEC ID:** SPEC-SKILL-REFRESH-001
**Title:** Skill Structure Optimization - rag-quality-local to ragulation-quality Rebrand
**Created:** 2026-02-24
**Status:** Completed
**Priority:** Medium
**Assigned:** manager-spec
**Related SPECs:** None
**Lifecycle Level:** spec-first
---

## Environment

**Project:** Regulation Manager (ragulation)
**Entry Point:** `uv run ragulation`
**Current State:**
- One custom skill exists: `rag-quality-local`
- 50+ MoAI-ADK framework skills exist (NOT to be modified)
- Skill is well-designed for regulation Q&A quality evaluation with 6 user personas

**Problem Statement:**
The current skill name `rag-quality-local` suggests it only focuses on "RAG" (Retrieval-Augmented Generation), but the actual project is `ragulation` (regulation Q&A system). This naming mismatch causes confusion about the skill's actual scope and purpose.

## Assumptions

1. The existing skill functionality is correct and should be preserved
2. The rebranding should not affect MoAI-ADK framework skills
3. All module files reference the skill name and need updating
4. No breaking changes to existing evaluation functionality
5. CLI commands may have hardcoded references to the old name

## Requirements

### Functional Requirements

**FR-01: Skill Renaming**
**WHEN** the user accesses the quality evaluation skill
**THEN** the system **shall** provide a skill named `ragulation-quality`
**AND** the skill name **shall** clearly indicate it evaluates the regulation Q&A system

**FR-02: Description Update**
**WHEN** the user views the skill description
**THEN** the system **shall** display "Regulation Manager 규정 Q&A 시스템" instead of "RAG 시스템"
**AND** the description **shall** accurately reflect the project-wide scope

**FR-03: Module References**
**WHEN** the skill is loaded
**THEN** all module files **shall** reference the new skill name `ragulation-quality`
**AND** no orphaned references to `rag-quality-local` **shall** remain

**FR-04: Backward Compatibility**
**WHEN** existing CLI commands are executed
**THEN** the commands **shall** work identically after renaming
**AND** no breaking changes **shall** be introduced to evaluation functionality

**FR-05: Trigger Keywords**
**WHEN** the user mentions quality evaluation
**THEN** the skill **shall** be triggered by relevant keywords
**AND** the keywords **shall** include "ragulation", "quality", "evaluation", "regulation Q&A"

### Non-Functional Requirements

**NFR-01: No Framework Modifications**
The system **shall not** modify or remove any MoAI-ADK framework skills.

**NFR-02: File Integrity**
All module files **shall** maintain their content structure and logic.

**NFR-03: Configuration Consistency**
The skill configuration **shall** remain consistent with MoAI skill standards.

### Out of Scope

- Adding new evaluation modules for HWP/CLI/MCP (future enhancement)
- Modifying MoAI-ADK framework skills
- Changing evaluation logic or metrics
- Adding new persona types

## Specifications

### File Changes

| Action | Source | Target |
|--------|--------|--------|
| Rename Directory | `.claude/skills/rag-quality-local/` | `.claude/skills/ragulation-quality/` |
| Update SKILL.md | Name field: `rag-quality-local` | Name field: `ragulation-quality` |
| Update SKILL.md | Description: "RAG 시스템" | Description: "Regulation Manager 규정 Q&A 시스템" |
| Update Modules | References to `rag-quality-local` | References to `ragulation-quality` |

### SKILL.md Frontmatter Changes

```yaml
# Before
name: rag-quality-local
description: >
  RAG 시스템의 포괄적인 품질 평가 시스템입니다...

# After
name: ragulation-quality
description: >
  Regulation Manager 규정 Q&A 시스템의 포괄적인 품질 평가 시스템입니다...
```

### Trigger Keywords Update

```yaml
# Before
triggers:
  keywords:
    - "RAG 평가"
    - "품질 테스트"
    - ...

# After
triggers:
  keywords:
    - "ragulation 평가"
    - "규정 Q&A 품질"
    - "품질 테스트"
    - ...
```

## Traceability

| Requirement ID | Source | Verification |
|----------------|--------|--------------|
| FR-01 | User concern about skill naming | Directory rename successful |
| FR-02 | Context from deep analysis | Description updated in SKILL.md |
| FR-03 | Module structure analysis | Grep returns no old references |
| FR-04 | Existing CLI functionality | CLI commands work unchanged |
| FR-05 | Keyword trigger requirements | New keywords present in frontmatter |
| NFR-01 | Framework integrity requirement | No MoAI-ADK files modified |
| NFR-02 | Functionality preservation | Evaluation tests pass |

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hardcoded references in CLI | Low | Medium | Search and update all references |
| Missing module updates | Medium | Low | Grep all files for old name |
| Cache invalidation | Low | Low | Clear skill cache after rename |
| User confusion | Low | Low | Document change in CHANGELOG |

## Success Metrics

1. Directory renamed successfully from `rag-quality-local` to `ragulation-quality`
2. SKILL.md frontmatter updated with new name and description
3. All module files updated with new references
4. Zero grep results for old skill name in project files
5. CLI commands function identically
6. Quality evaluation runs successfully after rename

---

**TAG:** SPEC-SKILL-REFRESH-001
