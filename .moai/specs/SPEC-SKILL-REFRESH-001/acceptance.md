# Acceptance Criteria - SPEC-SKILL-REFRESH-001

---
**SPEC ID:** SPEC-SKILL-REFRESH-001
**Title:** Skill Structure Optimization Acceptance Criteria
**Created:** 2026-02-24
**Status:** Planned
---

## Acceptance Criteria

### AC-01: Directory Structure

**Given** the skill directory exists at `.claude/skills/rag-quality-local/`
**When** the renaming is complete
**Then** the directory shall exist at `.claude/skills/ragulation-quality/`
**And** all files shall be present in the new directory

**Verification:**
```bash
ls -la .claude/skills/ragulation-quality/
# Expected: SKILL.md, modules/, reference.md
```

### AC-02: SKILL.md Name Field

**Given** the SKILL.md file exists
**When** the renaming is complete
**Then** the `name` field shall be `ragulation-quality`
**And** the name shall NOT be `rag-quality-local`

**Verification:**
```bash
grep "name:" .claude/skills/ragulation-quality/SKILL.md | head -1
# Expected: name: ragulation-quality
```

### AC-03: Description Update

**Given** the SKILL.md file exists
**When** the renaming is complete
**Then** the description shall reference "Regulation Manager 규정 Q&A 시스템"
**And** the description shall NOT reference "RAG 시스템" as the primary system name

**Verification:**
```bash
grep "Regulation Manager" .claude/skills/ragulation-quality/SKILL.md
# Expected: At least one match
```

### AC-04: No Orphaned References

**Given** all files have been updated
**When** a search is performed for the old skill name
**Then** no references to `rag-quality-local` shall exist in the skill directory

**Verification:**
```bash
grep -r "rag-quality-local" .claude/skills/ragulation-quality/
# Expected: No output (empty result)
```

### AC-05: Module Files Present

**Given** the renaming is complete
**When** the modules directory is checked
**Then** all module files shall be present
**And** files include: personas.md, scenarios.md, evaluation.md, metrics.md

**Verification:**
```bash
ls .claude/skills/ragulation-quality/modules/
# Expected: personas.md, scenarios.md, evaluation.md, metrics.md
```

### AC-06: Trigger Keywords Updated

**Given** the SKILL.md file exists
**When** the renaming is complete
**Then** the triggers.keywords section shall include "ragulation"
**And** the keywords may include "규정 Q&A" or "regulation"

**Verification:**
```bash
grep -A 20 "triggers:" .claude/skills/ragulation-quality/SKILL.md | grep -E "ragulation|규정"
# Expected: At least one match
```

### AC-07: Framework Skills Unchanged

**Given** the MoAI-ADK framework skills exist
**When** the renaming is complete
**Then** no files in `.claude/skills/moai-*` shall be modified

**Verification:**
```bash
git diff --name-only | grep -c "^\.claude/skills/moai-"
# Expected: 0 (no MoAI skills modified)
```

### AC-08: Functionality Preserved

**Given** the skill has been renamed
**When** the quality evaluation is invoked
**Then** the evaluation shall execute successfully
**And** all 6 personas shall be available
**And** all scenario categories shall be available

**Verification:**
Manual test:
```bash
# Quick evaluation should run without errors
uv run python run_rag_quality_eval.py --quick --summary
```

## Test Scenarios

### Scenario 1: Directory Rename

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Check old directory | Does not exist |
| 2 | Check new directory | Exists with all files |
| 3 | Verify file count | Same as before rename |

### Scenario 2: Frontmatter Update

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Read SKILL.md | File loads correctly |
| 2 | Check name field | `ragulation-quality` |
| 3 | Check description | Contains "Regulation Manager" |

### Scenario 3: Module Integrity

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Read personas.md | No references to old name |
| 2 | Read scenarios.md | No references to old name |
| 3 | Read evaluation.md | No references to old name |
| 4 | Read metrics.md | No references to old name |

### Scenario 4: End-to-End Verification

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Run grep for old name | No results |
| 2 | Load skill in Claude | Skill recognized |
| 3 | Check git status | Only skill files changed |

## Definition of Done

- [ ] Directory renamed to `ragulation-quality`
- [ ] SKILL.md name field updated
- [ ] SKILL.md description updated
- [ ] All module references updated
- [ ] Zero grep results for old skill name
- [ ] All acceptance criteria passed
- [ ] No framework skills modified
- [ ] Git commit created with reference to SPEC

## Quality Gates

### TRUST 5 Compliance

| Pillar | Requirement | Status |
|--------|-------------|--------|
| Tested | All acceptance criteria verified | Pending |
| Readable | Clear naming conventions followed | Pending |
| Unified | Consistent style maintained | Pending |
| Secured | No security implications | N/A |
| Trackable | Git commit with SPEC reference | Pending |

---

**TAG:** SPEC-SKILL-REFRESH-001
