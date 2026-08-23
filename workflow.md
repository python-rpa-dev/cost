# General Workflow of Interaction

> Status: This is a historical snapshot of the development process for `token_tracker`.
> It documents *how* changes were made, not the tool's current behavior — see `setup.md`
> for the current spec and `README.md` for usage.

## Pattern: Iterative Code Development with LSP Feedback

### 1. Requirement Specification
- User states a feature requirement or change request
- Assistant confirms understanding and plans implementation

### 2. Read & Understand
- Assistant reads relevant files to understand current code state
- Reads may span multiple turns if file is large or context is needed

### 3. Edit / Implement
- Assistant applies code changes via the `edit` tool
- Multiple edit attempts may occur as the assistant refines the change
- First attempts may fail (string not found, identical strings) — assistant retries with corrected content

### 4. LSP Error Resolution
- After each edit, LSP reports type errors and other issues
- Assistant diagnoses errors and applies fixes
- May require multiple rounds: fix return types, update call sites, adjust signatures

### 5. Test & Verify
- Assistant runs commands to verify functionality works correctly
- Tests include both the new feature and existing functionality
- Output is examined to confirm correctness

### 6. Documentation Update
- Assistant updates spec files (e.g., `setup.md`) to reflect new requirements/features
- Changes are documented alongside related existing requirements

### 7. Iterate on Clarification
- User may restate or clarify requirements (e.g., "add a default option" repeated with clarification "in pricing.json")
- Assistant incorporates clarified requirements into implementation

### 8. Summary & Confirmation
- Assistant summarizes all changes made
- Lists new features, flags, and behavior

---

## Key Interaction Patterns

### Edit Retry Pattern
1. `edit` fails with "Could not find oldString" → assistant reads file, finds exact content
2. `edit` fails with "oldString and newString are identical" → assistant revises the change
3. `edit` succeeds → LSP errors reported → assistant fixes type mismatches and call sites

### Multi-File Coordination
- Changes span multiple files: source code (`token_tracker.py`), data files (`pricing.json`), specs (`setup.md`)
- Each file type serves a different purpose: implementation, configuration, documentation

### Error-Driven Iteration
- LSP errors drive the next round of edits (type annotations, return types, parameter signatures)
- Runtime test output reveals functional issues (missing features, wrong behavior)

### User Clarification Loop
- User rephrases unclear requirements
- Assistant asks for or waits for clarification before proceeding

---

## Tool Usage Statistics from Session

| Tool | Count | Purpose |
|------|-------|---------|
| `edit` | 10 | Apply code changes, update specs |
| `read` | 7 | Understand current state, fix failed edits |
| `bash` | 3 | Run tests, verify functionality |
| `write` | 1 | Create new files (pricing.json) |

## Workflow Diagram

```
User Requirement
    │
    ▼
┌──────────┐
│  Read    │
│ Current  │
│ Code     │
└────┬─────┘
     │
     ▼
┌──────────┐
│ Already  │──yes──▶ Done
│ Done?    │
└────┬─────┘
     │
     │no
     ▼
┌──────────┐
│  Edit/   │
│ Write    │
└────┬─────┘
     │
     ▼
┌──────────┐
│  LSP     │──yes──▶┌──────────┐
│ Errors?  │        │  Fix     │
└────┬─────┘        │  Errors  │
     │              └────┬─────┘
     │no                 │
     ▼                   ▼
┌──────────┐     ┌──────────┐
│  Test/   │────▶│  Run     │
│  Run     │     └──────────┘
└────┬─────┘
     │
     ▼
┌──────────┐
│  User    │
│  Clarif. │
└────┬─────┘
     │
     ▼
┌──────────┐
│  Update  │
│  Docs    │
└────┬─────┘
     │
     ▼
┌──────────┐
│ Summary  │──yes──▶ Done
│ & Confirm│
└────┬─────┘
     │
     │no
     │
     └──────▶ (loop back to Edit/Write)
```

## Principles Observed

1. **Read before edit** — always verify current file content before editing
2. **Fix LSP errors immediately** — type mismatches are caught and resolved in the same session
3. **Test after every major change** — verify both new and existing functionality
4. **Update specs alongside code** — requirements documentation stays in sync
5. **Handle user clarification gracefully** — rephrased requirements are incorporated without losing context
6. **Atomic changes** — each edit does one focused thing; large changes are split across multiple edits
7. **Graceful degradation** — errors in loading config files produce warnings, not crashes

## Recommended Improvements (from session analysis)

8. **Auto-retry on edit failure** — when `edit` fails, automatically re-read and retry instead of waiting for manual intervention
9. **LSP error auto-fix** — detect common error patterns (return type mismatches, missing params) and fix them proactively
10. **Test gate before completion** — require verification tests to pass before claiming a task is done
11. **Change summary for confirmation** — present a diff/summary of all changes before marking a session complete
12. **Duplicate requirement detection** — detect when a user restates an already-addressed requirement and ask for clarification
13. **Config validation** — validate external JSON files for correct structure before using them
14. **Dependency checks** — verify all dependencies are available at startup with clear error messages

## Workflow Requirements (merged from setup.md)

- **Dependency availability** — all external dependencies (`filelock`, `sqlite3`) must be checked at startup with a clear error message if unavailable
- **Config file validation** — external JSON files (e.g. `pricing.json`) must be validated for correct structure before use, with a warning on invalid format
- **Edit auto-retry** — when `edit` fails with "oldString not found" or "identical string", the assistant should automatically re-read the file and retry with corrected content
- **LSP error auto-diagnosis** — common LSP errors (return type mismatches, missing parameters) should be diagnosed and fixed in the same iteration rather than requiring separate rounds
- **Test before claim** — the assistant must run verification tests before claiming a task is complete, ensuring both new and existing functionality works
- **Change summary** — before marking a session complete, the assistant should present a diff or summary of all changes for user confirmation
- **Duplicate requirement detection** — when a user restates a requirement that was already addressed, the assistant should detect this and ask for clarification rather than silently reprocessing
