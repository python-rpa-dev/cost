# Agent Guidelines — token_tracker

## Tool Call Problems Encountered

### 1. Corrupted `engram_mem_doctor` / `engram_mem_judge` Schema
The function schemas for `engram_mem_doctor` and `engram_mem_judge` contained **duplicate, contradictory definitions** where the description field leaked internal parameter schemas into user-facing text:

```json
{
  "description": "Fixed FTS5 syntax error on special chars\"\n  type: \"bugfix\"\n  content: \"**What**: Wrapped each search term in quotes before passing to FTS5 MATCH\\n**Why**: Users typing queries like 'fix auth bug' would crash because FTS5 interprets special chars as operators\\n**Where**: src/middleware/auth.ts, src/routes/login.ts\\n**Learned**: FTS5 MATCH syntax is NOT the same as LIKE — always sanitize user input\"",
  "type": "string"
}
```

This made the function descriptions unreadable and semantically broken. The `description` field contained raw JSON fragments mixed with markdown, parameter schemas, and internal implementation details that should never have been exposed.

### 2. Prompt Content Corruption
User prompts stored in memory were sometimes corrupted with:
- Internal JSON schema fragments (`"type": "object", "properties": {...}`) leaking into the prompt text
- Parameter definitions mixed into content fields
- Duplicate function declarations for `engram_mem_save_prompt` with conflicting schemas

### 3. Shell Command Tokenization (PowerShell vs Bash)
Windows PowerShell 5.1 does **not** support `&&` as a command separator. Commands must use semicolons (`;`) or the `$? { }` pattern:

```powershell
# WRONG (bash style):
cd /path && git init && git config ...

# CORRECT (PowerShell):
cd /path; git init; git config ...

# OR with conditional chaining:
cd /path; if ($?) { git init }
```

### 4. String Literal Escaping in PowerShell Here-Strings
Using `<< 'PYEOF'` heredoc syntax from bash does not work reliably in PowerShell. Python code embedded in here-strings can cause syntax errors due to unescaped quotes and special characters. Use the `write` tool or a separate `.py` file instead.

### 5. Ruff False Positives on Tuple Types
Ruff reports tuple size mismatches when using `tuple[float, ...]` (variable-length) but expecting `tuple[float, float]` (fixed-length). The fix requires explicit type narrowing with `tuple[float, float]` annotations or suppressing the check.

### 6. LSP Type Inference Gaps
The LSP could not resolve that `tuple[float, ...]` from `_get_pricing()` was always length-2 in practice. Adding explicit return type hints (`-> tuple[float, float]`) resolved the issue.

---

## General Guidelines for Future Sessions

1. **Always use semicolons** for command chaining in PowerShell
2. **Never embed Python here-strings** directly — write to files and execute
3. **Check ruff output** after every edit before committing
4. **Verify tuple types** explicitly when using `_get_pricing()` return values
5. **Use the `write` tool** for creating new files instead of shell redirections
