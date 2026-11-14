# Agent Guidelines — token_tracker

## Environment
- Active development happens in WSL (Linux). Windows is a *data source* only: its opencode DB is scanned as `opencode@windows` into the shared ledger.
- Per-OS venvs: `.venv-linux/` (WSL) and `.venv-windows/` (Windows). `token_tracker.sh` / `token_tracker.cmd` self-heal their venv; `make check` uses the WSL venv.
- On Windows shells (PowerShell 5.1): chain with `;`, never `&&`; never embed Python here-strings in shell commands — write a file and run it.

## Workflow
- Keep ruff, vulture, pyright, and pytest clean; `.githooks/pre-commit` enforces all four (`git config core.hooksPath .githooks` is set).
- Use the `write` tool for new files, not shell redirection.
- Release flow: commit on `dev`, fast-forward `main`, tag `vX.Y.Z`, push branches and tags.

## Domain gotchas
- Pricing is NEVER stored in the ledger; costs are computed at display time so price changes reprice history.
- The ledger (`data/token_log.json`) carries `_meta` (`schema`, `watermark`, `months`, `last_scan`, `merged`). Readers must ignore unknown `_meta` keys and any top-level key starting with `_`. On a format change, bump `_LEDGER_SCHEMA` in `token_tracker.py` and update `adr.md`.
- `adr.md` must stay current AND mirrored to the codebase-memory graph (MCP `manage_adr`, project `home-developer-projekte-cost`, content = adr.md minus the leading HTML-comment mirror lines).
- opencode.db is foreign data: open read-only (`mode=ro` URI), never write to it; degrade gracefully on malformed rows instead of crashing.
- Model keys come from harness data and drift over time (`lmstudio_remote/...` vs `lm-studio-remote/...`); pricing falls back via whole-segment match, then `_default`.
