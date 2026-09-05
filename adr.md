# ADR — token_tracker

<!-- Source of truth is this file (git-tracked). After editing, mirror it into the
     codebase-memory graph index with manage_adr(mode="update", content=<this file>). -->

## PURPOSE
CLI (`token-tracker`) that tracks LLM token usage and cost across client harnesses. Scanners read exact reported usage: primary source is the opencode SQLite DB (`message` table, `tokens.input`/`tokens.output`); second scanner reads oh-my-pi session logs (`~/.omp/agent/sessions`, JSONL v3, assistant-message `usage.input`/`usage.output`). Aggregates per model and per month, prices it, prints aligned tables, and persists a cumulative ledger to `data/token_log.json`; scans are idempotent via per-source watermarks.

## STACK
- Python >=3.9 (ruff target py39), single-module package built with setuptools; console script `token-tracker = token_tracker:main`.
- Runtime deps: exactly one — `filelock>=3.0`. Everything else is stdlib (`sqlite3`, `json`, `argparse`, `datetime`).
- Dev: pytest + pytest-cov (coverage in addopts), ruff (E,F,W,I,N,UP,B,SIM,RUF; E501 ignored), pyright, vulture (paths include tests so public API used by them counts as live; ignore_names covers only TypedDict members read via subscript).
- Dual-OS repo by design: `.venv-windows`/`.venv-linux`, `token_tracker.cmd`/`token_tracker.sh` (self-healing launchers), venv-pinned Makefile targets (WSL2 + Windows workflow).

## ARCHITECTURE
Flat single module `token_tracker.py` with banner-separated regions: Helpers -> Public API -> CLI entry point. Tests mirror it 1:1 in `tests/test_token_tracker.py`.

Shared core (the client-plugin seam): scanners normalize their source into `UsageRecord(key, input_tokens, output_tokens, ts, source)` tuples; `_scan_report(records, overrides, monthly, since_ts, until_ts)` filters to the `[since_ts, until_ts)` window, gates records against per-source watermarks under the ledger's `_meta`, then — inside one FileLock read-modify-write — sums fresh records via `_aggregate` (per `(source, model)`, `TotalsEntry` totals + monthly buckets + time span), merges them into the log per model (sources merged), advances each source's watermark to its newest fresh timestamp, and atomically writes. Rendering: Source-column table (`Source / Model / Input / Output / Cost`, grand-total row `Total scanned`) plus `_print_monthly`/`_print_date_range`. `get_totals` keeps the plain 4-column model table; manual `track_tokens` shares the log via `_track_batch`.

- `scan_opencode_db`: resolve DB path (explicit or XDG/AppData/cwd candidates) -> read-only sqlite URI (`mode=ro`) -> parse `message.data` JSON rows -> records keyed `provider/model_id[/variant]`.
- `scan_pi`: walk `~/.omp/agent/sessions/**/*.jsonl` (one dir per working directory, `<ts>_<uuid>.jsonl`) -> assistant-message lines only -> records keyed `provider/model`; pi's own cost fields ignored (zero for local models), cache read/write counts deliberately excluded.
- CLI: `totals | scan-oc | scan-pi | scan-all | clear | pricing | export`, flags `--pricing-file --obfuscate --no-monthly --db-path --sessions-dir --since --until`; `scan-all` composes both collectors into one Source-tagged report and returns newly tracked totals keyed `(source, model)`.

Storage: `data/token_log.json` (gitignored, dir auto-created) is the cumulative ledger — per-model usage plus a `_meta.watermark` map of source -> newest tracked epoch seconds; readers must skip keys starting with `_`. `pricing.json` supplies per-model overrides merged over `_DEFAULT_PRICING = (0.03, 0.05)` USD per million tokens.

## PATTERNS
- Locked atomic persistence: FileLock(timeout=10) around read-modify-write, then write `.tmp` + `Path.replace` — crash-safe and concurrent-writer-safe (`_track_batch`).
- Idempotent scans: per-source timestamp watermark under `_meta`; records at/below it are skipped and reported (`Skipped N already-tracked record(s)...`), so re-scanning full-history sources never double-counts. Undated records count once, on the source's first scan. `clear` resets ledger and watermarks together.
- Read-only opens for data you do not own (opencode.db via `mode=ro` URI).
- Graceful degradation everywhere external: `_safe_read_json` -> None on failure; malformed JSON rows/lines skipped; unreadable session files warn-and-skip; `sqlite3.OperationalError` -> message + `{}`. Scans never crash on foreign/corrupt data.
- Pricing resolution order in `_get_pricing`: exact key -> most specific override path prefix -> `__default` -> `_DEFAULT_PRICING`. Explicit `-> tuple[float, float]` return type — required to close an LSP/ruff inference gap (`tuple[float, ...]` was not narrowed); keep it.
- Typed contracts: `TotalsEntry` (TypedDict) is the totals shape; `PricePair = tuple[float, float]`; `_cost()`/`_priced()` are the only places cost math happens.
- Timestamp normalization `_to_epoch_seconds`: ms-vs-s heuristic (`ts > 1e11`) shared by all scanners.
- Batch writes: one locked read-modify-write per scan, never per row.

## TRADEOFFS
- Single module vs package split: accepted for a small CLI; cost is fan-in concentration on `_get_pricing` (highest in graph) — any signature change there needs full reference check.
- Exact usage only: the legacy `scan-json`/`estimate_tokens` path (~1.3 tok/word text estimation) was deleted once oh-my-pi logs provided exact counts; no scanner may persist estimates (setup.md requirement).
- No client registry yet: the two scanners share the core via direct calls and `scan_all` composes their collectors; a `{name: scan_fn}` dispatch table appears only when a third client lands (no premature abstraction).
- JSON ledger instead of SQLite for own log: human-readable and trivially diffable; pays for it with FileLock because there is no transactionality.
- Watermark is monotonic per source: a `--since` window on a fresh ledger permanently skips older history (the ledger only moves forward); accepted for the simple invariant — full rebuild is one `clear` + rescan away.
- Per-OS venv dirs instead of one cross-platform env: duplication buys self-healing launcher scripts on both Windows and WSL2.

## PHILOSOPHY
Zero heavy dependencies; never mutate data you do not own; degrade gracefully rather than crash on foreign input; atomic-or-nothing for anything persisted; exact counts over estimates; explicit types over tool inference. PowerShell-first shell habits (no `&&`; no Python here-strings) per AGENTS.md — Windows is a first-class host.
