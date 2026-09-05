# token_tracker.py — Requirements

## Overview

A module that tracks LLM token usage per model, persisted to a local JSON file. Reads real reported usage from the opencode SQLite database (`opencode.db`) and oh-my-pi session logs.

## API

### `track_tokens(model: str, input_tokens: float, output_tokens: float)`
Persist raw token counts per model to `data/token_log.json`. Pricing is applied at display time (see pricing resolution), never stored. Manual entries coexist with scanner watermarks under the `_meta` key, which readers must ignore.

### `get_totals()`
Print per-model and aggregate input/output token totals, plus cost as an aligned table (`Model / Input / Output / Cost`) with a separator rule before the `Total` row.

### `scan_opencode_db(db_path: str | None = None, pricing_file: str | None = None, obfuscate: bool = False, monthly: bool = True, since_ts: float | None = None, until_ts: float | None = None)`
Read actual token counts from the opencode SQLite database. Auto-detects the DB path across platforms. Applies pricing via whole-segment matching against `--pricing-file` entries, falling back to the file's `_default` key and then `_DEFAULT_PRICING`. When `obfuscate=True`, the DB path in output is truncated to the last 25 characters. Only records newer than the ledger watermark for source `opencode` are tracked; `since_ts`/`until_ts` bound a UTC epoch-seconds window (start inclusive, end exclusive). Returns newly tracked totals keyed by model.

Scan output format: a first line reporting message count and source path (`Scanning <N> messages from <path>...`), then the usage table with a leading `Source` column (`Source / Model / Input / Output / Cost`; grand-total row labeled `Total scanned`), an optional monthly breakdown (`Month / Input / Output / Cost`), and a date-range line. Already-tracked records are reported per source (`Skipped N already-tracked record(s) from <source>.`); when nothing is new, the scan prints `No new usage to track.` and renders no table.

### `scan_pi(sessions_dir: str | None = None, pricing_file: str | None = None, obfuscate: bool = False, monthly: bool = True, since_ts: float | None = None, until_ts: float | None = None)`
Read exact token counts (`usage.input` / `usage.output`) from assistant messages in oh-my-pi JSONL session logs (default `~/.omp/agent/sessions`, one subdirectory per working directory, `<ts>_<uuid>.jsonl` files). Model keys are `provider/model`. pi's own cost fields are ignored — pricing follows the shared rules. Unreadable files are skipped with a warning; malformed lines silently. Watermark and window semantics identical to `scan_opencode_db` (source `oh-my-pi`).

### `scan_all(db_path: str | None = None, sessions_dir: str | None = None, pricing_file: str | None = None, obfuscate: bool = False, monthly: bool = True, since_ts: float | None = None, until_ts: float | None = None)`
Scan every known client source (opencode DB + oh-my-pi session logs) and render one combined table with a Source column. Missing sources are reported and skipped. Watermarks advance independently per source. Returns newly tracked totals keyed by `(source, model)`; the ledger persists per model (sources merged).

### `export_csv(pricing_file: str | None = None)`
Print the tracked ledger as CSV on stdout (`model,input,output,input_cost,output_cost,total_cost`), priced like `get_totals`. The `_meta` section is excluded. Redirect to a file for spreadsheet use.

### `clear_log()`
Remove the token log file.

### `_list_pricing_options()`
List all available pricing configurations, known models, active overrides, and the default fallback pricing.

### `_fmt(n: float) -> str`
Format number with apostrophe delimiter (e.g. `1'234'567`).

## CLI

- `python token_tracker.py` — print totals
- `python token_tracker.py scan-oc [--db-path <path>] [--pricing-file <path>] [--obfuscate]` — scan opencode SQLite database (primary)
- `python token_tracker.py scan-pi [--sessions-dir <path>] [--pricing-file <path>] [--obfuscate]` — scan oh-my-pi session logs
- `python token_tracker.py scan-all [--db-path <path>] [--sessions-dir <path> ...]` — scan all client sources; combined Source-tagged table
- `python token_tracker.py pricing [--pricing-file <path>]` — describe pricing resolution; with a file, also list its entries
- `python token_tracker.py export [--pricing-file <path>]` — print the ledger as CSV on stdout
- `python token_tracker.py clear` — clear the log

## Data Sources

- **opencode SQLite database** — `C:\Users\<user>\.local\share\opencode\opencode.db` (Windows/Linux), `~/.local/share/opencode/opencode.db` (macOS/Linux). Reads actual `tokens.input` / `tokens.output` from the `message` table.
- **oh-my-pi session logs** — `~/.omp/agent/sessions/**/*.jsonl`. Reads exact `usage.input` / `usage.output` from assistant messages.

## Pricing System

### Priority Order (highest to lowest)
1. `--pricing-file <path>` — external JSON file with model pricing (exact key match, then most-specific whole-segment match)
2. `_default` key in the pricing file — fallback within the file when no key matches
3. `_DEFAULT_PRICING` — built-in fallback when nothing else matches (default: `(0.03, 0.05)` = `$0.03/M input, $0.05/M output`)

### External Pricing File Format (`pricing.json`)
```json
{
    "_default": [1.0, 5.0],
    "lmstudio_remote/qwen/qwen3.6-35b-a3b": [3.0, 15.0],
    "lmstudio_remote/google/gemma-4-e4b": [1.0, 5.0]
}
```

### Configuration Points
- `_DEFAULT_PRICING` — fallback `(input_per_m, output_per_m)` for any model without a match (default: `(0.03, 0.05)`)
- `--pricing-file <path>` — external JSON of per-model prices; supports a `_default` key to override the fallback
- Pricing resolution order: exact key match → most-specific whole-segment match → file `_default` → `_DEFAULT_PRICING`

### No Built-in Per-Model Prices
There is no hardcoded per-model price table in the code. Every model resolves to
`_DEFAULT_PRICING` (`$0.03/M input, $0.05/M output`) unless you supply a `--pricing-file`
with an explicit entry (whole-segment matching applies, so e.g. a `qwen` entry matches
any `.../qwen/...` model).

## Output Format

- **Token amounts** — formatted with apostrophe delimiter (e.g. `302'210'678`)
- **Model line** — each model line shows `input=<n>, output=<n>, cost=$X.XX` (token counts use apostrophe delimiter)
- **Totals** — a grand-total line shows combined `input`, `output`, and `cost`; scan adds a monthly breakdown and date-range span
- **Scan path** — scan output shows the DB path (or obfuscated version with `--obfuscate`)
- **Message count** — scan output shows total messages scanned

## Requirements (non-functional)

- **Safe JSON reads** — all JSON file reads must handle missing/corrupted files gracefully, returning `{}` or skipping
- **Concurrent safety** — all ledger writers must use file locking to prevent race conditions on `data/token_log.json`
- **Atomic writes** — token log writes must be atomic (temp file + rename) to prevent corruption
- **Idempotent scans** — scanners must skip records at or below the per-source timestamp watermark stored under `_meta` in the ledger; re-running a scan must never double-count
- **Cross-platform DB discovery** — `scan_opencode_db` must auto-detect the opencode SQLite database on Windows (`AppData/Roaming`, `.local/share`) and Linux/macOS (`.local/share`)
- **Use real token data** — `scan_opencode_db` must read actual `tokens.input` / `tokens.output` from message JSON, not estimate from text
- **Pricing support** — usage is persisted as raw counts; `get_totals`, `scan_opencode_db`, and `scan_pi` must compute cost via the shared pricing resolution (`_get_pricing`) and display per-model and total cost
- **Configurable pricing** — must support a `--pricing-file` CLI flag for external JSON pricing files (with `_default` fallback support) and whole-segment matching against configured model keys
- **Default fallback pricing** — `_DEFAULT_PRICING` (`(0.03, 0.05)`) must be used as fallback when no model match is found; `--pricing-file` JSON files may override it via the `_default` key
- **Pricing listing** — `pricing` CLI command must describe how pricing is configured (defaults vs `--pricing-file`) and show the current `_DEFAULT_PRICING` fallback
- **Per-model totals** — `get_totals` must display per-model input + output sum + cost
- **Scan output cost** — `scan_opencode_db` must display per-model `cost=$X.XX` and total `cost=$X.XX` in scan output
- **Scan path display** — scan output must show the source path being scanned
- **Message count** — scan output must show the number of messages scanned
- **Date range** — scan output must show minimum start date, maximum end date, and total days span
- **Obfuscate path** — `scan_opencode_db` must support `--obfuscate` flag to truncate the DB path in output to the last 25 characters
- **Number formatting** — token amounts must use apostrophe delimiter (e.g. `302'210'678`)
- **Graceful session parsing** — `scan_pi` must skip unreadable session files with a warning and malformed JSONL lines silently instead of crashing
- **Exact usage only** — scanners must persist real reported token counts, never text-based estimates
- **Model name normalization** — model identifiers stored as JSON strings (e.g. `{"id":"x","providerID":"y"}`) must be parsed into a readable `provider/model` format
- **Correct formatting** — no unnecessary format specifiers on `int` values
- **Type hints** — all public functions must have type hints
- **Log cleanup** — must provide a `clear_log()` function to reset the token log

## Dependencies

- `filelock` — for file-level concurrency protection
