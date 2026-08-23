# token_tracker.py — Requirements

## Overview

A module that tracks LLM token usage per model, persisted to a local JSON file. Supports reading from the opencode SQLite database (`opencode.db`) and legacy JSON session files.

## API

### `track_tokens(model: str, input_tokens: float, output_tokens: float, input_price_per_m: float = 0.0, output_price_per_m: float = 0.0)`
Persist token usage per model to `token_log.json`, including pricing info.

### `get_totals()`
Print per-model and aggregate input/output token totals, plus cost. Output format:
```
<model>: input=<N>, output=<N>, total=<N>, input_price=$X.XX/M, output_price=$X.XX/M, cost=$X.XX
Total: input=<N>, output=<N>, cost=$X.XX
```

### `estimate_tokens(text: str) -> int`
Rough word-to-token estimate (`words * 1.3`).

### `scan_opencode_db(db_path: str | None = None, pricing_file: str | None = None, obfuscate: bool = False)`
Read actual token counts from the opencode SQLite database. Auto-detects the DB path across platforms. Applies pricing via whole-segment matching against `--pricing-file` entries, falling back to the file's `_default` key and then `_DEFAULT_PRICING`. When `obfuscate=True`, the DB path in output is truncated to the last 25 characters.

Scan output format:
```
Scanning <N> messages from <path>...
  <model>: input=<N>, output=<N>, input_price=$X.XX/M, output_price=$X.XX/M, cost=$X.XX
Total scanned: input=<N>, output=<N>, cost=$X.XX
Date range: <start_date> to <end_date>, <N> days span
```

### `scan_sessions(session_dir: str = ".opencode/sessions", model: str = "local-llm")`
Walk legacy JSON session files, estimate tokens from user/assistant message content, track via `track_tokens`.

### `clear_log()`
Remove the token log file.

### `_list_pricing_options()`
List all available pricing configurations, known models, active overrides, and the default fallback pricing.

### `_fmt(n: float) -> str`
Format number with apostrophe delimiter (e.g. `1'234'567`).

## CLI

- `python token_tracker.py` — print totals
- `python token_tracker.py scan [--pricing-file <path>] [--obfuscate]` — scan opencode SQLite database (primary)
- `python token_tracker.py scan-json` — scan legacy JSON session files
- `python token_tracker.py pricing` — list all available pricing options
- `python token_tracker.py clear` — clear the log

## Data Sources

- **opencode SQLite database** — `C:\Users\<user>\.local\share\opencode\opencode.db` (Windows/Linux), `~/.local/share/opencode/opencode.db` (macOS/Linux). Reads actual `tokens.input` / `tokens.output` from the `message` table.
- **Legacy JSON sessions** — `.opencode/sessions/*.json` (fallback for older opencode versions).

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
- **Concurrent safety** — `track_tokens` must use file locking to prevent race conditions on `token_log.json`
- **Atomic writes** — token log writes must be atomic (temp file + rename) to prevent corruption
- **Cross-platform DB discovery** — `scan_opencode_db` must auto-detect the opencode SQLite database on Windows (`AppData/Roaming`, `.local/share`) and Linux/macOS (`.local/share`)
- **Use real token data** — `scan_opencode_db` must read actual `tokens.input` / `tokens.output` from message JSON, not estimate from text
- **Pricing support** — `track_tokens` must accept `input_price_per_m` and `output_price_per_m` parameters; `get_totals` must compute and display cost per model and total
- **Configurable pricing** — must support a `--pricing-file` CLI flag for external JSON pricing files (with `_default` fallback support) and whole-segment matching against configured model keys
- **Default fallback pricing** — `_DEFAULT_PRICING` (`(0.03, 0.05)`) must be used as fallback when no model match is found; `--pricing-file` JSON files may override it via the `_default` key
- **Pricing listing** — `pricing` CLI command must describe how pricing is configured (defaults vs `--pricing-file`) and show the current `_DEFAULT_PRICING` fallback
- **Per-model totals** — `get_totals` must display per-model input + output sum + cost
- **Scan output cost** — `scan_opencode_db` must display per-model `cost=$X.XX` and total `cost=$X.XX` in scan output
- **Scan path display** — scan output must show the DB path being scanned
- **Message count** — scan output must show the number of messages scanned
- **Date range** — scan output must show minimum start date, maximum end date, and total days span
- **Obfuscate path** — `scan_opencode_db` must support `--obfuscate` flag to truncate the DB path in output to the last 25 characters
- **Number formatting** — token amounts must use apostrophe delimiter (e.g. `302'210'678`)
- **Graceful session parsing** — `scan_sessions` must skip unparseable session files with a warning instead of crashing
- **Configurable model name** — `scan_sessions` must accept a `model` parameter instead of hardcoding
- **Model name normalization** — model identifiers stored as JSON strings (e.g. `{"id":"x","providerID":"y"}`) must be parsed into a readable `provider/model` format
- **Correct formatting** — no unnecessary format specifiers on `int` values
- **Type hints** — all public functions must have type hints
- **Log cleanup** — must provide a `clear_log()` function to reset the token log

## Dependencies

- `filelock` — for file-level concurrency protection
