# token_tracker

Track LLM token usage per model with cost estimation. Reads real reported usage from the opencode SQLite database (`opencode.db`) and oh-my-pi session logs.

## Quick Start

```bash
# Easiest: the launcher script creates the per-OS venv and installs deps on first run
./token_tracker.sh          # Linux/macOS  -> .venv-linux
token_tracker.cmd           # Windows      -> .venv-windows

# Or manually:
python -m venv .venv-linux             # or .venv-windows on Windows
.venv-windows\Scripts\activate         # Windows
source .venv-linux/bin/activate        # macOS/Linux
pip install -e ".[dev]"

# Run
python token_tracker.py
```

## CLI Usage

| Command | Description |
|---------|-------------|
| `python token_tracker.py [totals] [--pricing-file <path>]` | Print per-model totals (input/output tokens + cost) |
| `python token_tracker.py scan-oc [--db-path <path>] [--pricing-file <path>] [--obfuscate] [--no-monthly]` | Scan opencode SQLite database |
| `python token_tracker.py scan-pi [--sessions-dir <path>] [--pricing-file <path>] [--obfuscate] [--no-monthly]` | Scan oh-my-pi session logs (`~/.omp/agent/sessions`) |
| `python token_tracker.py scan-all [--db-path <path>] [--sessions-dir <path>] [--pricing-file <path>] [--obfuscate] [--no-monthly]` | Scan all client sources; one table with a Source column |
| `python token_tracker.py pricing` | Describe how pricing is configured and show the default fallback |
| `python token_tracker.py clear` | Clear the token log file |

### Options

- `--db-path <path>` — Scan an explicit opencode.db file (`scan-oc`/`scan-all`; default: auto-detect)
- `--sessions-dir <path>` — Directory of pi session logs (`scan-pi`/`scan-all`; default: `~/.omp/agent/sessions`)
- `--pricing-file <path>` — Use an external JSON file for pricing overrides (applies to `totals`, `scan-oc`, `scan-pi`, and `scan-all`)
- `--no-monthly` — Suppress the monthly cost breakdown

> A `token-tracker` console script is also installed (`pip install -e .`), so you can run e.g. `token-tracker scan`.

## Data Sources

1. **opencode SQLite database** — Reads actual `tokens.input` / `tokens.output` from the `message` table (opened read-only)
   - Windows: `%APPDATA%\opencode\opencode.db`
   - macOS/Linux: `~/.local/share/opencode/opencode.db`

2. **oh-my-pi session logs** — Reads exact `usage.input` / `usage.output` from assistant messages in JSONL transcripts under `~/.omp/agent/sessions` (one subdirectory per working directory, `<ts>_<uuid>.jsonl` files)

## Pricing System

Pricing resolution follows this priority order (highest to lowest):

1. `--pricing-file <path>` — External JSON file with model-specific pricing (exact key match, then most-specific whole-segment match)
2. `_default` key in the pricing file — Fallback within the file when no key matches
3. `_DEFAULT_PRICING` — Built-in fallback of `(0.03, 0.05)` ($0.03/M input, $0.05/M output)

### Pricing File Format

```json
{
  "_default": [0.03, 0.05],
  "openai/gpt-4o": [2.5, 10.0],
  "anthropic/claude-3": [3.0, 15.0]
}
```

## Development

### Running Tests

```bash
python -m pytest
python -m pytest --cov=token_tracker --cov-report=term-missing
```

### Linting

```bash
ruff check token_tracker.py
ruff format token_tracker.py
```

### Type Checking

```bash
pyright token_tracker.py tests   # or: make typecheck
```
