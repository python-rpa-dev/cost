# token_tracker

Track LLM token usage per model with cost estimation. Supports reading from the opencode SQLite database (`opencode.db`) and legacy JSON session files.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -e ".[dev]"

# Run
python token_tracker.py
```

## CLI Usage

| Command | Description |
|---------|-------------|
| `python token_tracker.py` | Print per-model totals (input/output tokens + cost) |
| `python token_tracker.py scan [--pricing-file <path>] [--obfuscate]` | Scan opencode SQLite database |
| `python token_tracker.py scan-json` | Scan legacy JSON session files (`.opencode/sessions/*.json`) |
| `python token_tracker.py pricing` | List all available pricing options and known models |
| `python token_tracker.py clear` | Clear the token log file |

### Options

- `--pricing-file <path>` — Use an external JSON file for pricing overrides
- `--obfuscate` — Truncate the database path in output to the last 25 characters

## Data Sources

1. **opencode SQLite database** — Reads actual `tokens.input` / `tokens.output` from the `message` table
   - Windows: `C:\Users\<user>\AppData\Roaming\ai.opencode.desktop\opencode.global.dat` or `.local/share/opencode/opencode.db`
   - macOS/Linux: `~/.local/share/opencode/opencode.db`

2. **Legacy JSON sessions** — Fallback for older opencode versions (`.opencode/sessions/*.json`)

## Pricing System

Pricing resolution follows this priority order (highest to lowest):

1. `--pricing-file <path>` — External JSON file with model-specific pricing
2. `PRICING_OVERRIDES` — In-script dictionary for quick overrides
3. `_KNOWN_MODELS` — Auto-detected models from your opencode instance
4. `_DEFAULT_PRICING` — Fallback for unknown models (default: $1.00/M input, $5.00/M output)

### External Pricing File Format (`pricing.json`)

```json
{
  "_default": [1.0, 5.0],
  "openai/gpt-4o": [2.5, 10.0],
  "anthropic/claude-3.5-sonnet": [3.0, 15.0]
}
```

Each entry maps a model key to `[input_price_per_million, output_price_per_million]`.

## API Reference

### `track_tokens(model: str, input_tokens: float, output_tokens: float, input_price_per_m: float = 0.0, output_price_per_m: float = 0.0) -> None`

Persist token usage per model to `token_log.json`, including pricing info. Uses atomic file writes with locking for concurrent safety.

### `get_totals() -> dict[str, dict[str, float]]`

Read and display all tracked token usage with costs. Returns a dictionary keyed by model name.

Output format:
```
gpt-4o: input=1'234'567, output=89'012, cost=$3.45
claude-3.5-sonnet: input=567'890, output=234'567, cost=$2.10

Total: input=1'802'457, output=323'579, cost=$5.55
```

### `scan_opencode_db(db_path: str | None = None, pricing_file: str | None = None, obfuscate: bool = False) -> dict[str, dict[str, float]]`

Scan opencode SQLite database and track token usage. Reads actual `tokens.input` / `tokens.output` from the `message` table. Auto-detects DB path across platforms. Applies pricing from `pricing_file`, then `_KNOWN_MODELS`, then defaults.

### `scan_sessions(session_dir: str = ".opencode/sessions", model: str = "local-llm") -> None`

Walk legacy JSON session files, estimate tokens from user/assistant message content, and track via `track_tokens`. Skips unparseable files with a warning.

### `clear_log() -> None`

Remove the token log file (`token_log.json`).

## Project Structure

```
cost/
├── token_tracker.py    # Main module (all logic)
├── pricing.json        # External pricing overrides
├── token_log.json      # Persisted token usage data
├── setup.md            # Requirements & API docs
├── workflow.md         # Interaction workflow
├── AGENTS.md           # Agent guidelines & gotchas
├── pyproject.toml      # Dependencies & tool config (ruff, vulture, pytest)
├── .gitignore          # Excludes .venv/, __pycache__, etc.
└── tests/
    └── test_token_tracker.py  # Unit tests
```

## Development

### Dependencies

- Python 3.9+
- `filelock` — File-level concurrency protection
- `sqlite3` — Standard library (bundled with Python)

### Code Quality

```bash
# Lint
ruff check token_tracker.py

# Dead code detection
vulture token_tracker.py

# Run tests with coverage
pytest -v --cov=token_tracker
```

### Branch Strategy

- `dev` — Active development branch
- `main` — Stable/milestone branch (merge from dev when ready)

## License

MIT
