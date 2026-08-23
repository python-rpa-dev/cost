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
2. `PRICING` dict in code — Hardcoded known model prices
3. `_DEFAULT_PRICING` — Fallback default of `(0.03, 0.05)` (input/output per million tokens)

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
