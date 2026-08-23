import json
import glob
import tempfile
import os
from pathlib import Path
from filelock import FileLock
from typing import Optional

TOKEN_LOG = Path("token_log.json")
_LOCK_PATH = str(TOKEN_LOG) + ".lock"

# Default pricing per model key: (input_per_m, output_per_m)
# All known models from your opencode instance
# Models listed here with explicit pricing override the default
_KNOWN_MODELS: dict[str, tuple[float, float]] = {
    # OpenAI defaults
    "openai/gpt-4o": (2.5, 10.0),
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-4": (10.0, 30.0),
    "openai/gpt-4-turbo": (10.0, 30.0),
    "openai/o1": (15.0, 60.0),
    "openai/o3-mini": (1.1, 4.4),
    # Anthropic defaults
    "anthropic/claude-3.5-sonnet": (3.0, 15.0),
    "anthropic/claude-3-opus": (15.0, 75.0),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "anthropic/claude-3.5-haiku": (0.8, 4.0),
    "anthropic/claude-3.7-sonnet": (3.0, 15.0),
    # Google defaults
    "google/gemini-pro": (0.5, 1.5),
    "google/gemini-ultra": (2.5, 7.5),
}

# Default pricing used when no model match is found: (input_per_m, output_per_m)
_DEFAULT_PRICING_DEFAULT = (1.0, 5.0)

# User-editable pricing overrides: map of model_key -> (input_per_m, output_per_m)
# Add your local model pricing here. Format: model_key = (input_price_per_million, output_price_per_million)
# Models not listed here fall through to _DEFAULT_PRICING_DEFAULT ($1.00/M input, $5.00/M output)
# Use "_default" key to override the default for this dict:
#   PRICING_OVERRIDES = {
#       "_default": [1.0, 5.0],
#       "lmstudio_remote/qwen/qwen3.6-35b-a3b": (3.0, 15.0),
#   }
_PricingOverridesWithDefault = {
    "_default": [1.0, 5.0],
}
PRICING_OVERRIDES: dict[str, tuple[float, float]] = {
    # Example: "lmstudio_remote/qwen/qwen3.6-35b-a3b": (3.0, 15.0),
}

# Default opencode SQLite database path (cross-platform)
_DEFAULT_DB_PATHS = [
    Path.home() / ".local" / "share" / "opencode" / "opencode.db",
    Path.home() / "AppData" / "Roaming" / "ai.opencode.desktop" / "opencode.global.dat",
]


def _safe_read_json(path: Path) -> dict:
    """Read JSON from a file, returning {} on any error."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _find_db() -> Optional[Path]:
    """Find the opencode SQLite database across platforms."""
    for p in _DEFAULT_DB_PATHS:
        if p.exists():
            return p
    # Try Windows roaming profile for OpenCode (Electron app)
    win_paths = [
        Path.home() / "AppData" / "Roaming" / "OpenCode" / "opencode.db",
    ]
    for p in win_paths:
        if p.exists():
            return p
    return None


def track_tokens(
    model: str,
    input_tokens: float,
    output_tokens: float,
    input_price_per_m: float = 0.0,
    output_price_per_m: float = 0.0,
):
    """Persist token usage per model to a local JSON file."""
    lock = FileLock(_LOCK_PATH, timeout=5)
    with lock:
        log = _safe_read_json(TOKEN_LOG)
        if model not in log:
            log[model] = {"input": 0, "output": 0, "input_price_per_m": input_price_per_m, "output_price_per_m": output_price_per_m}
        log[model]["input"] += input_tokens
        log[model]["output"] += output_tokens
        log[model]["input_price_per_m"] = input_price_per_m
        log[model]["output_price_per_m"] = output_price_per_m
        fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=TOKEN_LOG.parent)
        try:
            os.write(fd, json.dumps(log, indent=2).encode())
            os.close(fd)
            Path(tmp_path).replace(TOKEN_LOG)
        except Exception:
            os.close(fd)
            Path(tmp_path).unlink(missing_ok=True)
            raise


def _fmt(n: float) -> str:
    """Format number with apostrophe delimiter."""
    return f"{int(n):,}".replace(",", "'")


def get_totals():
    """Read and display all tracked token usage."""
    if not TOKEN_LOG.exists():
        print("No token log found.")
        return
    log = _safe_read_json(TOKEN_LOG)
    total_input = sum(m["input"] for m in log.values())
    total_output = sum(m["output"] for m in log.values())
    total_cost = 0.0
    for model, counts in log.items():
        input_cost = counts["input"] * counts.get("input_price_per_m", 0) / 1_000_000
        output_cost = counts["output"] * counts.get("output_price_per_m", 0) / 1_000_000
        model_cost = input_cost + output_cost
        total_cost += model_cost
        input_p = counts.get("input_price_per_m", 0)
        output_p = counts.get("output_price_per_m", 0)
        total = counts["input"] + counts["output"]
        print(
            f"{model}: "
            f"input={_fmt(counts['input'])}, output={_fmt(counts['output'])}, "
            f"total={_fmt(total)}, "
            f"input_price=${input_p:.2f}/M, output_price=${output_p:.2f}/M, "
            f"cost=${model_cost:.2f}"
        )
    print(f"Total: input={_fmt(total_input)}, output={_fmt(total_output)}, cost=${total_cost:.2f}")


def estimate_tokens(text: str) -> int:
    """Rough word-to-token estimate (1 token ~ 1.3 words for most models)."""
    return int(len(text.split()) * 1.3)


def _parse_model_str(model_str: str) -> str:
    """Extract a readable model name from a JSON string like '{"id":"x","providerID":"y"}'."""
    try:
        parsed = json.loads(model_str)
        provider = parsed.get("providerID", "unknown")
        model_id = parsed.get("id", "unknown")
        variant = parsed.get("variant", "")
        if variant:
            return f"{provider}/{model_id}/{variant}"
        return f"{provider}/{model_id}"
    except (json.JSONDecodeError, TypeError):
        return str(model_str)


def _get_pricing(model_key: str, fallback: tuple[float, float] | None = None) -> tuple[float, float]:
    """Look up (input_price_per_m, output_price_per_m) for a model key."""
    if model_key in PRICING_OVERRIDES:
        return PRICING_OVERRIDES[model_key]
    if model_key in _KNOWN_MODELS:
        return _KNOWN_MODELS[model_key]
    # Try prefix matching: e.g. "lmstudio_remote/qwen/qwen3.6-35b-a3b" matches "qwen"
    for prefix, pricing in _KNOWN_MODELS.items():
        if prefix in model_key or model_key.split("/")[-1] in prefix.split("/"):
            return pricing
    # Check for _default in PRICING_OVERRIDES
    if "_default" in _PricingOverridesWithDefault:
        d = _PricingOverridesWithDefault["_default"]
        return (float(d[0]), float(d[1]))
    return fallback if fallback is not None else _DEFAULT_PRICING_DEFAULT


def _list_pricing_options():
    """List all available pricing configurations."""
    print("Available pricing options:")
    print()
    print("  1. Use defaults from _KNOWN_MODELS (explicit pricing for cloud providers)")
    print("  2. Use PRICING_OVERRIDES (configure in the script)")
    print("  3. Specify a custom pricing file: --pricing-file <path>")
    print()
    print(f"Default (no match): ${_DEFAULT_PRICING_DEFAULT[0]:.2f}/M input, ${_DEFAULT_PRICING_DEFAULT[1]:.2f}/M output")
    print()
    print("To add pricing for a model, edit PRICING_OVERRIDES in this script:")
    print('  PRICING_OVERRIDES = {')
    print('      "lmstudio_remote/qwen/qwen3.6-35b-a3b": (3.0, 15.0),')
    print('  }')
    print()
    print("Known models with explicit pricing:")
    for model, pricing in sorted(_KNOWN_MODELS.items()):
        print(f"  {model}: ${pricing[0]:.2f}/M input, ${pricing[1]:.2f}/M output")
    print()
    if PRICING_OVERRIDES:
        print("Active overrides:")
        for model, pricing in sorted(PRICING_OVERRIDES.items()):
            print(f"  {model}: ${pricing[0]:.2f}/M input, ${pricing[1]:.2f}/M output")
        print()


def _load_pricing_overrides(pricing_file: Optional[str] = None) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """Load pricing overrides from a JSON file, or return PRICING_OVERRIDES.
    
    Returns (overrides_dict, default_pricing). Supports a '_default' key in the JSON file.
    The returned default is used as the ultimate fallback.
    """
    default = _DEFAULT_PRICING_DEFAULT
    if pricing_file:
        try:
            p = Path(pricing_file)
            if p.exists():
                data = json.loads(p.read_text())
                if "_default" in data:
                    default = (float(data["_default"][0]), float(data["_default"][1]))
                return {k: (float(v[0]), float(v[1])) for k, v in data.items() if k != "_default"}, default
            else:
                print(f"Warning: pricing file not found: {pricing_file}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: failed to load pricing file: {e}")
    # No pricing file: use _DEFAULT_PRICING_DEFAULT as fallback
    return PRICING_OVERRIDES, _DEFAULT_PRICING_DEFAULT


def scan_opencode_db(db_path: Optional[str] = None, pricing_file: Optional[str] = None, obfuscate: bool = False):
    """Read token usage from the opencode SQLite database."""
    if db_path:
        db = Path(db_path)
    else:
        db = _find_db()
    if not db or not db.exists():
        print("No opencode database found.")
        return

    overrides, file_default = _load_pricing_overrides(pricing_file)

    import sqlite3

    conn = sqlite3.connect(str(db))
    cursor = conn.execute("SELECT data, time_created FROM message")

    totals: dict[str, dict[str, int | float]] = {}
    rows = cursor.fetchall()
    if obfuscate:
        display_path = "..." + str(db)[-25:]
    else:
        display_path = str(db)
    print(f"Scanning {len(rows)} messages from {display_path}...")

    min_time = None
    max_time = None
    for row in rows:
        data = json.loads(row[0])
        time_created = row[1]
        role = data.get("role")
        tokens = data.get("tokens")
        if not tokens or (tokens.get("input", 0) == 0 and tokens.get("output", 0) == 0):
            continue

        if time_created is not None:
            if min_time is None or time_created < min_time:
                min_time = time_created
            if max_time is None or time_created > max_time:
                max_time = time_created

        model_str = data.get("modelID", "unknown")
        provider = data.get("providerID", "unknown")
        variant = data.get("variant", "")
        if isinstance(model_str, dict):
            provider = model_str.get("providerID", provider)
            model_str = model_str.get("id", model_str)
        if isinstance(model_str, str) and provider:
            key = f"{provider}/{model_str}"
        else:
            key = str(model_str)

        if key not in totals:
            totals[key] = {"input": 0, "output": 0, "input_price_per_m": 0.0, "output_price_per_m": 0.0}

        totals[key]["input"] += tokens.get("input", 0)
        totals[key]["output"] += tokens.get("output", 0)

    conn.close()

    grand_input = 0
    grand_output = 0
    grand_cost = 0.0
    for model, counts in totals.items():
        pricing = overrides.get(model, None)
        if pricing is None:
            pricing = _get_pricing(model, file_default)
        input_p, output_p = pricing
        counts["input_price_per_m"] = input_p
        counts["output_price_per_m"] = output_p
        input_cost = counts["input"] * input_p / 1_000_000
        output_cost = counts["output"] * output_p / 1_000_000
        model_cost = input_cost + output_cost
        grand_cost += model_cost
        track_tokens(model, counts["input"], counts["output"], input_p, output_p)
        print(f"  {model}: input={_fmt(counts['input'])}, output={_fmt(counts['output'])}, "
              f"input_price=${input_p:.2f}/M, output_price=${output_p:.2f}/M, cost=${model_cost:.2f}")

    grand_input = sum(c["input"] for c in totals.values())
    grand_output = sum(c["output"] for c in totals.values())
    print(f"Total scanned: input={_fmt(grand_input)}, output={_fmt(grand_output)}, cost=${grand_cost:.2f}")

    if min_time is not None and max_time is not None:
        from datetime import datetime, timezone
        min_dt = datetime.fromtimestamp(min_time / 1000, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(max_time / 1000, tz=timezone.utc)
        days = (max_dt - min_dt).days
        print(f"Date range: {min_dt.strftime('%Y-%m-%d %H:%M:%S')} to {max_dt.strftime('%Y-%m-%d %H:%M:%S')}, {_fmt(days)} day{'s' if days != 1 else ''} span")


def scan_sessions(session_dir: str = ".opencode/sessions", model: str = "local-llm"):
    """Parse opencode session files and track token usage (legacy JSON format)."""
    total_input = 0
    total_output = 0

    for session_path in glob.glob(f"{session_dir}/*.json"):
        try:
            data = _safe_read_json(Path(session_path))
        except Exception:
            print(f"Warning: skipping {session_path}")
            continue
        for msg in data.get("messages", []):
            content = msg.get("content", "")
            if msg.get("role") == "user":
                total_input += estimate_tokens(content)
            elif msg.get("role") == "assistant":
                total_output += estimate_tokens(content)

    track_tokens(model, total_input, total_output)
    print(f"Scanned sessions: input={total_input}, output={total_output}")


def clear_log():
    """Remove the token log file."""
    if TOKEN_LOG.exists():
        TOKEN_LOG.unlink()
        print("Token log cleared.")
    else:
        print("No token log to clear.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "scan":
            pricing_file = None
            obfuscate = False
            if "--pricing-file" in sys.argv:
                idx = sys.argv.index("--pricing-file")
                if idx + 1 < len(sys.argv):
                    pricing_file = sys.argv[idx + 1]
            if "--obfuscate" in sys.argv:
                obfuscate = True
            scan_opencode_db(pricing_file=pricing_file, obfuscate=obfuscate)
        elif sys.argv[1] == "scan-json":
            scan_sessions()
        elif sys.argv[1] == "clear":
            clear_log()
        elif sys.argv[1] == "pricing":
            _list_pricing_options()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: token_tracker.py [scan|scan-json|pricing|clear]")
    else:
        get_totals()
