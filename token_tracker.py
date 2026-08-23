"""Token usage tracker for LLM interactions."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

TOKEN_LOG = Path("token_log.json")
_LOCK_PATH = str(TOKEN_LOG) + ".lock"


_DEFAULT_PRICING: tuple[float, float] = (0.03, 0.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(n: float) -> str:
    """Format number with apostrophe delimiter."""
    return f"{int(n):,}".replace(",", "'")


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file safely; returns None on failure."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _load_pricing_overrides(pricing_file: str | None = None) -> dict[str, tuple[float, float]]:
    """Load pricing overrides from a JSON file.

    Returns a merged dict of {model_key: (input_per_m, output_per_m)}.
    Supports ``_default`` key for fallback pricing.
    """
    result: dict[str, tuple[float, float]] = {}

    if pricing_file:
        path = Path(pricing_file)
        data = _safe_read_json(path)
        if data is not None:
            default = _DEFAULT_PRICING
            if "_default" in data:
                with contextlib.suppress(TypeError, ValueError):
                    vals = [float(v) for v in data["_default"]]
                    default = (vals[0], vals[1])
            result["__default"] = default
            for key, value in data.items():
                if key == "_default":
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    vals = [float(v) for v in value]
                    result[key] = (vals[0], vals[1])

    return result


def _get_pricing(model_key: str, overrides: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """Look up pricing for a model key.

    Resolution order: exact match, then the most specific override whose path
    segments line up with the model key (as a leading prefix or as full
    contained segments), else the default. Matching is done on whole path
    segments so e.g. ``gpt`` never matches ``gpt2``.
    """
    if model_key in overrides:
        return overrides[model_key]

    model_parts = model_key.split("/")
    best_key: str | None = None
    best_len = -1
    for key in overrides:
        if key.startswith("__default"):
            continue
        key_parts = key.split("/")
        is_prefix = len(key_parts) <= len(model_parts) and all(
            k == m for k, m in zip(key_parts, model_parts)
        )
        is_contained = all(k in model_parts for k in key_parts)
        if (is_prefix or is_contained) and len(key_parts) > best_len:
            best_key, best_len = key, len(key_parts)

    if best_key is not None:
        return overrides[best_key]
    return overrides.get("__default", _DEFAULT_PRICING)


def _parse_model_str(data: dict[str, Any]) -> str:
    """Parse a model identifier from raw API response data."""
    provider = data.get("providerID", data.get("provider", "unknown"))
    model_id = data.get("id", data.get("modelID", data.get("model", "unknown")))
    variant = data.get("variant", "")
    if variant:
        return f"{provider}/{model_id}/{variant}"
    return f"{provider}/{model_id}"


def estimate_tokens(text: str) -> int:
    """Rough word-to-token estimate (1 token ~= 1.3 words)."""
    return int(len(text.split()) * 1.3) if text else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def track_tokens(
    model: str,
    input_tokens: float,
    output_tokens: float,
    input_price_per_m: float = 0.0,
    output_price_per_m: float = 0.0,
) -> None:
    """Persist token usage per model to ``token_log.json``."""
    lock = FileLock(_LOCK_PATH, timeout=10)
    with lock:
        log = _safe_read_json(TOKEN_LOG) or {}
        if model not in log:
            log[model] = {"input": 0, "output": 0}
        log[model]["input"] += input_tokens
        log[model]["output"] += output_tokens

        # Write atomically
        tmp = TOKEN_LOG.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
        tmp.replace(TOKEN_LOG)


def get_totals() -> dict[str, dict[str, float]]:
    """Read and display all tracked token usage with costs."""
    log = _safe_read_json(TOKEN_LOG)
    if not log:
        print("No token log found.")
        return {}

    totals: dict[str, dict[str, float]] = {}
    for model, data in log.items():
        input_t = data.get("input", 0)
        output_t = data.get("output", 0)
        pricing = _get_pricing(model, {"__default": _DEFAULT_PRICING})
        input_cost = input_t * pricing[0] / 1_000_000
        output_cost = output_t * pricing[1] / 1_000_000
        totals[model] = {
            "input": input_t,
            "output": output_t,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }

    # Display
    grand_input = 0.0
    grand_output = 0.0
    grand_cost = 0.0
    for model, t in totals.items():
        print(
            f"{model}: input={_fmt(t['input'])}, output={_fmt(t['output'])}, "
            f"cost=${t['total_cost']:.2f}"
        )
        grand_input += t["input"]
        grand_output += t["output"]
        grand_cost += t["total_cost"]

    print(f"\nTotal: input={_fmt(grand_input)}, output={_fmt(grand_output)}, cost=${grand_cost:.2f}")
    return totals


def scan_opencode_db(
    db_path: str | None = None,
    pricing_file: str | None = None,
    obfuscate: bool = False,
    monthly: bool = True,
) -> dict[str, dict[str, float]]:
    """Scan opencode SQLite database and track token usage.

    Reads actual ``tokens.input`` / ``tokens.output`` from the ``message`` table.
    Applies pricing from *pricing_file* (with whole-segment fallback matching),
    then ``_DEFAULT_PRICING``.
    """
    import sqlite3

    # Resolve DB path
    if db_path:
        db = Path(db_path)
    else:
        candidates = [
            Path.home() / ".local/share/opencode/opencode.db",
            Path.home() / "AppData/Roaming/opencode/opencode.db",
            Path("opencode.db"),
        ]
        db = next((p for p in candidates if p.exists()), None)

    if not db or not db.exists():
        print("No opencode database found.")
        return {}

    display_path = "..." + str(db)[-25:] if obfuscate else str(db)

    # Load pricing
    overrides = _load_pricing_overrides(pricing_file)

    conn = sqlite3.connect(str(db))
    try:
        cursor = conn.execute("SELECT data, time_created FROM message")
        rows = cursor.fetchall()
    finally:
        conn.close()

    print(f"Scanning {len(rows)} messages from {display_path}...")

    totals: dict[str, dict[str, float]] = {}
    by_month: dict[str, dict[str, float]] = {}
    min_time: float | None = None
    max_time: float | None = None

    for row in rows:
        try:
            data = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            continue

        tokens = data.get("tokens", {})
        input_t = tokens.get("input", 0) or 0
        output_t = tokens.get("output", 0) or 0

        # Skip zero-usage entries
        if input_t == 0 and output_t == 0:
            continue

        # _parse_model_str() already returns "provider/model_id[/variant]".
        # Do NOT prepend the provider again (that produced doubled prefixes).
        key = _parse_model_str(data)

        if key not in totals:
            pricing = _get_pricing(key, overrides)
            totals[key] = {
                "input": 0.0,
                "output": 0.0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
            }

        totals[key]["input"] += input_t
        totals[key]["output"] += output_t

        # Track timestamps and monthly buckets
        time_created = row[1]
        if time_created:
            ts = float(time_created) / 1000 if len(str(time_created)) > 12 else float(time_created)
            if min_time is None or ts < min_time:
                min_time = ts
            if max_time is None or ts > max_time:
                max_time = ts

            month_key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
            if month_key not in by_month:
                by_month[month_key] = {"input": 0.0, "output": 0.0, "cost": 0.0}
            row_pricing = _get_pricing(key, overrides)
            by_month[month_key]["input"] += input_t
            by_month[month_key]["output"] += output_t
            by_month[month_key]["cost"] += (input_t * row_pricing[0] + output_t * row_pricing[1]) / 1_000_000

    # Calculate costs and persist
    grand_input = 0.0
    grand_output = 0.0
    grand_cost = 0.0
    for key, t in totals.items():
        pricing = _get_pricing(key, overrides)
        t["input_cost"] = t["input"] * pricing[0] / 1_000_000
        t["output_cost"] = t["output"] * pricing[1] / 1_000_000
        t["total_cost"] = t["input_cost"] + t["output_cost"]

        grand_input += t["input"]
        grand_output += t["output"]
        grand_cost += t["total_cost"]

        track_tokens(key, t["input"], t["output"])

    # Display per-model results
    for key, t in totals.items():
        print(
            f"  {key}: input={_fmt(t['input'])}, output={_fmt(t['output'])}, "
            f"cost=${t['total_cost']:.2f}"
        )

    # Display monthly breakdown before grand total
    if monthly:
        print("\nMonthly Breakdown:")
        for month_key in sorted(by_month.keys()):
            m = by_month[month_key]
            print(
                f"  {month_key}: input={_fmt(m['input'])}, output={_fmt(m['output'])}, "
                f"cost=${m['cost']:.2f}"
            )

    print(f"\nTotal scanned: input={_fmt(grand_input)}, output={_fmt(grand_output)}, cost=${grand_cost:.2f}")

    # Date range info
    if min_time is not None and max_time is not None:
        min_dt = datetime.fromtimestamp(min_time, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(max_time, tz=timezone.utc)
        days = (max_dt - min_dt).days
        print(
            f"Date range: {min_dt.strftime('%Y-%m-%d %H:%M:%S')} to "
            f"{max_dt.strftime('%Y-%m-%d %H:%M:%S')}, {_fmt(days)} day{'s' if days != 1 else ''} span"
        )

    return totals


def scan_sessions(
    session_dir: str = ".opencode/sessions",
    model: str = "local-llm",
) -> None:
    """Walk legacy JSON session files, estimate tokens, and track usage."""
    total_input = 0
    total_output = 0

    for session_path in Path(session_dir).glob("*.json"):
        data = _safe_read_json(session_path)
        if not data:
            print(f"Warning: skipping {session_path}")
            continue

        for msg in data.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                total_input += estimate_tokens(content)
            elif role == "assistant":
                total_output += estimate_tokens(content)

    track_tokens(model, total_input, total_output)
    print(f"Scanned sessions: input={total_input}, output={total_output}")


def clear_log() -> None:
    """Remove the token log file."""
    if TOKEN_LOG.exists():
        TOKEN_LOG.unlink()
        print("Token log cleared.")
    else:
        print("No token log to clear.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="token_tracker",
        description="Track LLM token usage and costs.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="totals",
        choices=["totals", "scan", "scan-json", "clear", "pricing"],
        help="Command to run (default: totals)",
    )
    parser.add_argument("--pricing-file", metavar="PATH", default=None, help="External JSON pricing overrides")
    parser.add_argument("--obfuscate", action="store_true", help="Truncate the DB path in output")
    parser.add_argument("--no-monthly", action="store_true", help="Suppress the monthly breakdown")
    return parser


def _parse_cli_args(argv: list[str]) -> dict[str, Any]:
    """Parse command-line arguments into a plain dict."""
    ns = _build_parser().parse_args(argv[1:] if argv else [])
    return {
        "command": ns.command,
        "pricing_file": ns.pricing_file,
        "obfuscate": ns.obfuscate,
        "monthly": not ns.no_monthly,
    }


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv

    parsed = _parse_cli_args(argv)
    cmd = parsed["command"]

    if cmd == "scan":
        scan_opencode_db(
            pricing_file=parsed.get("pricing_file"),
            obfuscate=parsed.get("obfuscate", False),
            monthly=parsed.get("monthly", True),
        )
    elif cmd == "scan-json":
        scan_sessions()
    elif cmd == "clear":
        clear_log()
    elif cmd == "pricing":
        _list_pricing_options()
    else:
        get_totals()


def _list_pricing_options() -> None:
    """List all available pricing configurations."""
    print("Available pricing options:\n")
    print("  1. Use defaults from _KNOWN_MODELS (explicit pricing for cloud providers)")
    print("  2. Use PRICING_OVERRIDES (configure in the script)")
    print("  3. Specify a custom pricing file: --pricing-file <path>\n")

    print(f"Default (no match): ${_DEFAULT_PRICING[0]:.2f}/M input, ${_DEFAULT_PRICING[1]:.2f}/M output\n")
    print("To add pricing for a model, use the --pricing-file option or set PRICING_OVERRIDES:\n")


if __name__ == "__main__":
    main()
