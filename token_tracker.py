"""Token usage tracker for LLM client harnesses (opencode, oh-my-pi)."""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from filelock import FileLock

TOKEN_LOG = Path("token_log.json")
_LOCK_PATH = str(TOKEN_LOG) + ".lock"


_DEFAULT_PRICING: tuple[float, float] = (0.03, 0.05)

#: ``(input_price_per_million, output_price_per_million)``.
PricePair = tuple[float, float]

#: Sentinel key under which a pricing file's own fallback is stored.
_DEFAULT_KEY = "__default"


class TotalsEntry(TypedDict):
    """Aggregated usage and cost for one model."""

    input: float
    output: float
    input_cost: float
    output_cost: float
    total_cost: float


class UsageRecord(NamedTuple):
    """One usage observation from any client harness.

    ``ts`` is epoch seconds (already normalized from ms) or ``None``.
    """

    key: str
    input_tokens: float
    output_tokens: float
    ts: float | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(n: float) -> str:
    """Format number with apostrophe delimiter."""
    return f"{int(n):,}".replace(",", "'")


def _render_row(cells: tuple[str, ...], widths: list[int], aligns: str) -> str:
    """Join cells into one line padded to *widths*, aligned per column ('<' or '>')."""
    return "  ".join(
        f"{c:>{w}}" if a == ">" else f"{c:<{w}}"
        for c, w, a in zip(cells, widths, aligns)
    )


def _print_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    aligns: str = "<>>>",
    rule_before: int | None = None,
) -> None:
    """Print aligned columns; *aligns* is one '<'/'>' char per column.

    A separator line is drawn before row index *rule_before* (e.g. the total row).
    """
    widths = [max(len(cells[i]) for cells in (headers, *rows)) for i in range(len(headers))]
    rule = "  ".join("-" * w for w in widths)
    lines = [_render_row(headers, widths, aligns), rule]
    for i, row in enumerate(rows):
        if i == rule_before:
            lines.append(rule)
        lines.append(_render_row(row, widths, aligns))
    print("\n".join(lines))


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file safely; returns None on failure."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _coerce_pair(value: Any) -> PricePair | None:
    """Coerce a JSON ``[in_per_m, out_per_m]`` value; None if malformed."""
    with contextlib.suppress(TypeError, ValueError, IndexError):
        vals = [float(v) for v in value]
        return (vals[0], vals[1])
    return None


def _load_pricing_overrides(pricing_file: str | None = None) -> dict[str, PricePair]:
    """Load pricing overrides from a JSON file.

    Returns a merged dict of {model_key: (input_per_m, output_per_m)}.
    Supports ``_default`` key for fallback pricing.
    """
    result: dict[str, PricePair] = {}

    if pricing_file:
        path = Path(pricing_file)
        data = _safe_read_json(path)
        if data is not None:
            default = _DEFAULT_PRICING
            if "_default" in data:
                pair = _coerce_pair(data["_default"])
                if pair is not None:
                    default = pair
            result[_DEFAULT_KEY] = default
            for key, value in data.items():
                if key == "_default":
                    continue
                pair = _coerce_pair(value)
                if pair is not None:
                    result[key] = pair

    return result


def _get_pricing(model_key: str, overrides: dict[str, PricePair]) -> PricePair:
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
        if key.startswith(_DEFAULT_KEY):
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
    return overrides.get(_DEFAULT_KEY, _DEFAULT_PRICING)


def _parse_model_str(data: dict[str, Any]) -> str:
    """Parse a model identifier from raw API response data."""
    provider = data.get("providerID", data.get("provider", "unknown"))
    model_id = data.get("id", data.get("modelID", data.get("model", "unknown")))
    variant = data.get("variant", "")
    if variant:
        return f"{provider}/{model_id}/{variant}"
    return f"{provider}/{model_id}"


def _to_epoch_seconds(value: Any) -> float | None:
    """Normalize an epoch timestamp (seconds or milliseconds) to seconds."""
    if not value:
        return None
    with contextlib.suppress(TypeError, ValueError):
        ts = float(value)
        # Values above ~1e11 are milliseconds since epoch (seconds stay below until year 5138).
        return ts / 1000 if ts > 1e11 else ts
    return None


def _cost(tokens: float, per_m: float) -> float:
    """Cost in USD for *tokens* at *per_m* dollars per million tokens."""
    return tokens * per_m / 1_000_000


def _priced(input_tokens: float, output_tokens: float, price: PricePair) -> TotalsEntry:
    """Build a totals entry from raw token counts and a pricing pair."""
    input_cost = _cost(input_tokens, price[0])
    output_cost = _cost(output_tokens, price[1])
    return {
        "input": input_tokens,
        "output": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def _aggregate(
    records: Iterable[UsageRecord], overrides: dict[str, PricePair]
) -> tuple[dict[str, TotalsEntry], dict[str, dict[str, float]], float | None, float | None]:
    """Sum usage records into per-model totals plus monthly buckets.

    Shared core for every client scanner. Records with zero input *and* output
    are skipped. Returns ``(totals, by_month, min_ts, max_ts)`` where
    ``by_month`` maps ``"YYYY-MM"`` (UTC) to summed usage and cost.
    """
    sums: dict[str, list[float]] = {}
    price_cache: dict[str, PricePair] = {}
    by_month: dict[str, dict[str, float]] = {}
    min_ts: float | None = None
    max_ts: float | None = None

    for rec in records:
        if rec.input_tokens == 0 and rec.output_tokens == 0:
            continue

        if rec.key not in sums:
            sums[rec.key] = [0.0, 0.0]
            price_cache[rec.key] = _get_pricing(rec.key, overrides)
        acc = sums[rec.key]
        acc[0] += rec.input_tokens
        acc[1] += rec.output_tokens

        if rec.ts is not None:
            if min_ts is None or rec.ts < min_ts:
                min_ts = rec.ts
            if max_ts is None or rec.ts > max_ts:
                max_ts = rec.ts

            month_key = datetime.fromtimestamp(rec.ts, tz=timezone.utc).strftime("%Y-%m")
            if month_key not in by_month:
                by_month[month_key] = {"input": 0.0, "output": 0.0, "cost": 0.0}
            price = price_cache[rec.key]
            by_month[month_key]["input"] += rec.input_tokens
            by_month[month_key]["output"] += rec.output_tokens
            by_month[month_key]["cost"] += _cost(rec.input_tokens, price[0]) + _cost(
                rec.output_tokens, price[1]
            )

    totals = {key: _priced(acc[0], acc[1], price_cache[key]) for key, acc in sums.items()}
    return totals, by_month, min_ts, max_ts


def _print_model_table(totals: dict[str, TotalsEntry], total_label: str = "Total") -> None:
    """Print the per-model usage table with a grand-total row."""
    rows: list[tuple[str, ...]] = [
        (key, _fmt(t["input"]), _fmt(t["output"]), f"${t['total_cost']:.2f}")
        for key, t in totals.items()
    ]
    grand_input = sum(t["input"] for t in totals.values())
    grand_output = sum(t["output"] for t in totals.values())
    grand_cost = sum(t["total_cost"] for t in totals.values())
    rows.append((total_label, _fmt(grand_input), _fmt(grand_output), f"${grand_cost:.2f}"))
    print()
    _print_table(("Model", "Input", "Output", "Cost"), rows, "<>>>", rule_before=len(rows) - 1)


def _print_monthly(by_month: dict[str, dict[str, float]]) -> None:
    """Print the monthly usage breakdown, if any buckets exist."""
    month_rows = [
        (month_key, _fmt(m["input"]), _fmt(m["output"]), f"${m['cost']:.2f}")
        for month_key, m in sorted(by_month.items())
    ]
    if not month_rows:
        return
    print("\nMonthly Breakdown:")
    _print_table(("Month", "Input", "Output", "Cost"), month_rows)


def _print_date_range(min_ts: float | None, max_ts: float | None) -> None:
    """Print the covered date range, if any timestamp was seen."""
    if min_ts is None or max_ts is None:
        return
    min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc)
    max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)
    days = (max_dt - min_dt).days
    print(
        f"Date range: {min_dt.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{max_dt.strftime('%Y-%m-%d %H:%M:%S')}, {_fmt(days)} day{'s' if days != 1 else ''} span"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _track_batch(entries: dict[str, PricePair]) -> None:
    """Persist multiple model usages to ``token_log.json`` in one write."""
    lock = FileLock(_LOCK_PATH, timeout=10)
    with lock:
        log = _safe_read_json(TOKEN_LOG) or {}
        for model, (input_tokens, output_tokens) in entries.items():
            if model not in log:
                log[model] = {"input": 0, "output": 0}
            log[model]["input"] += input_tokens
            log[model]["output"] += output_tokens

        # Write atomically
        tmp = TOKEN_LOG.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
        tmp.replace(TOKEN_LOG)


def track_tokens(model: str, input_tokens: float, output_tokens: float) -> None:
    """Persist token usage per model to ``token_log.json``."""
    _track_batch({model: (input_tokens, output_tokens)})


def get_totals(pricing_file: str | None = None) -> dict[str, TotalsEntry]:
    """Read and display all tracked token usage with costs.

    *pricing_file* applies the same overrides used by ``scan``.
    """
    log = _safe_read_json(TOKEN_LOG)
    if not log:
        print("No token log found.")
        return {}

    overrides = _load_pricing_overrides(pricing_file)
    totals = {
        model: _priced(
            data.get("input", 0), data.get("output", 0), _get_pricing(model, overrides)
        )
        for model, data in log.items()
    }

    _print_model_table(totals)
    return totals


def _resolve_db_path(db_path: str | None) -> Path | None:
    """Locate the opencode DB; an explicit path wins, else well-known candidates."""
    if db_path:
        db = Path(db_path)
        return db if db.exists() else None
    candidates = [
        Path.home() / ".local/share/opencode/opencode.db",
        Path.home() / "AppData/Roaming/opencode/opencode.db",
        Path("opencode.db"),
    ]
    return next((p for p in candidates if p.exists()), None)


def _opencode_record(data_text: Any, time_created: Any) -> UsageRecord | None:
    """Turn one ``message`` row into a usage record; None if unusable."""
    try:
        data = json.loads(data_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    tokens = data.get("tokens") or {}

    # _parse_model_str() already returns "provider/model_id[/variant]".
    # Do NOT prepend the provider again (that produced doubled prefixes).
    key = _parse_model_str(data)
    return UsageRecord(
        key,
        tokens.get("input", 0) or 0,
        tokens.get("output", 0) or 0,
        _to_epoch_seconds(time_created),
    )


def scan_opencode_db(
    db_path: str | None = None,
    pricing_file: str | None = None,
    obfuscate: bool = False,
    monthly: bool = True,
) -> dict[str, TotalsEntry]:
    """Scan opencode SQLite database and track token usage.

    Reads actual ``tokens.input`` / ``tokens.output`` from the ``message`` table.
    Applies pricing from *pricing_file* (with whole-segment fallback matching),
    then ``_DEFAULT_PRICING``.
    """
    db = _resolve_db_path(db_path)
    if db is None:
        print("No opencode database found.")
        return {}

    display_path = "..." + str(db)[-25:] if obfuscate else str(db)

    # Load pricing
    overrides = _load_pricing_overrides(pricing_file)

    # Open read-only so a scan can never lock or modify a live opencode.db.
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    records: list[UsageRecord] = []
    raw_count = 0
    try:
        for data_text, time_created in conn.execute("SELECT data, time_created FROM message"):
            raw_count += 1
            record = _opencode_record(data_text, time_created)
            if record is not None:
                records.append(record)
    except sqlite3.OperationalError as exc:
        print(f"Could not read the opencode database: {exc}")
        return {}
    finally:
        conn.close()

    print(f"Scanning {raw_count} messages from {display_path}...")

    totals, by_month, min_ts, max_ts = _aggregate(records, overrides)

    # Persist all models in a single locked read-modify-write.
    _track_batch({key: (t["input"], t["output"]) for key, t in totals.items()})

    # Display per-model results
    _print_model_table(totals, total_label="Total scanned")

    # Display monthly breakdown before grand total
    if monthly:
        _print_monthly(by_month)

    # Date range info
    _print_date_range(min_ts, max_ts)

    return totals


def _pi_record(line: str) -> UsageRecord | None:
    """Turn one pi session JSONL line into a usage record; None if not applicable.

    Only assistant messages carry ``usage``; pi's own cost fields are ignored
    (they are zero for local models) — pricing is applied tracker-side.
    Cache read/write counts are deliberately excluded from input/output.
    """
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or obj.get("type") != "message":
        return None

    msg = obj.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return None

    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return None

    provider = msg.get("provider", "unknown")
    model = msg.get("model", "unknown")
    return UsageRecord(
        f"{provider}/{model}",
        usage.get("input", 0) or 0,
        usage.get("output", 0) or 0,
        _to_epoch_seconds(msg.get("timestamp")),
    )


def scan_pi(
    sessions_dir: str | None = None,
    pricing_file: str | None = None,
    obfuscate: bool = False,
    monthly: bool = True,
) -> dict[str, TotalsEntry]:
    """Scan oh-my-pi session logs and track token usage.

    Reads exact ``usage.input`` / ``usage.output`` from assistant messages in
    the JSONL transcripts under *sessions_dir* (default ``~/.omp/agent/sessions``,
    one subdirectory per working directory, ``<ts>_<uuid>.jsonl`` files). Model
    keys are ``provider/model``, matching the shared pricing-key scheme.
    """
    root = Path(sessions_dir) if sessions_dir else Path.home() / ".omp" / "agent" / "sessions"
    display = "..." + str(root)[-25:] if obfuscate else str(root)

    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    if not files:
        print(f"No pi sessions found in {display}.")
        return {}

    overrides = _load_pricing_overrides(pricing_file)

    records: list[UsageRecord] = []
    raw_count = 0
    for path in files:
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    raw_count += 1
                    record = _pi_record(line)
                    if record is not None:
                        records.append(record)
        except OSError:
            print(f"Warning: skipping {path}")

    print(f"Scanning {len(files)} session files, {raw_count} lines from {display}...")

    totals, by_month, min_ts, max_ts = _aggregate(records, overrides)

    # Persist all models in a single locked read-modify-write.
    _track_batch({key: (t["input"], t["output"]) for key, t in totals.items()})

    _print_model_table(totals, total_label="Total scanned")

    if monthly:
        _print_monthly(by_month)

    _print_date_range(min_ts, max_ts)

    return totals


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
        choices=["totals", "scan", "scan-pi", "clear", "pricing"],
        help="Command to run (default: totals)",
    )
    parser.add_argument("--pricing-file", metavar="PATH", default=None, help="External JSON pricing overrides")
    parser.add_argument("--obfuscate", action="store_true", help="Truncate the data path in output")
    parser.add_argument("--no-monthly", action="store_true", help="Suppress the monthly breakdown")
    parser.add_argument("--db-path", metavar="PATH", default=None, help="Explicit path to an opencode.db file (scan)")
    parser.add_argument(
        "--sessions-dir",
        metavar="PATH",
        default=None,
        help="Directory of pi session logs (default: ~/.omp/agent/sessions)",
    )
    return parser


def _parse_cli_args(argv: list[str]) -> dict[str, Any]:
    """Parse command-line arguments into a plain dict."""
    ns = _build_parser().parse_args(argv[1:] if argv else [])
    return {
        "command": ns.command,
        "pricing_file": ns.pricing_file,
        "obfuscate": ns.obfuscate,
        "monthly": not ns.no_monthly,
        "db_path": ns.db_path,
        "sessions_dir": ns.sessions_dir,
    }


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv

    parsed = _parse_cli_args(argv)
    cmd = parsed["command"]

    if cmd == "scan":
        scan_opencode_db(
            db_path=parsed.get("db_path"),
            pricing_file=parsed.get("pricing_file"),
            obfuscate=parsed.get("obfuscate", False),
            monthly=parsed.get("monthly", True),
        )
    elif cmd == "scan-pi":
        scan_pi(
            sessions_dir=parsed.get("sessions_dir"),
            pricing_file=parsed.get("pricing_file"),
            obfuscate=parsed.get("obfuscate", False),
            monthly=parsed.get("monthly", True),
        )
    elif cmd == "clear":
        clear_log()
    elif cmd == "pricing":
        _list_pricing_options()
    else:
        get_totals(pricing_file=parsed.get("pricing_file"))


def _list_pricing_options() -> None:
    """List all available pricing configurations."""
    print("Pricing resolution order (highest to lowest):\n")
    print("  1. --pricing-file <path>: exact model-key match, then most-specific whole-segment match")
    print('  2. "_default" key inside the pricing file: fallback within the file')
    print("  3. Built-in default fallback\n")

    print(f"Default (no match): ${_DEFAULT_PRICING[0]:.2f}/M input, ${_DEFAULT_PRICING[1]:.2f}/M output\n")
    print('Pricing file format: {"_default": [in_per_m, out_per_m], "provider/model": [in_per_m, out_per_m]}\n')


if __name__ == "__main__":
    main()
