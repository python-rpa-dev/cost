"""Token usage tracker for LLM client harnesses (opencode, oh-my-pi)."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from filelock import FileLock

TOKEN_LOG = Path("data") / "token_log.json"
_LOCK_PATH = str(TOKEN_LOG) + ".lock"


_DEFAULT_PRICING: tuple[float, float] = (0.03, 0.05)
#: Ledger format version, written to ``_meta.schema`` on every write.
#: Absent means a pre-0.5 ledger (watermark + months only). Readers must
#: ignore unknown ``_meta`` keys; future migrations key off this value.
_LEDGER_SCHEMA = 1

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

    ``ts`` is epoch seconds (already normalized from ms) or ``None``;
    ``source`` names the client ("opencode", "oh-my-pi").
    """

    key: str
    input_tokens: float
    output_tokens: float
    ts: float | None
    source: str = ""


class Location(NamedTuple):
    """One declared scan source: label namespaces watermarks, kind is "opencode"|"pi"."""

    label: str
    kind: str
    path: str


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


def _load_locations(path: str) -> list[Location]:
    """Parse a locations JSON file into declared scan sources; warn-and-skip invalid ones.

    Format: ``{"locations": [{"label": ..., "type": "opencode"|"pi", "path": ...}, ...]}``.
    A label must be a non-empty string without ``@`` so ``source@label`` watermark
    keys stay unambiguous; duplicate labels keep the first occurrence.
    """
    data = _safe_read_json(Path(path))
    if data is None:
        print(f"Could not read locations file: {path}")
        return []
    entries = data.get("locations") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        print(f'Locations file must be {{"locations": [...]}}, got {path}')
        return []

    locations: list[Location] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            print(f"Skipping invalid location entry: {entry!r}")
            continue
        label, kind, loc_path = entry.get("label"), entry.get("type"), entry.get("path")
        if (
            not isinstance(label, str)
            or not label
            or "@" in label
            or kind not in ("opencode", "pi")
            or not isinstance(loc_path, str)
            or not loc_path
        ):
            print(f"Skipping invalid location entry: {entry!r}")
            continue
        if label in seen:
            print(f"Duplicate location label {label!r}; keeping the first.")
            continue
        seen.add(label)
        locations.append(Location(label, kind, loc_path))
    return locations


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

    def rank(key: str) -> tuple[int, int, str]:
        parts = key.split("/")
        return (len(parts), len(key), key)

    matches = []
    for key in overrides:
        if key.startswith(_DEFAULT_KEY):
            continue
        key_parts = key.split("/")
        is_prefix = len(key_parts) <= len(model_parts) and all(
            k == m for k, m in zip(key_parts, model_parts)
        )
        is_contained = all(k in model_parts for k in key_parts)
        if is_prefix or is_contained:
            matches.append(key)

    if matches:
        best = max(matches, key=rank)
        best_segs = len(best.split("/"))
        if sum(1 for k in matches if len(k.split("/")) == best_segs) > 1:
            print(f"Warning: ambiguous pricing for {model_key}; using {best!r}.")
        return overrides[best]
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
) -> tuple[dict[tuple[str, str], TotalsEntry], dict[str, dict[str, float]], float | None, float | None]:
    """Sum usage records into per-(source, model) totals plus monthly buckets.

    Shared core for every client scanner. Records with zero input *and* output
    are skipped. Returns ``(totals, by_month, min_ts, max_ts)`` where
    ``by_month`` maps ``"YYYY-MM"`` (UTC) to summed usage and cost.
    """
    sums: dict[tuple[str, str], list[float]] = {}
    price_cache: dict[str, PricePair] = {}
    by_month: dict[str, dict[str, float]] = {}
    min_ts: float | None = None
    max_ts: float | None = None

    for rec in records:
        if rec.input_tokens == 0 and rec.output_tokens == 0:
            continue

        ident = (rec.source, rec.key)
        if ident not in sums:
            sums[ident] = [0.0, 0.0]
        price_cache.setdefault(rec.key, _get_pricing(rec.key, overrides))
        acc = sums[ident]
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

    totals = {ident: _priced(acc[0], acc[1], price_cache[key]) for ident, acc in sums.items() for key in (ident[1],)}
    return totals, by_month, min_ts, max_ts


def _scan_report(
    records: list[UsageRecord],
    overrides: dict[str, PricePair],
    monthly: bool,
    since_ts: float | None = None,
    until_ts: float | None = None,
    scan_started: float | None = None,
) -> dict[tuple[str, str], TotalsEntry]:
    """Aggregate, persist, and render one scan; totals keyed by ``(source, model)``.

    Only records newer than the per-source watermark stored under ``_meta`` in
    the ledger are tracked, so re-scanning never double-counts; each source's
    watermark advances to its newest persisted timestamp; a windowed scan
    (*since_ts*/*until_ts*, UTC epoch seconds, start inclusive, end exclusive)
    never advances watermarks, so a later full scan still picks up records the
    window skipped — at the cost of re-counting records inside the window.
    Records without a timestamp count once, on the first scan of their source.

    The ledger is persisted per model (sources merged); the table carries a
    Source column so combined scans stay attributable.
    """
    if since_ts is not None or until_ts is not None:
        records = [r for r in records if _in_window(r.ts, since_ts, until_ts)]

    _ensure_data_dir()
    with FileLock(_LOCK_PATH, timeout=10):
        log = _safe_read_json(TOKEN_LOG) or {}
        watermarks = _get_watermarks(log)

        fresh: list[UsageRecord] = []
        skipped: dict[str, int] = {}
        for rec in records:
            watermark = watermarks.get(rec.source)
            if rec.ts is None:
                # No timestamp to gate on: count once, on the source's first scan.
                if rec.source in watermarks:
                    skipped[rec.source] = skipped.get(rec.source, 0) + 1
                else:
                    fresh.append(rec)
            elif watermark is not None and rec.ts <= watermark:
                skipped[rec.source] = skipped.get(rec.source, 0) + 1
            else:
                fresh.append(rec)

        totals, by_month, min_ts, max_ts = _aggregate(fresh, overrides)

        changed = False
        for (_src, model), t in totals.items():
            entry = log.setdefault(model, {"input": 0, "output": 0})
            if isinstance(entry, dict):
                entry["input"] += t["input"]
                entry["output"] += t["output"]
                changed = True
        new_marks: dict[str, float] = {}
        for rec in fresh:
            if rec.ts is not None:
                new_marks[rec.source] = max(new_marks.get(rec.source, rec.ts), rec.ts)
            else:
                # Undated records count once per source: mark the source seen.
                new_marks.setdefault(rec.source, 0)
        new_scans: dict[str, float] = {}
        if scan_started is None:
            scan_started = time.time()
        # A windowed scan never saw the source's full history, so it must not
        # advance watermarks: records below the window were never tracked and a
        # later full scan would skip them forever. Re-counting inside the window
        # on a later full scan is the accepted cost.
        if since_ts is not None or until_ts is not None:
            new_marks = {}
        else:
            # Full scan: every file/row was read at scan_started, so later
            # collectors may skip sources untouched since then (_meta.last_scan).
            for src in {rec.source for rec in records}:
                new_scans[src] = scan_started
        if changed or new_marks or new_scans:
            meta = log.setdefault("_meta", {})
            marks = meta.setdefault("watermark", {}) if isinstance(meta, dict) else None
            if isinstance(marks, dict):
                meta["schema"] = _LEDGER_SCHEMA
                for src, ts in new_marks.items():
                    prev = marks.get(src)
                    if not isinstance(prev, (int, float)) or isinstance(prev, bool) or ts > prev:
                        marks[src] = ts
                scans = meta.setdefault("last_scan", {})
                if isinstance(scans, dict):
                    scans.update(new_scans)
                buckets = meta.setdefault("months", {})
                if isinstance(buckets, dict):
                    _accumulate_months(buckets, fresh)
                _write_log(log)
            else:
                print("Warning: ledger _meta is malformed; usage tracked without watermark.")
                _write_log({k: v for k, v in log.items() if k != "_meta"})

    for src, n in sorted(skipped.items()):
        print(f"Skipped {_fmt(n)} already-tracked record(s) from {src}.")
    if not fresh:
        print("No new usage to track.")
        return {}

    rows: list[tuple[str, ...]] = [
        (src, model, _fmt(t["input"]), _fmt(t["output"]), f"${t['total_cost']:.2f}")
        for (src, model), t in totals.items()
    ]
    grand_input = sum(t["input"] for t in totals.values())
    grand_output = sum(t["output"] for t in totals.values())
    grand_cost = sum(t["total_cost"] for t in totals.values())
    rows.append(("Total scanned", "", _fmt(grand_input), _fmt(grand_output), f"${grand_cost:.2f}"))
    print()
    _print_table(
        ("Source", "Model", "Input", "Output", "Cost"),
        rows,
        "<<>>>",
        rule_before=len(rows) - 1,
    )

    if monthly:
        _print_monthly(by_month)
    _print_date_range(min_ts, max_ts)
    return totals


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


def _ensure_data_dir() -> None:
    """Create the directory holding the token log and its lock file."""
    Path(_LOCK_PATH).parent.mkdir(parents=True, exist_ok=True)


def _write_log(log: dict[str, Any]) -> None:
    """Atomically replace the token log file."""
    TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_LOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
    tmp.replace(TOKEN_LOG)


def _get_watermarks(log: dict[str, Any]) -> dict[str, float]:
    """Extract per-source watermarks from the ledger's ``_meta`` section."""
    meta = log.get("_meta")
    raw = meta.get("watermark") if isinstance(meta, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {
        src: ts
        for src, ts in raw.items()
        if isinstance(ts, (int, float)) and not isinstance(ts, bool)
    }


def _in_window(ts: float | None, since_ts: float | None, until_ts: float | None) -> bool:
    """True if *ts* lies in the ``[since_ts, until_ts)`` window (None = open end)."""
    if ts is None:
        return False
    if since_ts is not None and ts < since_ts:
        return False
    return until_ts is None or ts < until_ts


def _detect_env() -> str:
    """Best-effort environment label: ``wsl``, ``windows``, or ``local``."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return "wsl"
    if sys.platform == "win32":
        return "windows"
    return "local"


def _tag_env(records: list[UsageRecord], env: str) -> list[UsageRecord]:
    """Suffix each record's source with the environment, e.g. ``opencode@wsl``."""
    return [rec._replace(source=f"{rec.source}@{env}") for rec in records]


def _collect_location(loc: Location, obfuscate: bool) -> list[UsageRecord] | None:
    """Collect one declared location, tagged ``source@<label>``; None if source absent.

    Passing ``loc.label`` as ``_collect_pi``'s env label makes its mtime
    prefilter read ``_meta.last_scan["oh-my-pi@<label>"]``, matching the
    tagged source key.
    """
    if loc.kind == "opencode":
        db = Path(loc.path)
        if not db.exists():
            print(f"Location {loc.label!r}: no opencode database at {loc.path}.")
            return None
        recs = _opencode_records(db, obfuscate)
    else:
        recs = _collect_pi(loc.path, obfuscate, loc.label)
    if recs is None:
        return None
    return _tag_env(recs, loc.label)


def _accumulate_months(buckets: dict[str, Any], records: list[UsageRecord]) -> None:
    """Accumulate raw per-month/per-model tokens into the ledger's ``_meta.months``.

    Buckets hold unpriced token counts (``month -> model -> [input, output]``);
    costs are computed at display time so price changes reprice history.
    Undated and zero-usage records are not bucketed.
    """
    for rec in records:
        if rec.ts is None or (rec.input_tokens == 0 and rec.output_tokens == 0):
            continue
        month_key = datetime.fromtimestamp(rec.ts, tz=timezone.utc).strftime("%Y-%m")
        entry = buckets.setdefault(month_key, {})
        if not isinstance(entry, dict):
            continue
        prev = entry.get(rec.key)
        pair = prev if isinstance(prev, list) and len(prev) == 2 else [0, 0]
        entry[rec.key] = [pair[0] + rec.input_tokens, pair[1] + rec.output_tokens]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _track_batch(entries: dict[str, PricePair]) -> None:
    """Persist multiple model usages to the token log in one write."""
    _ensure_data_dir()
    lock = FileLock(_LOCK_PATH, timeout=10)
    with lock:
        log = _safe_read_json(TOKEN_LOG) or {}
        for model, (input_tokens, output_tokens) in entries.items():
            entry = log.setdefault(model, {"input": 0, "output": 0})
            if isinstance(entry, dict):
                entry["input"] += input_tokens
                entry["output"] += output_tokens
        meta = log.setdefault("_meta", {})
        if isinstance(meta, dict):
            meta["schema"] = _LEDGER_SCHEMA
        _write_log(log)


def track_tokens(model: str, input_tokens: float, output_tokens: float) -> None:
    """Persist token usage per model to ``token_log.json``."""
    _track_batch({model: (input_tokens, output_tokens)})


def _monthly_from_ledger(
    log: dict[str, Any], overrides: dict[str, PricePair]
) -> dict[str, dict[str, float]]:
    """Reprice the ledger's stored month buckets with current pricing."""
    meta = log.get("_meta")
    months = meta.get("months") if isinstance(meta, dict) else None
    if not isinstance(months, dict):
        return {}

    by_month: dict[str, dict[str, float]] = {}
    for month_key in sorted(months):
        per_model = months[month_key]
        if not isinstance(per_model, dict):
            continue
        bucket = {"input": 0.0, "output": 0.0, "cost": 0.0}
        for model_key, pair in per_model.items():
            if not (isinstance(pair, list) and len(pair) == 2):
                continue
            try:
                input_tokens = float(pair[0])
                output_tokens = float(pair[1])
            except (TypeError, ValueError):
                continue
            t = _priced(input_tokens, output_tokens, _get_pricing(model_key, overrides))
            bucket["input"] += input_tokens
            bucket["output"] += output_tokens
            bucket["cost"] += t["total_cost"]
        by_month[str(month_key)] = bucket
    return by_month


def get_totals(pricing_file: str | None = None) -> dict[str, TotalsEntry]:
    """Read and display all tracked token usage with costs.

    *pricing_file* applies the same overrides used by the scanners. When the
    ledger carries month buckets (written by scans), a cumulative monthly
    breakdown is printed below the model table.
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
        if not model.startswith("_") and isinstance(data, dict)
    }

    _print_model_table(totals)
    by_month = _monthly_from_ledger(log, overrides)
    if by_month:
        _print_monthly(by_month)
    return totals


def export_csv(pricing_file: str | None = None) -> None:
    """Print the tracked ledger as CSV on stdout, priced like ``get_totals``."""
    log = _safe_read_json(TOKEN_LOG)
    if not log:
        print("No token log found.")
        return

    overrides = _load_pricing_overrides(pricing_file)
    writer = csv.writer(sys.stdout)
    writer.writerow(["model", "input", "output", "input_cost", "output_cost", "total_cost"])
    for model, data in log.items():
        if model.startswith("_") or not isinstance(data, dict):
            continue
        t = _priced(
            data.get("input", 0), data.get("output", 0), _get_pricing(model, overrides)
        )
        writer.writerow(
            [
                model,
                int(t["input"]),
                int(t["output"]),
                f"{t['input_cost']:.6f}",
                f"{t['output_cost']:.6f}",
                f"{t['total_cost']:.6f}",
            ]
        )


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
        "opencode",
    )


def _opencode_records(db: Path, obfuscate: bool) -> list[UsageRecord] | None:
    """Read usage records from one opencode DB file; None if unreadable."""
    display_path = "..." + str(db)[-25:] if obfuscate else str(db)

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
        return None
    finally:
        conn.close()

    print(f"Scanning {raw_count} messages from {display_path}...")
    return records


def _collect_opencode(db_path: str | None, obfuscate: bool) -> list[UsageRecord] | None:
    """Gather usage records from the auto-detected opencode DB; None if absent."""
    db = _resolve_db_path(db_path)
    if db is None:
        print("No opencode database found.")
        return None
    return _opencode_records(db, obfuscate)


def scan_opencode_db(
    db_path: str | None = None,
    pricing_file: str | None = None,
    obfuscate: bool = False,
    monthly: bool = True,
    since_ts: float | None = None,
    until_ts: float | None = None,
    env: str | None = None,
    locations: list[Location] | None = None,
) -> dict[str, TotalsEntry]:
    """Scan opencode SQLite database and track token usage.

    Reads actual ``tokens.input`` / ``tokens.output`` from the ``message`` table.
    Applies pricing from *pricing_file* (with whole-segment fallback matching),
    then ``_DEFAULT_PRICING``. Records are tagged ``opencode@<env>`` (*env*
    defaults to auto-detection: ``wsl``/``windows``/``local``) so each OS's own
    database keeps an independent watermark in a shared ledger. Only usage
    newer than that watermark is tracked; *since_ts*/*until_ts* bound a UTC
    epoch-seconds window (start inclusive, end exclusive) and never advance
    the watermark. *locations* adds declared ``opencode`` sources (each tagged
    with its own label) on top of the auto-detected one. Returns the newly
    tracked totals.
    """
    t0 = time.time()  # before collection: writes after this stamp stay visible
    auto = _collect_opencode(db_path, obfuscate)
    records = _tag_env(auto, env or _detect_env()) if auto else []
    for loc in locations or []:
        if loc.kind == "opencode":
            recs = _collect_location(loc, obfuscate)
            if recs is not None:
                records.extend(recs)
    if not records:
        return {}

    overrides = _load_pricing_overrides(pricing_file)
    totals = _scan_report(records, overrides, monthly, since_ts, until_ts, scan_started=t0)
    return {model: t for (_src, model), t in totals.items()}


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
        "oh-my-pi",
    )


def _collect_pi(sessions_dir: str | None, obfuscate: bool, env_label: str) -> list[UsageRecord] | None:
    """Gather usage records from pi session logs; None if no source is present.

    Files whose modification time predates the source's last full scan
    (``_meta.last_scan``) were already read completely at that time, so they
    are skipped without reading — steady-state scans only touch freshly
    written sessions.
    """
    root = Path(sessions_dir) if sessions_dir else Path.home() / ".omp" / "agent" / "sessions"
    display = "..." + str(root)[-25:] if obfuscate else str(root)

    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    if not files:
        print(f"No pi sessions found in {display}.")
        return None

    meta = (_safe_read_json(TOKEN_LOG) or {}).get("_meta")
    scans = meta.get("last_scan") if isinstance(meta, dict) else None
    cutoff = scans.get(f"oh-my-pi@{env_label}") if isinstance(scans, dict) else None
    skipped_files = 0
    if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool):
        fresh_files = []
        for path in files:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = float("inf")  # unreadable stat: read the file, let the gate decide
            if mtime > cutoff:
                fresh_files.append(path)
        skipped_files = len(files) - len(fresh_files)
        files = fresh_files

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

    note = f" ({skipped_files} unchanged since last scan)" if skipped_files else ""
    print(f"Scanning {len(files)} session files, {raw_count} lines from {display}{note}...")
    return records


def scan_pi(
    sessions_dir: str | None = None,
    pricing_file: str | None = None,
    obfuscate: bool = False,
    monthly: bool = True,
    since_ts: float | None = None,
    until_ts: float | None = None,
    env: str | None = None,
    locations: list[Location] | None = None,
) -> dict[str, TotalsEntry]:
    """Scan oh-my-pi session logs and track token usage.

    Reads exact ``usage.input`` / ``usage.output`` from assistant messages in
    the JSONL transcripts under *sessions_dir* (default ``~/.omp/agent/sessions``,
    one subdirectory per working directory, ``<ts>_<uuid>.jsonl`` files). Model
    keys are ``provider/model``, matching the shared pricing-key scheme. Records
    are tagged ``oh-my-pi@<env>`` (see :func:`scan_opencode_db`); each OS's own
    session directory keeps an independent watermark. Only usage newer than
    that watermark is tracked; *since_ts*/*until_ts* bound a UTC epoch-seconds
    window (start inclusive, end exclusive) and never advance the watermark.
    *locations* adds declared ``pi`` sources (each tagged with its own label)
    on top of the auto-detected one. Returns the newly tracked totals.
    """
    t0 = time.time()  # before collection: writes after this stamp stay visible
    env_label = env or _detect_env()
    auto = _collect_pi(sessions_dir, obfuscate, env_label)
    records = _tag_env(auto, env_label) if auto else []
    for loc in locations or []:
        if loc.kind == "pi":
            recs = _collect_location(loc, obfuscate)
            if recs is not None:
                records.extend(recs)
    if not records:
        return {}

    overrides = _load_pricing_overrides(pricing_file)
    totals = _scan_report(records, overrides, monthly, since_ts, until_ts, scan_started=t0)
    return {model: t for (_src, model), t in totals.items()}


def scan_all(
    db_path: str | None = None,
    sessions_dir: str | None = None,
    pricing_file: str | None = None,
    obfuscate: bool = False,
    monthly: bool = True,
    since_ts: float | None = None,
    until_ts: float | None = None,
    env: str | None = None,
    locations: list[Location] | None = None,
) -> dict[tuple[str, str], TotalsEntry]:
    """Scan every known client source and report usage in one Source-tagged table.

    Combines the opencode DB and oh-my-pi session logs; missing sources are
    reported and skipped. Records carry ``source@<env>`` labels (see
    :func:`scan_opencode_db`), so scans from several environments aggregate
    into one ledger with independent watermarks per source *and* environment.
    *locations* adds declared sources (each tagged with its own label) on top
    of the auto-detected ones. Only usage newer than each watermark is tracked
    (*since_ts*/*until_ts* additionally bound a UTC epoch-seconds window and
    never advance the watermarks). Returns newly tracked totals keyed by
    ``(source@env, model)``; the ledger persists per model (sources merged).
    """
    records: list[UsageRecord] = []
    found = False
    env_label = env or _detect_env()
    t0 = time.time()  # before collection: writes after this stamp stay visible

    oc_records = _collect_opencode(db_path, obfuscate)
    if oc_records is not None:
        records.extend(_tag_env(oc_records, env_label))
        found = True
    pi_records = _collect_pi(sessions_dir, obfuscate, env_label)
    if pi_records is not None:
        records.extend(_tag_env(pi_records, env_label))
        found = True
    for loc in locations or []:
        recs = _collect_location(loc, obfuscate)
        if recs is not None:
            records.extend(recs)
            found = True

    if not found:
        print("No data sources found.")
        return {}

    overrides = _load_pricing_overrides(pricing_file)
    return _scan_report(records, overrides, monthly, since_ts, until_ts, scan_started=t0)


def merge_ledger(path: str, label: str) -> None:
    """Fold a foreign token_tracker ledger into the local one, idempotently.

    Model totals are added; the foreign ``watermark``/``last_scan`` keys are
    suffixed with ``@<label>`` so a foreign machine's ``opencode@wsl`` stays
    distinct from the local one (keys already ending in ``@<label>`` come from
    a previous merge and are kept as-is). Month buckets are added per
    month/model. The foreign file's SHA-256 is recorded under
    ``_meta.merged``; re-merging identical content is a no-op.
    """
    foreign_path = Path(path)
    if foreign_path.resolve() == TOKEN_LOG.resolve():
        print("Cannot merge the ledger into itself.")
        return
    foreign = _safe_read_json(foreign_path)
    if foreign is None:
        print(f"Could not read foreign ledger: {path}")
        return
    if not isinstance(foreign, dict):
        print("Foreign ledger is not a JSON object.")
        return
    models = {k: v for k, v in foreign.items() if not k.startswith("_") and isinstance(v, dict)}
    if not models:
        print("Foreign ledger has no usage.")
        return

    try:
        digest = hashlib.sha256(foreign_path.read_bytes()).hexdigest()
    except OSError as exc:
        print(f"Could not read foreign ledger: {path} ({exc})")
        return

    _ensure_data_dir()
    with FileLock(_LOCK_PATH, timeout=10):
        log = _safe_read_json(TOKEN_LOG) or {}
        meta = log.setdefault("_meta", {})
        if not isinstance(meta, dict):
            meta = {"watermark": {}, "last_scan": {}, "months": {}, "merged": []}
            log["_meta"] = meta
        merged = meta.setdefault("merged", [])
        if isinstance(merged, list) and any(
            isinstance(m, dict) and m.get("sha256") == digest for m in merged
        ):
            print("Foreign ledger already merged (same content); skipping.")
            return

        count = 0
        for model, data in models.items():
            with contextlib.suppress(TypeError, ValueError):
                in_tok, out_tok = float(data.get("input", 0)), float(data.get("output", 0))
                entry = log.setdefault(model, {"input": 0, "output": 0})
                if isinstance(entry, dict):
                    entry["input"] += in_tok
                    entry["output"] += out_tok
                    count += 1

        def relabel(keys: dict[str, Any]) -> dict[str, Any]:
            return {k if k.endswith(f"@{label}") else f"{k}@{label}": v for k, v in keys.items()}

        for meta_key in ("watermark", "last_scan"):
            foreign_map = foreign.get("_meta", {}).get(meta_key) if isinstance(foreign.get("_meta"), dict) else None
            if not isinstance(foreign_map, dict):
                continue
            local_map = meta.setdefault(meta_key, {})
            if not isinstance(local_map, dict):
                continue
            for key, value in relabel(foreign_map).items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                prev = local_map.get(key)
                if not isinstance(prev, (int, float)) or isinstance(prev, bool) or value > prev:
                    local_map[key] = value

        foreign_months = foreign.get("_meta", {}).get("months") if isinstance(foreign.get("_meta"), dict) else None
        if isinstance(foreign_months, dict):
            local_months = meta.setdefault("months", {})
            if isinstance(local_months, dict):
                for month, per_model in foreign_months.items():
                    if not isinstance(per_model, dict):
                        continue
                    bucket = local_months.setdefault(month, {})
                    if not isinstance(bucket, dict):
                        continue
                    for model_key, pair in per_model.items():
                        if not (isinstance(pair, list) and len(pair) == 2):
                            continue
                        try:
                            add = [float(pair[0]), float(pair[1])]
                        except (TypeError, ValueError):
                            continue
                        prev = bucket.get(model_key)
                        base = prev if isinstance(prev, list) and len(prev) == 2 else [0, 0]
                        bucket[model_key] = [base[0] + add[0], base[1] + add[1]]

        meta["schema"] = _LEDGER_SCHEMA
        if isinstance(merged, list):
            merged.append({"sha256": digest, "file": foreign_path.name, "at": int(time.time())})
        _write_log(log)
    print(f"Merged {count} model(s) from {path} as {label}.")


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


def _date_arg(value: str) -> float:
    """argparse type: ``YYYY-MM-DD`` (UTC calendar date) to epoch seconds."""
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc
    return dt.replace(tzinfo=timezone.utc).timestamp()


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
        choices=["totals", "scan-oc", "scan-pi", "scan-all", "clear", "pricing", "export", "merge"],
        help="Command to run (default: totals)",
    )
    parser.add_argument(
        "merge_path",
        nargs="?",
        metavar="LEDGER",
        default=None,
        help="Foreign ledger file for the merge command",
    )
    parser.add_argument("--pricing-file", metavar="PATH", default=None, help="External JSON pricing overrides")
    parser.add_argument("--obfuscate", action="store_true", help="Truncate the data path in output")
    parser.add_argument("--no-monthly", action="store_true", help="Suppress the monthly breakdown")
    parser.add_argument("--db-path", metavar="PATH", default=None, help="Explicit path to an opencode.db file (scan-oc/scan-all)")
    parser.add_argument(
        "--sessions-dir",
        metavar="PATH",
        default=None,
        help="Directory of pi session logs for scan-pi/scan-all (default: ~/.omp/agent/sessions)",
    )
    parser.add_argument(
        "--since",
        dest="since",
        metavar="DATE",
        type=_date_arg,
        default=None,
        help="Track only usage on/after this UTC date (scanners)",
    )
    parser.add_argument(
        "--until",
        dest="until",
        metavar="DATE",
        type=_date_arg,
        default=None,
        help="Track only usage through this UTC date, inclusive (scanners)",
    )
    parser.add_argument(
        "--env",
        metavar="NAME",
        default=None,
        help="Environment label appended to scan sources and watermarks (default: auto-detect wsl/windows/local)",
    )
    parser.add_argument(
        "--locations",
        metavar="PATH",
        default=None,
        help="JSON file of declared scan locations (scan-oc/scan-pi/scan-all)",
    )
    parser.add_argument(
        "--as",
        dest="as_label",
        metavar="LABEL",
        default=None,
        help="Label namespaces a merged foreign ledger's watermark keys",
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
        "env": ns.env,
        "locations_file": ns.locations,
        "merge_path": ns.merge_path,
        "as_label": ns.as_label,
        "since_ts": ns.since,
        # --until names an inclusive calendar day; the window end is exclusive.
        "until_ts": None if ns.until is None else ns.until + 86_400,
    }


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv

    parsed = _parse_cli_args(argv)
    cmd = parsed["command"]
    locations = _load_locations(parsed["locations_file"]) if parsed["locations_file"] else []

    if cmd == "scan-oc":
        scan_opencode_db(
            db_path=parsed["db_path"],
            pricing_file=parsed["pricing_file"],
            obfuscate=parsed["obfuscate"],
            monthly=parsed["monthly"],
            since_ts=parsed["since_ts"],
            until_ts=parsed["until_ts"],
            env=parsed["env"],
            locations=locations,
        )
    elif cmd == "scan-pi":
        scan_pi(
            sessions_dir=parsed["sessions_dir"],
            pricing_file=parsed["pricing_file"],
            obfuscate=parsed["obfuscate"],
            monthly=parsed["monthly"],
            since_ts=parsed["since_ts"],
            until_ts=parsed["until_ts"],
            env=parsed["env"],
            locations=locations,
        )
    elif cmd == "scan-all":
        scan_all(
            db_path=parsed["db_path"],
            sessions_dir=parsed["sessions_dir"],
            pricing_file=parsed["pricing_file"],
            obfuscate=parsed["obfuscate"],
            monthly=parsed["monthly"],
            since_ts=parsed["since_ts"],
            until_ts=parsed["until_ts"],
            env=parsed["env"],
            locations=locations,
        )
    elif cmd == "merge":
        if not parsed["merge_path"]:
            print("merge requires a ledger path.")
        elif not parsed["as_label"]:
            print("merge requires --as LABEL.")
        else:
            merge_ledger(parsed["merge_path"], parsed["as_label"])
    elif cmd == "clear":
        clear_log()
    elif cmd == "pricing":
        _list_pricing_options(parsed["pricing_file"])
    elif cmd == "export":
        export_csv(pricing_file=parsed["pricing_file"])
    else:
        get_totals(pricing_file=parsed["pricing_file"])


def _list_pricing_options(pricing_file: str | None = None) -> None:
    """List all available pricing configurations; with *pricing_file*, also its entries."""
    print("Pricing resolution order (highest to lowest):\n")
    print("  1. --pricing-file <path>: exact model-key match, then most-specific whole-segment match")
    print('  2. "_default" key inside the pricing file: fallback within the file')
    print("  3. Built-in default fallback\n")

    print(f"Default (no match): ${_DEFAULT_PRICING[0]:.2f}/M input, ${_DEFAULT_PRICING[1]:.2f}/M output\n")
    print('Pricing file format: {"_default": [in_per_m, out_per_m], "provider/model": [in_per_m, out_per_m]}\n')

    if pricing_file is not None:
        overrides = _load_pricing_overrides(pricing_file)
        print(f"Entries from {pricing_file}:\n")
        for key in sorted(overrides):
            pair = overrides[key]
            label = "_default (file fallback)" if key == _DEFAULT_KEY else key
            print(f"  {label}: ${pair[0]:.2f}/M input, ${pair[1]:.2f}/M output")


if __name__ == "__main__":
    main()
