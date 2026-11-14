"""Tests for token_tracker module."""

import json
import os
import sqlite3
from typing import ClassVar
from unittest.mock import patch

import pytest

from token_tracker import (
    _DEFAULT_PRICING,
    _LEDGER_SCHEMA,
    Location,
    _detect_env,
    _fmt,
    _get_pricing,
    _list_pricing_options,
    _load_locations,
    _load_pricing_overrides,
    _parse_cli_args,
    _parse_model_str,
    _safe_read_json,
    clear_log,
    export_csv,
    get_totals,
    main,
    merge_ledger,
    scan_all,
    scan_opencode_db,
    scan_pi,
    track_tokens,
)


def _make_db(tmp_path, rows, timestamps=None):
    """Create a temp opencode-style SQLite DB with the given message rows."""
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE message (data TEXT, time_created INTEGER)")
        ts_list = timestamps if timestamps is not None else [1_700_000_000] * len(rows)
        for r, ts in zip(rows, ts_list):
            payload = json.dumps(r) if not isinstance(r, str) else r
            conn.execute(
                "INSERT INTO message (data, time_created) VALUES (?, ?)",
                (payload, ts),
            )
        conn.commit()
    finally:
        conn.close()
    return db


class TestFmt:
    def test_small_number(self):
        assert _fmt(1234) == "1'234"

    def test_large_number(self):
        assert _fmt(1234567890) == "1'234'567'890"

    def test_zero(self):
        assert _fmt(0) == "0"


class TestSafeReadJson:
    def test_nonexistent_file(self, tmp_path):
        result = _safe_read_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_json(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        result = _safe_read_json(f)
        assert result == {"key": "value"}

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "invalid.json"
        f.write_text("{invalid json}")
        result = _safe_read_json(f)
        assert result is None


class TestLoadPricingOverrides:
    def test_no_file(self):
        result = _load_pricing_overrides(None)
        assert result == {}

    def test_nonexistent_file(self):
        result = _load_pricing_overrides("/nonexistent/path.json")
        assert result == {}

    def test_default_pricing(self, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text('{"_default": [2.0, 10.0]}')
        result = _load_pricing_overrides(str(f))
        assert result["__default"] == (2.0, 10.0)

    def test_model_override(self, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text('{"my-model": [5.0, 20.0]}')
        result = _load_pricing_overrides(str(f))
        assert result["my-model"] == (5.0, 20.0)

    def test_combined(self, tmp_path):
        f = tmp_path / "pricing.json"
        f.write_text('{"_default": [1.0, 3.0], "gpt-4": [10.0, 30.0]}')
        result = _load_pricing_overrides(str(f))
        assert result["__default"] == (1.0, 3.0)
        assert result["gpt-4"] == (10.0, 30.0)


class TestGetPricing:
    def test_known_model(self):
        result = _get_pricing("openai/gpt-4o", {})
        assert result == (0.03, 0.05)

    def test_override_takes_precedence(self):
        overrides = {"my-model": (5.0, 25.0)}
        result = _get_pricing("my-model", overrides)
        assert result == (5.0, 25.0)

    def test_fallback_default(self):
        overrides = {"__default": (3.0, 15.0)}
        result = _get_pricing("unknown-model", overrides)
        assert result == (3.0, 15.0)

    def test_global_default_when_no_match(self):
        result = _get_pricing("completely-new-model", {})
        assert result == _DEFAULT_PRICING

    def test_path_prefix_match(self):
        # "lmstudio_remote/qwen" is a leading path prefix of the model key.
        overrides = {"lmstudio_remote/qwen": (3.0, 4.0)}
        assert _get_pricing("lmstudio_remote/qwen/qwen3.6-35b-a3b", overrides) == (3.0, 4.0)

    def test_segment_contained_match(self):
        # A single-segment override matches if it appears as a whole segment.
        overrides = {"qwen": (5.0, 6.0)}
        assert _get_pricing("lmstudio_remote/qwen/qwen3.6-35b-a3b", overrides) == (5.0, 6.0)

    def test_tie_prefers_longer_key_and_warns(self, capsys):
        # Equal segment counts: the longer key wins deterministically (not by
        # file order) and the ambiguity is surfaced.
        overrides = {"beta": (2.0, 2.0), "alpha": (1.0, 1.0)}
        assert _get_pricing("alpha/beta/x", overrides) == (1.0, 1.0)
        assert "ambiguous pricing for alpha/beta/x" in capsys.readouterr().out

    def test_no_substring_false_positive(self):
        # "gpt" must NOT match "gpt2" (whole-segment matching, not substring).
        overrides = {"openai/gpt": (7.0, 8.0)}
        assert _get_pricing("openai/gpt2", overrides) == _DEFAULT_PRICING

    def test_most_specific_wins(self):
        overrides = {
            "qwen": (1.0, 1.0),
            "lmstudio_remote/qwen": (2.0, 2.0),
        }
        assert _get_pricing("lmstudio_remote/qwen/qwen3.6-35b-a3b", overrides) == (2.0, 2.0)


class TestParseModelStr:
    def test_basic(self):
        data = {"providerID": "openai", "id": "gpt-4"}
        result = _parse_model_str(data)
        assert result == "openai/gpt-4"

    def test_with_variant(self):
        data = {"providerID": "anthropic", "id": "claude-3", "variant": "sonnet"}
        result = _parse_model_str(data)
        assert result == "anthropic/claude-3/sonnet"

    def test_missing_provider(self):
        data = {"id": "gpt-4"}
        result = _parse_model_str(data)
        assert result == "unknown/gpt-4"


class TestTrackTokens:
    def test_track_and_read(self, tmp_path):
        log_file = tmp_path / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log_file):
            track_tokens("test-model", 100, 50)
            data = json.loads(log_file.read_text())
            assert data["test-model"]["input"] == 100
            assert data["test-model"]["output"] == 50

    def test_accumulate(self, tmp_path):
        log_file = tmp_path / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log_file):
            track_tokens("test-model", 100, 50)
            track_tokens("test-model", 200, 100)
            data = json.loads(log_file.read_text())
            assert data["test-model"]["input"] == 300
            assert data["test-model"]["output"] == 150


class TestGetTotals:
    def test_empty_log(self, tmp_path, capsys):
        log_file = tmp_path / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log_file):
            result = get_totals()
            assert result == {}

    def test_with_data(self, tmp_path, capsys):
        log_file = tmp_path / "token_log.json"
        log_file.write_text(json.dumps({"gpt-4": {"input": 1000, "output": 500}}))
        with patch("token_tracker.TOKEN_LOG", log_file):
            result = get_totals()
            assert "gpt-4" in result
            assert result["gpt-4"]["input"] == 1000


class TestScanOpencodeDb:
    def test_no_database(self, tmp_path, capsys):
        with patch("token_tracker.Path.home", return_value=tmp_path):
            result = scan_opencode_db()
            assert result == {}

    def test_single_prefix_key(self, tmp_path, capsys):
        db = _make_db(
            tmp_path,
            rows=[
                {
                    "providerID": "lmstudio_remote",
                    "id": "qwen/qwen3.6-35b-a3b",
                    "tokens": {"input": 1000, "output": 500},
                },
            ],
        )
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")
        ):
            result = scan_opencode_db(db_path=str(db), monthly=False)
        assert "lmstudio_remote/qwen/qwen3.6-35b-a3b" in result
        # Regression: the provider prefix must not be doubled.
        assert not any(k.startswith("lmstudio_remote/lmstudio_remote/") for k in result)

    def test_pricing_file_resolves_configured_model(self, tmp_path, capsys):
        db = _make_db(
            tmp_path,
            rows=[
                {
                    "providerID": "lmstudio_remote",
                    "id": "qwen/qwen3.6-35b-a3b",
                    "tokens": {"input": 1_000_000, "output": 1_000_000},
                },
            ],
        )
        pricing = tmp_path / "pricing.json"
        pricing.write_text(json.dumps({"lmstudio_remote/qwen/qwen3.6-35b-a3b": [2.0, 4.0]}))
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")
        ):
            result = scan_opencode_db(db_path=str(db), pricing_file=str(pricing), monthly=False)
        t = result["lmstudio_remote/qwen/qwen3.6-35b-a3b"]
        # 1M tokens at $2/$4 per million -> exactly $2 input, $4 output (not the default).
        assert t["input_cost"] == pytest.approx(2.0)
        assert t["output_cost"] == pytest.approx(4.0)

    def test_malformed_database_returns_empty(self, tmp_path, capsys):
        # DB file exists but has no `message` table -> should not raise.
        bad = tmp_path / "bad.db"
        conn = sqlite3.connect(str(bad))
        conn.execute("CREATE TABLE something_else (x INTEGER)")
        conn.commit()
        conn.close()
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), \
                patch("token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")):
            result = scan_opencode_db(db_path=str(bad), monthly=False)
        assert result == {}


class TestParseCliArgs:
    def test_default(self):
        args = _parse_cli_args(["token_tracker.py"])
        assert args["command"] == "totals"
        assert args["monthly"] is True
        assert args["obfuscate"] is False
        assert args["pricing_file"] is None

    def test_scan_with_flags(self):
        args = _parse_cli_args(
            ["token_tracker.py", "scan-oc", "--pricing-file", "p.json", "--obfuscate", "--no-monthly"]
        )
        assert args["command"] == "scan-oc"
        assert args["pricing_file"] == "p.json"
        assert args["obfuscate"] is True
        assert args["monthly"] is False

    def test_db_path_flag(self):
        args = _parse_cli_args(["token_tracker.py", "scan-oc", "--db-path", "/tmp/x.db"])
        assert args["command"] == "scan-oc"
        assert args["db_path"] == "/tmp/x.db"

    def test_scan_pi_flags(self):
        args = _parse_cli_args(
            ["token_tracker.py", "scan-pi", "--sessions-dir", "/tmp/s", "--obfuscate"]
        )
        assert args["command"] == "scan-pi"
        assert args["sessions_dir"] == "/tmp/s"
        assert args["obfuscate"] is True

    def test_scan_all_command(self):
        args = _parse_cli_args(["token_tracker.py", "scan-all"])
        assert args["command"] == "scan-all"

    def test_invalid_command_raises(self):
        with pytest.raises(SystemExit):
            _parse_cli_args(["token_tracker.py", "bogus"])


class TestMainDispatch:
    def test_dispatch_scan_oc(self, capsys):
        with patch("token_tracker.scan_opencode_db") as m:
            main(["token_tracker.py", "scan-oc"])
        m.assert_called_once()

    def test_dispatch_scan_all(self, capsys):
        with patch("token_tracker.scan_all") as m:
            main(["token_tracker.py", "scan-all"])
        m.assert_called_once()

    def test_dispatch_totals_default(self, capsys):
        with patch("token_tracker.get_totals") as m:
            main(["token_tracker.py"])
        m.assert_called_once()

    def test_dispatch_clear(self, capsys):
        with patch("token_tracker.clear_log") as m:
            main(["token_tracker.py", "clear"])
        m.assert_called_once()

    def test_dispatch_pricing(self, capsys):
        with patch("token_tracker._list_pricing_options") as m:
            main(["token_tracker.py", "pricing"])
        m.assert_called_once()

    def test_dispatch_scan_pi(self, capsys):
        with patch("token_tracker.scan_pi") as m:
            main(["token_tracker.py", "scan-pi"])
        m.assert_called_once()

    def test_dispatch_export(self, capsys):
        with patch("token_tracker.export_csv") as m:
            main(["token_tracker.py", "export"])
        m.assert_called_once()


class TestListPricingOptions:
    def test_prints_default(self, capsys):
        _list_pricing_options()
        out = capsys.readouterr().out
        assert "pricing" in out.lower()

    def test_no_stale_identifiers(self, capsys):
        _list_pricing_options()
        out = capsys.readouterr().out
        # These names never existed in the code; help must not mention them.
        assert "_KNOWN_MODELS" not in out
        assert "PRICING_OVERRIDES" not in out


class TestClearLog:
    def test_clear_removes_file(self, tmp_path, capsys):
        log_file = tmp_path / "token_log.json"
        log_file.write_text("{}")
        with patch("token_tracker.TOKEN_LOG", log_file):
            clear_log()
        assert not log_file.exists()
        assert "cleared" in capsys.readouterr().out.lower()

    def test_clear_noop(self, tmp_path, capsys):
        with patch("token_tracker.TOKEN_LOG", tmp_path / "missing.json"):
            clear_log()
        assert "no token log" in capsys.readouterr().out.lower()


class TestGetTotalsPricingFile:
    def test_totals_honor_pricing_file(self, tmp_path):
        log_file = tmp_path / "token_log.json"
        log_file.write_text(json.dumps({"openai/gpt-4o": {"input": 1_000_000, "output": 0}}))
        pricing = tmp_path / "pricing.json"
        pricing.write_text(json.dumps({"openai/gpt-4o": [2.0, 10.0]}))
        with patch("token_tracker.TOKEN_LOG", log_file):
            result = get_totals(pricing_file=str(pricing))
        assert result["openai/gpt-4o"]["input_cost"] == pytest.approx(2.0)


class TestScanRobustness:
    def _scan(self, tmp_path, db, **kw):
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")
        ):
            return scan_opencode_db(db_path=str(db), **kw)

    def test_malformed_and_non_dict_rows_skipped(self, tmp_path):
        db = _make_db(
            tmp_path,
            rows=[
                "{not json{{{",  # raw invalid JSON text
                '"just a string"',  # valid JSON but not an object
                {"providerID": "p", "id": "m", "tokens": None},  # null tokens
                {"providerID": "p", "id": "m2", "tokens": {"input": 10, "output": 5}},
            ],
        )
        result = self._scan(tmp_path, db, monthly=False)
        assert list(result) == ["p/m2"]

    def test_zero_usage_rows_skipped(self, tmp_path):
        db = _make_db(
            tmp_path,
            rows=[{"providerID": "p", "id": "m", "tokens": {"input": 0, "output": 0}}],
        )
        assert self._scan(tmp_path, db, monthly=False) == {}

    def test_monthly_breakdown_and_millis_timestamps(self, tmp_path, capsys):
        good = {"providerID": "p", "id": "m", "tokens": {"input": 1000, "output": 500}}
        db = _make_db(
            tmp_path,
            rows=[good, good],
            timestamps=[1_700_000_000, 1_700_000_000_000],  # seconds and millis of one instant
        )
        result = self._scan(tmp_path, db)
        out = capsys.readouterr().out
        assert "Monthly Breakdown:" in out
        month_lines = [line for line in out.splitlines() if line.startswith("2023-")]
        assert len(month_lines) == 1  # seconds and millis rows merged into one bucket
        assert result["p/m"]["input"] == 2000


def _pi_line(provider, model, input_t, output_t, ts=1_700_000_000_000):
    """One assistant-message JSONL line with the given usage."""
    return json.dumps(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "provider": provider,
                "model": model,
                "usage": {"input": input_t, "output": output_t},
                "timestamp": ts,
            },
        }
    )


def _write_pi_session(tmp_path, name, lines):
    """Write a pi-style sessions tree; returns the sessions root."""
    session_dir = tmp_path / "sessions" / f"--{name}--"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "s.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "sessions"


class TestScanPi:
    def _scan(self, tmp_path, root, **kw):
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")
        ):
            return scan_pi(sessions_dir=str(root), **kw)

    def test_no_sessions(self, tmp_path, capsys):
        result = self._scan(tmp_path, tmp_path / "missing")
        assert result == {}
        assert "No pi sessions" in capsys.readouterr().out

    def test_aggregates_across_files(self, tmp_path):
        root = tmp_path / "sessions"
        _write_pi_session(root.parent, "proj-a", [_pi_line("lmstudio", "qwen/x@q8", 1000, 500)])
        # second project dir under the same root
        proj_b = root / "--proj-b--"
        proj_b.mkdir(parents=True)
        (proj_b / "t.jsonl").write_text(
            "\n".join(
                [_pi_line("lmstudio", "qwen/x@q8", 2000, 300), _pi_line("openai", "gpt-4o", 100, 40)]
            )
            + "\n"
        )
        result = self._scan(tmp_path, root, monthly=False)
        assert result["lmstudio/qwen/x@q8"]["input"] == 3000
        assert result["lmstudio/qwen/x@q8"]["output"] == 800
        assert result["openai/gpt-4o"]["input"] == 100

    def test_skips_malformed_non_assistant_and_zero(self, tmp_path):
        root = _write_pi_session(
            tmp_path,
            "p",
            [
                "{not json",  # malformed line
                json.dumps({"type": "session", "cwd": "/x"}),  # header record
                json.dumps({"type": "message", "message": {"role": "user", "content": []}}),
                _pi_line("p", "m-zero", 0, 0),  # zero usage on both sides
                _pi_line("p", "m-ok", 10, 5),
            ],
        )
        result = self._scan(tmp_path, root, monthly=False)
        assert list(result) == ["p/m-ok"]

    def test_pricing_file_and_monthly(self, tmp_path, capsys):
        root = _write_pi_session(
            tmp_path, "p", [_pi_line("lmstudio", "qwen/y", 1_000_000, 1_000_000)]
        )
        pricing = tmp_path / "pricing.json"
        pricing.write_text(json.dumps({"lmstudio": [2.0, 4.0]}))
        result = self._scan(tmp_path, root, pricing_file=str(pricing))
        t = result["lmstudio/qwen/y"]
        # Provider-prefix override must win: 1M tokens at $2/$4 per million.
        assert t["input_cost"] == pytest.approx(2.0)
        assert t["output_cost"] == pytest.approx(4.0)
        out = capsys.readouterr().out
        assert "Monthly Breakdown:" in out
        assert any(line.startswith("2023-") for line in out.splitlines())

    def test_persists_to_log(self, tmp_path):
        root = _write_pi_session(tmp_path, "p", [_pi_line("prov", "mdl", 700, 300)])
        self._scan(tmp_path, root, monthly=False)
        data = json.loads((tmp_path / "token_log.json").read_text())
        assert data["prov/mdl"] == {"input": 700, "output": 300}


class TestScanAll:
    def _scan_all(self, tmp_path, **kw):
        kw.setdefault("env", "t")
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")
        ):
            return scan_all(**kw)

    def test_combines_sources_with_source_column(self, tmp_path, capsys):
        db = _make_db(
            tmp_path,
            [{"providerID": "p", "id": "m", "tokens": {"input": 1000, "output": 500}}],
        )
        root = _write_pi_session(tmp_path, "proj", [_pi_line("prov", "mdl", 700, 300)])
        result = self._scan_all(
            tmp_path, db_path=str(db), sessions_dir=str(root), monthly=False
        )
        assert ("opencode@t", "p/m") in result
        assert ("oh-my-pi@t", "prov/mdl") in result
        assert "Source" in capsys.readouterr().out
        # Ledger persists per model; sources merge under the plain model key.
        data = json.loads((tmp_path / "token_log.json").read_text())
        assert data["p/m"]["input"] == 1000
        assert data["prov/mdl"]["output"] == 300

    def test_single_source_still_reports(self, tmp_path, capsys):
        root = _write_pi_session(tmp_path, "proj", [_pi_line("prov", "mdl", 700, 300)])
        result = self._scan_all(
            tmp_path,
            db_path=str(tmp_path / "missing.db"),
            sessions_dir=str(root),
            monthly=False,
        )
        assert list(result) == [("oh-my-pi@t", "prov/mdl")]
        assert "No opencode database found." in capsys.readouterr().out

    def test_no_sources(self, tmp_path, capsys):
        result = self._scan_all(
            tmp_path,
            db_path=str(tmp_path / "missing.db"),
            sessions_dir=str(tmp_path / "none"),
            monthly=False,
        )
        assert result == {}
        assert "No data sources found." in capsys.readouterr().out


class TestWatermark:
    """Re-scanning a source must never double-count its history (regression)."""

    def _scan(self, tmp_path, **kw):
        kw.setdefault("env", "t")
        log = tmp_path / "data" / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            return scan_opencode_db(monthly=False, **kw)

    def _log(self, tmp_path):
        return json.loads((tmp_path / "data" / "token_log.json").read_text())

    def test_rescan_does_not_double_count(self, tmp_path, capsys):
        db = _make_db(
            tmp_path, [{"providerID": "p", "id": "m", "tokens": {"input": 1000, "output": 500}}]
        )
        first = self._scan(tmp_path, db_path=str(db))
        assert first["p/m"]["input"] == 1000
        second = self._scan(tmp_path, db_path=str(db))
        assert second == {}
        assert "already-tracked" in capsys.readouterr().out
        assert self._log(tmp_path)["p/m"]["input"] == 1000

    def test_watermark_advances_for_newer_rows(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 10, "output": 1}}
        db = _make_db(tmp_path, [row, dict(row)], timestamps=[100, 200])
        assert self._scan(tmp_path, db_path=str(db))["p/m"]["input"] == 20

        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO message (data, time_created) VALUES (?, ?)", (json.dumps(row), 300)
        )
        conn.commit()
        conn.close()

        assert self._scan(tmp_path, db_path=str(db))["p/m"]["input"] == 10
        log = self._log(tmp_path)
        assert log["p/m"]["input"] == 30
        assert log["_meta"]["watermark"]["opencode@t"] == 300

    def test_window_filters_before_tracking(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 100, "output": 5}}
        db = _make_db(tmp_path, [row, dict(row)], timestamps=[100, 1000])
        got = self._scan(tmp_path, db_path=str(db), since_ts=500)
        assert got["p/m"]["input"] == 100

    def test_windowed_scan_does_not_poison_watermark(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 10, "output": 1}}
        db = _make_db(
            tmp_path, [dict(row) for _ in range(6)], timestamps=[100, 200, 300, 400, 500, 600]
        )
        # Backfill only the tail of history first (windowed scan).
        assert self._scan(tmp_path, db_path=str(db), since_ts=500)["p/m"]["input"] == 20
        # A later full scan must still pick up the records the window skipped.
        assert self._scan(tmp_path, db_path=str(db))["p/m"]["input"] == 60
        # 20 + 60: windowed rows recount (accepted cost) instead of older rows
        # being skipped forever.
        assert self._log(tmp_path)["p/m"]["input"] == 80
        assert self._log(tmp_path)["_meta"]["watermark"]["opencode@t"] == 600

    def test_undated_records_count_once(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 100, "output": 5}}
        db = _make_db(tmp_path, [row], timestamps=[None])
        assert self._scan(tmp_path, db_path=str(db))["p/m"]["input"] == 100
        assert self._scan(tmp_path, db_path=str(db)) == {}
        assert self._log(tmp_path)["p/m"]["input"] == 100

    def test_get_totals_ignores_meta(self, tmp_path, capsys):
        log = tmp_path / "data" / "token_log.json"
        log.parent.mkdir()
        log.write_text(
            json.dumps({"p/m": {"input": 100, "output": 10}, "_meta": {"watermark": {"opencode": 5}}})
        )
        with patch("token_tracker.TOKEN_LOG", log):
            totals = get_totals()
        assert list(totals) == ["p/m"]
        assert "_meta" not in capsys.readouterr().out

    def test_track_tokens_coexists_with_meta(self, tmp_path):
        log = tmp_path / "data" / "token_log.json"
        db = _make_db(
            tmp_path, [{"providerID": "p", "id": "m", "tokens": {"input": 100, "output": 5}}]
        )
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            scan_opencode_db(db_path=str(db), monthly=False)
            track_tokens("manual/x", 7, 3)
        data = json.loads(log.read_text())
        assert data["manual/x"] == {"input": 7, "output": 3}
        assert "_meta" in data


class TestExportCsv:
    def test_exports_priced_rows(self, tmp_path, capsys):
        log = tmp_path / "token_log.json"
        log.write_text(
            json.dumps({"p/m": {"input": 1_000_000, "output": 500_000}, "_meta": {"watermark": {}}})
        )
        with patch("token_tracker.TOKEN_LOG", log):
            export_csv()
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "model,input,output,input_cost,output_cost,total_cost"
        model, in_tok, out_tok, in_cost, out_cost, total = lines[1].split(",")
        assert (model, in_tok, out_tok) == ("p/m", "1000000", "500000")
        # Default pricing 0.03/M in + 0.05/M out -> $0.03 + $0.025 = $0.055
        assert (in_cost, out_cost, total) == ("0.030000", "0.025000", "0.055000")

    def test_no_log(self, tmp_path, capsys):
        with patch("token_tracker.TOKEN_LOG", tmp_path / "none.json"):
            export_csv()
        assert capsys.readouterr().out == "No token log found.\n"


class TestPricingFileListing:
    def test_lists_file_entries(self, tmp_path, capsys):
        f = tmp_path / "pricing.json"
        f.write_text('{"_default": [1.0, 2.0], "prov/model": [3.0, 4.0]}')
        _list_pricing_options(str(f))
        out = capsys.readouterr().out
        assert "prov/model: $3.00/M input, $4.00/M output" in out
        assert "_default (file fallback): $1.00/M input, $2.00/M output" in out

    def test_without_file_lists_no_entries(self, capsys):
        _list_pricing_options()
        assert "Entries from" not in capsys.readouterr().out


class TestWindowCli:
    def test_since_until_parsed_as_utc_days(self):
        args = _parse_cli_args(
            ["token_tracker.py", "scan-oc", "--since", "2023-01-01", "--until", "2023-01-05"]
        )
        assert args["since_ts"] == 1_672_531_200  # 2023-01-01T00:00Z, inclusive start
        assert args["until_ts"] == 1_672_876_800 + 86_400  # through 2023-01-05 inclusive

    def test_bad_date_rejected(self):
        with pytest.raises(SystemExit):
            _parse_cli_args(["token_tracker.py", "scan-oc", "--since", "not-a-date"])


class TestEnvTagging:
    """Watermarks must be independent per environment (WSL + Windows share one ledger)."""

    def test_detect_env_wsl(self, monkeypatch):
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-24.04")
        assert _detect_env() == "wsl"

    def test_detect_env_windows(self, monkeypatch):
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.delenv("WSL_INTEROP", raising=False)
        monkeypatch.setattr("token_tracker.sys.platform", "win32")
        assert _detect_env() == "windows"

    def test_detect_env_local(self, monkeypatch):
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.delenv("WSL_INTEROP", raising=False)
        monkeypatch.setattr("token_tracker.sys.platform", "linux")
        assert _detect_env() == "local"

    def test_two_environments_aggregate(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 100, "output": 5}}
        (tmp_path / "wsl").mkdir()
        (tmp_path / "win").mkdir()
        db_wsl = _make_db(tmp_path / "wsl", [row])
        db_win = _make_db(tmp_path / "win", [dict(row)])
        log = tmp_path / "data" / "token_log.json"

        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            first = scan_opencode_db(db_path=str(db_wsl), monthly=False, env="wsl")
            second = scan_opencode_db(db_path=str(db_win), monthly=False, env="windows")
            third = scan_opencode_db(db_path=str(db_wsl), monthly=False, env="wsl")

        assert first["p/m"]["input"] == 100
        # The windows scan must not be skipped by the wsl watermark.
        assert second["p/m"]["input"] == 100
        # Per-environment idempotence still holds.
        assert third == {}
        data = json.loads(log.read_text())
        assert data["p/m"]["input"] == 200  # both environments aggregated
        assert set(data["_meta"]["watermark"]) == {"opencode@wsl", "opencode@windows"}

    def test_env_cli_flag(self):
        args = _parse_cli_args(["token_tracker.py", "scan-all", "--env", "desktop"])
        assert args["env"] == "desktop"


class TestPiMtimePrefilter:
    """Steady-state pi scans must skip files unchanged since the last full scan."""

    def _scan(self, tmp_path, root, **kw):
        log = tmp_path / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            return scan_pi(sessions_dir=str(root), env="t", **kw)

    def test_unchanged_files_skipped(self, tmp_path, capsys):
        root = _write_pi_session(tmp_path, "proj", [_pi_line("p", "m", 100, 5)])
        f = next(root.rglob("*.jsonl"))
        assert self._scan(tmp_path, root, monthly=False)["p/m"]["input"] == 100
        os.utime(f, (1_600_000_000, 1_600_000_000))  # older than the last full scan
        assert self._scan(tmp_path, root, monthly=False) == {}
        out = capsys.readouterr().out
        assert "Scanning 0 session files" in out and "1 unchanged since last scan" in out

    def test_modified_file_reread(self, tmp_path):
        root = _write_pi_session(tmp_path, "proj", [_pi_line("p", "m", 100, 5)])
        f = next(root.rglob("*.jsonl"))
        self._scan(tmp_path, root, monthly=False)
        with f.open("a") as fh:
            fh.write(_pi_line("p", "m", 50, 2, ts=1_700_000_001_000) + "\n")
        os.utime(f, None)  # fresh mtime: newer than the last full scan
        got = self._scan(tmp_path, root, monthly=False)
        assert got["p/m"]["input"] == 50


class TestLedgerMonthly:
    """The default command must show cumulative monthly totals from the ledger."""

    JAN = 1_672_531_200  # 2023-01-01T00:00Z
    APR = 1_680_307_200  # 2023-04-01T00:00Z

    def _scan(self, tmp_path, db, **kw):
        log = tmp_path / "data" / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            return scan_opencode_db(db_path=str(db), env="t", **kw)

    def test_totals_prints_monthly_breakdown(self, tmp_path, capsys):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 1_000_000, "output": 500_000}}
        db = _make_db(tmp_path, [row, dict(row)], timestamps=[self.JAN, self.APR])
        self._scan(tmp_path, db)
        capsys.readouterr()  # drop scan-time output; assert on the totals command only

        log = tmp_path / "data" / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log):
            get_totals()
        out = capsys.readouterr().out
        assert "Monthly Breakdown:" in out
        jan_line = next(line for line in out.splitlines() if line.startswith("2023-01"))
        apr_line = next(line for line in out.splitlines() if line.startswith("2023-04"))
        # Default pricing: 1M in -> $0.03, 0.5M out -> $0.025 per month.
        assert "1'000'000" in jan_line and "$0.06" in jan_line
        assert "1'000'000" in apr_line and "$0.06" in apr_line

    def test_months_accumulate_across_scans(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 100, "output": 5}}
        db = _make_db(tmp_path, [row], timestamps=[self.JAN])
        self._scan(tmp_path, db)

        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO message (data, time_created) VALUES (?, ?)", (json.dumps(row), self.APR)
        )
        conn.commit()
        conn.close()
        self._scan(tmp_path, db)  # only the April row is new; January bucket untouched

        data = json.loads((tmp_path / "data" / "token_log.json").read_text())
        assert data["_meta"]["months"]["2023-01"]["p/m"] == [100, 5]
        assert data["_meta"]["months"]["2023-04"]["p/m"] == [100, 5]

    def test_totals_without_months_unchanged(self, tmp_path, capsys):
        log = tmp_path / "token_log.json"
        log.write_text('{"p/m": {"input": 10, "output": 1}}')
        with patch("token_tracker.TOKEN_LOG", log):
            get_totals()
        assert "Monthly Breakdown:" not in capsys.readouterr().out


class TestLedgerSchema:
    """Every write stamps ``_meta.schema``; readers tolerate unknown versions."""

    def test_scan_stamps_schema(self, tmp_path):
        row = {"providerID": "p", "id": "m", "tokens": {"input": 10, "output": 1}}
        db = _make_db(tmp_path, [row])
        log = tmp_path / "data" / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            scan_opencode_db(db_path=str(db), env="t")
        data = json.loads(log.read_text())
        assert data["_meta"]["schema"] == _LEDGER_SCHEMA

    def test_track_tokens_stamps_schema(self, tmp_path):
        log = tmp_path / "token_log.json"
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            track_tokens("p/m", 10, 1)
        data = json.loads(log.read_text())
        assert data["_meta"]["schema"] == _LEDGER_SCHEMA
        assert data["p/m"] == {"input": 10, "output": 1}

    def test_unknown_future_schema_still_readable(self, tmp_path, capsys):
        log = tmp_path / "token_log.json"
        log.write_text(json.dumps({
            "p/m": {"input": 1_000_000, "output": 500_000},
            "_meta": {"schema": 99, "months": {"2023-01": {"p/m": [1_000_000, 500_000]}}},
        }))
        with patch("token_tracker.TOKEN_LOG", log):
            get_totals()
        out = capsys.readouterr().out
        assert "Monthly Breakdown:" in out and "2023-01" in out


class TestLocations:
    """Declared locations augment auto-detected sources, each under its own label."""

    def _patch_log(self, tmp_path):
        log = tmp_path / "token_log.json"
        return (
            patch("token_tracker.TOKEN_LOG", log),
            patch("token_tracker._LOCK_PATH", str(log) + ".lock"),
            log,
        )

    def test_load_valid(self, tmp_path):
        f = tmp_path / "locations.json"
        f.write_text(json.dumps({"locations": [
            {"label": "laptop", "type": "opencode", "path": "/a/opencode.db"},
            {"label": "server1", "type": "pi", "path": "/srv/sessions"},
        ]}))
        assert _load_locations(str(f)) == [
            Location("laptop", "opencode", "/a/opencode.db"),
            Location("server1", "pi", "/srv/sessions"),
        ]

    def test_load_malformed_file(self, tmp_path, capsys):
        f = tmp_path / "bad.json"
        f.write_text("{not json")
        assert _load_locations(str(f)) == []
        assert "Could not read locations file" in capsys.readouterr().out

    def test_load_skips_invalid_entries(self, tmp_path, capsys):
        f = tmp_path / "locations.json"
        f.write_text(json.dumps({"locations": [
            {"label": "a@b", "type": "opencode", "path": "/x"},
            {"label": "ok", "type": "unknown", "path": "/x"},
            {"label": "ok", "type": "pi", "path": "/y"},
            {"label": "ok2", "type": "pi", "path": ""},
            "not-a-dict",
        ]}))
        assert _load_locations(str(f)) == [Location("ok", "pi", "/y")]
        out = capsys.readouterr().out
        assert "Skipping invalid location entry" in out
        assert "Duplicate location label 'ok'" not in out  # first 'ok' was invalid

    def test_load_duplicate_label(self, tmp_path, capsys):
        f = tmp_path / "locations.json"
        f.write_text(json.dumps({"locations": [
            {"label": "dup", "type": "pi", "path": "/first"},
            {"label": "dup", "type": "opencode", "path": "/second"},
        ]}))
        assert _load_locations(str(f)) == [Location("dup", "pi", "/first")]
        assert "Duplicate location label 'dup'" in capsys.readouterr().out

    def test_scan_all_combines_auto_and_locations(self, tmp_path, capsys):
        auto_db = _make_db(tmp_path, [{"providerID": "p", "id": "m", "tokens": {"input": 10, "output": 1}}])
        loc_dir = tmp_path / "loc"
        loc_dir.mkdir()
        loc_db = _make_db(loc_dir, [{"providerID": "q", "id": "n", "tokens": {"input": 20, "output": 2}}])
        root = _write_pi_session(tmp_path, "proj", [_pi_line("r", "o", 30, 3)])
        loc_root = _write_pi_session(loc_dir / "pi", "srv", [_pi_line("s", "p", 40, 4)])
        p1, p2, log = self._patch_log(tmp_path)
        with p1, p2:
            totals = scan_all(
                db_path=str(auto_db),
                sessions_dir=str(root),
                env="t",
                monthly=False,
                locations=[
                    Location("laptop", "opencode", str(loc_db)),
                    Location("server1", "pi", str(loc_root)),
                ],
            )
        assert ("opencode@t", "p/m") in totals
        assert ("opencode@laptop", "q/n") in totals
        assert ("oh-my-pi@t", "r/o") in totals
        assert ("oh-my-pi@server1", "s/p") in totals
        data = json.loads(log.read_text())
        marks = data["_meta"]["watermark"]
        assert {"opencode@t", "opencode@laptop", "oh-my-pi@t", "oh-my-pi@server1"} <= set(marks)

    def test_missing_location_warns_and_continues(self, tmp_path, capsys):
        db = _make_db(tmp_path, [{"providerID": "p", "id": "m", "tokens": {"input": 10, "output": 1}}])
        p1, p2, _ = self._patch_log(tmp_path)
        with p1, p2:
            totals = scan_all(
                db_path=str(db),
                sessions_dir=str(tmp_path / "no-such-sessions"),
                env="t",
                monthly=False,
                locations=[Location("ghost", "opencode", str(tmp_path / "gone.db"))],
            )
        assert ("opencode@t", "p/m") in totals
        assert "Location 'ghost': no opencode database at" in capsys.readouterr().out

    def test_scan_pi_filters_pi_locations(self, tmp_path):
        root = _write_pi_session(tmp_path, "proj", [_pi_line("p", "m", 10, 1)])
        loc_dir = tmp_path / "loc"
        loc_dir.mkdir()
        loc_root = _write_pi_session(loc_dir, "srv", [_pi_line("q", "n", 20, 2)])
        (loc_dir / "oc").mkdir()
        oc_db = _make_db(loc_dir / "oc", [{"providerID": "z", "id": "z", "tokens": {"input": 99, "output": 9}}])
        p1, p2, log = self._patch_log(tmp_path)
        with p1, p2:
            totals = scan_pi(
                sessions_dir=str(root),
                env="t",
                monthly=False,
                locations=[
                    Location("srv", "pi", str(loc_root)),
                    Location("wrong", "opencode", str(oc_db)),  # filtered out by scan_pi
                ],
            )
        assert "q/n" in totals  # scan_pi returns totals keyed by model (source stripped)
        assert "z/z" not in totals  # opencode-kind location filtered out by scan_pi
        data = json.loads(log.read_text())
        assert "oh-my-pi@srv" in data["_meta"]["watermark"]
        assert "opencode@wrong" not in data["_meta"]["watermark"]


class TestMerge:
    """merge folds a foreign ledger in, relabels watermark keys, and is idempotent."""

    FOREIGN: ClassVar[dict] = {
        "p/m": {"input": 100, "output": 5},
        "_meta": {
            "schema": 1,
            "watermark": {"opencode@wsl": 100},
            "last_scan": {"opencode@wsl": 90},
            "months": {"2023-01": {"p/m": [100, 5]}},
        },
    }

    def _merge(self, tmp_path, foreign, label="laptop"):
        log = tmp_path / "data" / "token_log.json"
        log.parent.mkdir(exist_ok=True)
        f = tmp_path / "foreign.json"
        f.write_text(json.dumps(foreign) if not isinstance(foreign, str) else foreign)
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            merge_ledger(str(f), label)
        return log

    def test_merge_adds_totals_relabels_and_buckets(self, tmp_path, capsys):
        log = self._merge(tmp_path, self.FOREIGN)
        out = capsys.readouterr().out
        assert "Merged 1 model(s)" in out and "as laptop" in out
        data = json.loads(log.read_text())
        assert data["p/m"] == {"input": 100, "output": 5}
        assert data["_meta"]["watermark"]["opencode@wsl@laptop"] == 100
        assert data["_meta"]["last_scan"]["opencode@wsl@laptop"] == 90
        assert data["_meta"]["months"]["2023-01"]["p/m"] == [100, 5]
        assert data["_meta"]["merged"][0]["file"] == "foreign.json"

    def test_merge_into_existing_ledger(self, tmp_path):
        log = self._merge(tmp_path, self.FOREIGN)
        log.write_text(json.dumps({
            "p/m": {"input": 1, "output": 1},
            "other/x": {"input": 7, "output": 7},
            "_meta": {"watermark": {"opencode@wsl": 50}, "months": {"2023-01": {"p/m": [1, 1]}}},
        }))
        f = tmp_path / "foreign.json"
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            merge_ledger(str(f), "laptop")
        data = json.loads(log.read_text())
        assert data["p/m"] == {"input": 101, "output": 6}
        assert data["other/x"] == {"input": 7, "output": 7}
        # local watermark untouched (different key), months added element-wise
        assert data["_meta"]["watermark"]["opencode@wsl"] == 50
        assert data["_meta"]["months"]["2023-01"]["p/m"] == [101, 6]

    def test_remerge_is_noop(self, tmp_path, capsys):
        log = self._merge(tmp_path, self.FOREIGN)
        before = log.read_text()
        capsys.readouterr()
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            merge_ledger(str(tmp_path / "foreign.json"), "laptop")
        assert "already merged" in capsys.readouterr().out
        assert log.read_text() == before

    def test_relabel_is_idempotent_on_merged_ledger(self, tmp_path):
        # Merging a ledger that was itself merged must not double-suffix keys.
        log = self._merge(tmp_path, self.FOREIGN)
        relabeled = json.loads(log.read_text())
        f2 = tmp_path / "again.json"
        f2.write_text(json.dumps(relabeled))
        with patch("token_tracker.TOKEN_LOG", tmp_path / "fresh" / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "fresh" / "token_log.json.lock")
        ):
            merge_ledger(str(f2), "laptop")
        data = json.loads((tmp_path / "fresh" / "token_log.json").read_text())
        assert "opencode@wsl@laptop" in data["_meta"]["watermark"]
        assert not any(k.endswith("@laptop@laptop") for k in data["_meta"]["watermark"])

    def test_bad_inputs_leave_ledger_untouched(self, tmp_path, capsys):
        log = tmp_path / "data" / "token_log.json"
        log.parent.mkdir(exist_ok=True)
        log.write_text('{"p/m": {"input": 1, "output": 1}}')
        before = log.read_text()
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        empty = tmp_path / "empty.json"
        empty.write_text("{}")
        with patch("token_tracker.TOKEN_LOG", log), patch(
            "token_tracker._LOCK_PATH", str(log) + ".lock"
        ):
            merge_ledger(str(bad), "laptop")
            assert "Could not read foreign ledger" in capsys.readouterr().out
            merge_ledger(str(empty), "laptop")
            assert "no usage" in capsys.readouterr().out
            merge_ledger(str(log), "laptop")
            assert "Cannot merge the ledger into itself" in capsys.readouterr().out
        assert log.read_text() == before

    def test_cli_merge_args(self):
        args = _parse_cli_args(["token_tracker.py", "merge", "f.json", "--as", "laptop"])
        assert args["command"] == "merge"
        assert args["merge_path"] == "f.json"
        assert args["as_label"] == "laptop"
        args = _parse_cli_args(["token_tracker.py", "scan-all", "--locations", "loc.json"])
        assert args["locations_file"] == "loc.json"

    def test_cli_merge_requires_args(self, tmp_path, capsys):
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"):
            main(["token_tracker.py", "merge"])
            assert "merge requires a ledger path." in capsys.readouterr().out
            main(["token_tracker.py", "merge", "f.json"])
            assert "merge requires --as LABEL." in capsys.readouterr().out
