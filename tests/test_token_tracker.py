"""Tests for token_tracker module."""

import json
import sqlite3
from unittest.mock import patch

import pytest

from token_tracker import (
    _DEFAULT_PRICING,
    _fmt,
    _get_pricing,
    _list_pricing_options,
    _load_pricing_overrides,
    _parse_cli_args,
    _parse_model_str,
    _safe_read_json,
    estimate_tokens,
    get_totals,
    main,
    scan_opencode_db,
    scan_sessions,
    track_tokens,
)


def _make_db(tmp_path, rows):
    """Create a temp opencode-style SQLite DB with the given message rows."""
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE message (data TEXT, time_created INTEGER)")
        for r in rows:
            conn.execute(
                "INSERT INTO message (data, time_created) VALUES (?, ?)",
                (json.dumps(r), 1_700_000_000),
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


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_single_word(self):
        assert estimate_tokens("hello") == 1

    def test_multiple_words(self):
        assert estimate_tokens("hello world foo") == 3

    def test_consistent_ratio(self):
        text = " ".join(["word"] * 100)
        tokens = estimate_tokens(text)
        assert tokens == 130


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


class TestParseCliArgs:
    def test_default(self):
        args = _parse_cli_args(["token_tracker.py"])
        assert args["command"] == "totals"
        assert args["monthly"] is True
        assert args["obfuscate"] is False
        assert args["pricing_file"] is None

    def test_scan_with_flags(self):
        args = _parse_cli_args(
            ["token_tracker.py", "scan", "--pricing-file", "p.json", "--obfuscate", "--no-monthly"]
        )
        assert args["command"] == "scan"
        assert args["pricing_file"] == "p.json"
        assert args["obfuscate"] is True
        assert args["monthly"] is False

    def test_invalid_command_raises(self):
        with pytest.raises(SystemExit):
            _parse_cli_args(["token_tracker.py", "bogus"])


class TestMainDispatch:
    def test_dispatch_scan(self, capsys):
        with patch("token_tracker.scan_opencode_db") as m:
            main(["token_tracker.py", "scan"])
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

    def test_dispatch_scan_json(self, capsys):
        with patch("token_tracker.scan_sessions") as m:
            main(["token_tracker.py", "scan-json"])
        m.assert_called_once()


class TestScanSessions:
    def test_scans_session_files(self, tmp_path, capsys):
        sess = tmp_path / "sessions"
        sess.mkdir()
        (sess / "a.json").write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "hello world foo bar"},
                        {"role": "assistant", "content": "hi there"},
                    ],
                }
            )
        )
        with patch("token_tracker.TOKEN_LOG", tmp_path / "token_log.json"), patch(
            "token_tracker._LOCK_PATH", str(tmp_path / "token_log.json.lock")
        ):
            scan_sessions(session_dir=str(sess), model="local-llm")
        data = json.loads((tmp_path / "token_log.json").read_text())
        assert data["local-llm"]["input"] > 0
        assert data["local-llm"]["output"] > 0


class TestListPricingOptions:
    def test_prints_default(self, capsys):
        _list_pricing_options()
        out = capsys.readouterr().out
        assert "pricing" in out.lower()
