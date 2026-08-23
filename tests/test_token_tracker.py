"""Tests for token_tracker module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from token_tracker import (
    _DEFAULT_PRICING,
    _fmt,
    _get_pricing,
    _load_pricing_overrides,
    _parse_model_str,
    _safe_read_json,
    estimate_tokens,
    get_totals,
    scan_opencode_db,
    track_tokens,
)


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
