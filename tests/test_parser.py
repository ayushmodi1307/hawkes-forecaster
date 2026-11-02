"""Tests for src/parser.py — timezone normalization, malformed rows, tie-breaking."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.parser import parse_logs, _parse_iso_to_utc, _severity_rank


# ---------------------------------------------------------------------------
# Reference epoch used internally by the parser
# ---------------------------------------------------------------------------
_REF_ISO = "2024-01-15T00:00:00+00:00"


def _make_event(ts_iso: str, event_type: str = "auth", source_ip: str = "1.2.3.4",
                host: str = "server-01", severity: str = "High") -> dict:
    return {
        "timestamp": ts_iso,
        "event_type": event_type,
        "source_ip": source_ip,
        "host": host,
        "severity": severity,
    }


def _write_jsonl(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# 1. ISO UTC normalisation — EST offset
# ---------------------------------------------------------------------------

def test_iso_utc_normalization():
    """'2024-01-15T10:00:00-05:00' should be 5 hours later than '...T10:00:00+00:00'."""
    t_est = _parse_iso_to_utc("2024-01-15T10:00:00-05:00")   # 15:00 UTC
    t_utc = _parse_iso_to_utc("2024-01-15T10:00:00+00:00")   # 10:00 UTC
    diff_h = (t_est - t_utc).total_seconds() / 3600.0
    assert abs(diff_h - 5.0) < 1e-6, (
        f"Expected 5.0 h difference, got {diff_h:.6f}"
    )


# ---------------------------------------------------------------------------
# 2. IST timezone
# ---------------------------------------------------------------------------

def test_ist_timezone():
    """'2024-01-15T10:00:00+05:30' should be UTC 04:30 → offset is -5.5 h from naive."""
    t_ist = _parse_iso_to_utc("2024-01-15T10:00:00+05:30")   # 04:30 UTC
    t_utc = _parse_iso_to_utc("2024-01-15T10:00:00+00:00")   # 10:00 UTC
    diff_h = (t_utc - t_ist).total_seconds() / 3600.0
    assert abs(diff_h - 5.5) < 1e-6, (
        f"Expected UTC - IST = 5.5 h, got {diff_h:.6f}"
    )


# ---------------------------------------------------------------------------
# 3. Malformed JSON lines skipped
# ---------------------------------------------------------------------------

def test_malformed_json_skipped(tmp_path):
    """3 valid + 2 corrupt JSON lines → parse_logs returns exactly 3 rows, no exception."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "events.jsonl"

    valid_records = [
        _make_event("2024-01-15T01:00:00+00:00"),
        _make_event("2024-01-15T02:00:00+00:00"),
        _make_event("2024-01-15T03:00:00+00:00"),
    ]
    with open(log_file, "w") as f:
        for r in valid_records:
            f.write(json.dumps(r) + "\n")
        f.write("THIS IS NOT JSON AT ALL\n")
        f.write('{"timestamp": broken}\n')

    df = parse_logs(str(log_dir))
    assert isinstance(df, pd.DataFrame), "parse_logs should return a DataFrame"
    assert len(df) == 3, f"Expected 3 valid rows, got {len(df)}"


# ---------------------------------------------------------------------------
# 4. Deterministic tie-breaking — identical timestamps
# ---------------------------------------------------------------------------

def test_deterministic_tie_breaking(tmp_path):
    """Two events with identical timestamp: output order identical across two runs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "events.jsonl"

    records = [
        _make_event("2024-01-15T06:00:00+00:00", event_type="login",  host="host-A"),
        _make_event("2024-01-15T06:00:00+00:00", event_type="logout", host="host-B"),
    ]
    _write_jsonl(log_file, records)

    df1 = parse_logs(str(log_dir))
    df2 = parse_logs(str(log_dir))

    assert list(df1["host"]) == list(df2["host"]), (
        f"Tie-breaking not deterministic: {list(df1['host'])} != {list(df2['host'])}"
    )


# ---------------------------------------------------------------------------
# 5. Severity rank ordering
# ---------------------------------------------------------------------------

def test_severity_rank_ordering():
    """High must rank below (numerically less than) Medium, which ranks below Low."""
    assert _severity_rank("High") < _severity_rank("Medium"), (
        "High severity should have a lower rank number than Medium"
    )
    assert _severity_rank("Medium") < _severity_rank("Low"), (
        "Medium severity should have a lower rank number than Low"
    )


# ---------------------------------------------------------------------------
# 6. Output schema
# ---------------------------------------------------------------------------

def test_output_schema(tmp_path):
    """parse_logs on a valid directory returns all required columns."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_jsonl(log_dir / "events.jsonl", [
        _make_event("2024-01-15T08:00:00+00:00"),
        _make_event("2024-01-15T09:30:00+00:00", severity="Low"),
    ])

    df = parse_logs(str(log_dir))
    required = {"timestamp_h", "timestamp_iso", "event_type", "source_ip", "host", "severity"}
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


# ---------------------------------------------------------------------------
# 7. Output sorted
# ---------------------------------------------------------------------------

def test_output_sorted(tmp_path):
    """Parsed events must be in ascending timestamp_h order."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    # Write out-of-order records
    records = [
        _make_event("2024-01-15T12:00:00+00:00"),
        _make_event("2024-01-15T06:00:00+00:00"),
        _make_event("2024-01-15T09:00:00+00:00"),
    ]
    _write_jsonl(log_dir / "events.jsonl", records)

    df = parse_logs(str(log_dir))
    ts = df["timestamp_h"].values
    assert np.all(np.diff(ts) >= 0), "Parsed events not sorted by timestamp_h"


# ---------------------------------------------------------------------------
# 8. Empty input directory
# ---------------------------------------------------------------------------

def test_empty_input_dir(tmp_path):
    """parse_logs on an empty directory returns an empty DataFrame without crashing."""
    log_dir = tmp_path / "empty_logs"
    log_dir.mkdir()

    df = parse_logs(str(log_dir))
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame even when empty"
    assert len(df) == 0, f"Expected 0 rows for empty dir, got {len(df)}"
