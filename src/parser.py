"""
src/parser.py — Multi-format SIEM log parser.

Deterministic tie-breaking policy (documented here and in parse_logs docstring):
  1. Primary sort  : timestamp (float hours, UTC)
  2. Secondary sort: severity rank (High=0, Medium=1, Low=2)
  3. Tertiary sort : MD5(source_ip + event_type) for fully deterministic ordering

Malformed rows are skipped; count is logged and returned in the metadata dict.
"""
import hashlib
import json
import logging
import os
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical event-type mapping
_EVENT_TYPE_CANON = {
    "4624": "4624_logon_success",
    4624: "4624_logon_success",
    "4625": "4625_failed_login",
    4625: "4625_failed_login",
    "4688": "4688_process_creation",
    4688: "4688_process_creation",
    "7045": "7045_service_install",
    7045: "7045_service_install",
    "firewall_block": "firewall_block",
    "process_create": "4688_process_creation",
    "edr_process": "4688_process_creation",
    "BLOCK": "firewall_block",
}

_SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

# Output schema columns
_OUTPUT_COLUMNS = ["timestamp_h", "timestamp_iso", "event_type", "source_ip", "host", "severity"]

# ---------------------------------------------------------------------------
# Helper: severity rank
# ---------------------------------------------------------------------------

def _severity_rank(s: str) -> int:
    """
    Map severity string to sort rank.

    High=0, Medium=1, Low=2, unknown=3.

    Parameters
    ----------
    s : str — severity label (case-insensitive)

    Returns
    -------
    int
    """
    if not isinstance(s, str):
        return 3
    return _SEVERITY_ORDER.get(s.strip().capitalize(), 3)


# ---------------------------------------------------------------------------
# Helper: parse ISO timestamp to UTC-localized pd.Timestamp
# ---------------------------------------------------------------------------

def _parse_iso_to_utc(ts_str: str) -> pd.Timestamp:
    """
    Parse an ISO8601 timestamp string (possibly with UTC offset) to a
    UTC-localized pd.Timestamp.

    Supported offset formats:
      - Z or +00:00   → UTC
      - -05:00        → EST
      - -08:00        → PST
      - +05:30        → IST
      - any other ±HH:MM offset

    Parameters
    ----------
    ts_str : str — ISO8601 timestamp, e.g. "2024-01-01T03:45:12-05:00"

    Returns
    -------
    pd.Timestamp — UTC-localized

    Raises
    ------
    ValueError if parsing fails
    """
    ts_str = ts_str.strip()

    # Replace trailing Z with +00:00
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"

    try:
        ts = pd.Timestamp(ts_str)
    except Exception as exc:
        raise ValueError(f"Cannot parse timestamp: {ts_str!r}") from exc

    if ts.tzinfo is None:
        # Assume UTC if no timezone info
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


# ---------------------------------------------------------------------------
# Parser: JSONL (Windows Security Events)
# ---------------------------------------------------------------------------

def _parse_jsonl_file(path: str) -> Tuple[pd.DataFrame, int]:
    """
    Parse a Windows Security Event JSONL file.

    Each valid line must be a JSON object with at least:
      TimeCreated, EventID, Computer, IpAddress, Severity

    Malformed / truncated lines are skipped and counted.

    Parameters
    ----------
    path : str

    Returns
    -------
    (df, n_dropped) — DataFrame of valid rows and count of dropped rows
    """
    rows = []
    n_dropped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_dropped += 1
                logger.debug("JSONL parse error at %s line %d — skipping", path, lineno)
                continue

            # Validate required fields
            ts_val = obj.get("TimeCreated") or obj.get("timestamp")
            if not ts_val:
                n_dropped += 1
                continue

            try:
                ts_utc = _parse_iso_to_utc(str(ts_val))
            except (ValueError, TypeError):
                n_dropped += 1
                continue

            eid = obj.get("EventID")
            if eid is not None:
                event_type = _EVENT_TYPE_CANON.get(eid, _EVENT_TYPE_CANON.get(str(eid), f"win_{eid}"))
            else:
                event_type = str(obj.get("event_type", "unknown"))

            source_ip = str(obj.get("IpAddress") or obj.get("source_ip", "0.0.0.0"))
            host = str(obj.get("Computer") or obj.get("host", "UNKNOWN"))
            severity = str(obj.get("Severity") or obj.get("severity", "Low"))

            rows.append({
                "timestamp_iso_utc": ts_utc,
                "event_type": event_type,
                "source_ip": source_ip,
                "host": host,
                "severity": severity,
                "_source_file": path,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp_iso_utc", "event_type", "source_ip", "host", "severity", "_source_file"]
    )
    return df, n_dropped


# ---------------------------------------------------------------------------
# Parser: CSV (EDR and Firewall)
# ---------------------------------------------------------------------------

def _parse_csv_file(path: str) -> Tuple[pd.DataFrame, int]:
    """
    Parse an EDR or Firewall CSV file.

    EDR CSVs: columns include [timestamp, event_type, host, source_ip, severity, ...]
    Firewall CSVs: columns include [timestamp, action, src_ip, dst_ip, dst_port, severity, ...]

    Rows missing 'timestamp' are dropped.

    Parameters
    ----------
    path : str

    Returns
    -------
    (df, n_dropped) — DataFrame of valid rows and count of dropped rows
    """
    try:
        raw_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        logger.warning("Could not read CSV %s: %s", path, exc)
        return pd.DataFrame(), 0

    n_total = len(raw_df)
    rows = []
    n_dropped = 0

    # Determine file type by column presence
    is_firewall = "src_ip" in raw_df.columns and "dst_ip" in raw_df.columns
    is_edr = "process_name" in raw_df.columns

    ts_col = "timestamp"
    if ts_col not in raw_df.columns:
        logger.warning("CSV %s has no 'timestamp' column — skipping file", path)
        return pd.DataFrame(), n_total

    for idx, row in raw_df.iterrows():
        ts_raw = row.get(ts_col, "").strip()
        if not ts_raw:
            n_dropped += 1
            continue
        try:
            ts_utc = _parse_iso_to_utc(ts_raw)
        except ValueError:
            n_dropped += 1
            continue

        if is_firewall:
            event_type = _EVENT_TYPE_CANON.get(
                row.get("action", "").strip(), "firewall_block"
            )
            source_ip = str(row.get("src_ip", "0.0.0.0"))
            host = str(row.get("dst_ip", "UNKNOWN"))
            severity = str(row.get("severity", "High"))
        elif is_edr:
            raw_etype = row.get("event_type", "process_create").strip()
            event_type = _EVENT_TYPE_CANON.get(raw_etype, "4688_process_creation")
            source_ip = str(row.get("source_ip", "0.0.0.0"))
            host = str(row.get("host", "UNKNOWN"))
            severity = str(row.get("severity", "Medium"))
        else:
            # Generic CSV: best-effort column mapping
            event_type = str(row.get("event_type", row.get("action", "unknown")))
            event_type = _EVENT_TYPE_CANON.get(event_type, event_type)
            source_ip = str(row.get("source_ip", row.get("src_ip", "0.0.0.0")))
            host = str(row.get("host", row.get("dst_ip", "UNKNOWN")))
            severity = str(row.get("severity", "Low"))

        rows.append({
            "timestamp_iso_utc": ts_utc,
            "event_type": event_type,
            "source_ip": source_ip,
            "host": host,
            "severity": severity,
            "_source_file": path,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["timestamp_iso_utc", "event_type", "source_ip", "host", "severity", "_source_file"]
    )
    return df, n_dropped


# ---------------------------------------------------------------------------
# Deterministic tie-breaking key
# ---------------------------------------------------------------------------

def _md5_tiebreak(source_ip: str, event_type: str) -> str:
    """
    Compute MD5(source_ip + event_type) as a hex string for deterministic
    tertiary sort ordering.
    """
    raw = (str(source_ip) + str(event_type)).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def parse_logs(input_dir: str, output_path: str = "data/processed/events.csv") -> pd.DataFrame:
    """
    Walk *input_dir* recursively, parse all *.jsonl and *.csv files, normalise
    event types, apply deterministic tie-breaking, and write the unified event
    table to *output_path*.

    Deterministic tie-breaking policy
    ----------------------------------
    1. Primary   : timestamp_h (float hours since global t0, UTC)
    2. Secondary : severity rank (High=0, Medium=1, Low=2, unknown=3)
    3. Tertiary  : MD5(source_ip + event_type) — fully deterministic

    Parameters
    ----------
    input_dir   : str — root directory to walk
    output_path : str — path to write the processed events CSV

    Returns
    -------
    pd.DataFrame with schema:
        timestamp_h   (float)  — hours since earliest event (UTC)
        timestamp_iso (str)    — UTC ISO8601 string
        event_type    (str)    — canonical event type
        source_ip     (str)
        host          (str)
        severity      (str)
    """
    input_path = pathlib.Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    all_frames = []
    total_dropped = 0

    # Walk directory recursively
    for root, _dirs, files in os.walk(input_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            if fname.endswith(".jsonl"):
                logger.info("Parsing JSONL: %s", fpath)
                df, n_drop = _parse_jsonl_file(fpath)
                total_dropped += n_drop
            elif fname.endswith(".csv"):
                logger.info("Parsing CSV: %s", fpath)
                df, n_drop = _parse_csv_file(fpath)
                total_dropped += n_drop
            else:
                continue

            if not df.empty:
                all_frames.append(df)

    logger.info("Total malformed/dropped rows across all files: %d", total_dropped)

    if not all_frames:
        logger.warning("No parseable log files found under %s", input_dir)
        empty_df = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_path, index=False)
        return empty_df

    combined = pd.concat(all_frames, ignore_index=True)

    # Convert timestamps to float hours relative to global t0
    ts_series = combined["timestamp_iso_utc"]
    t0 = ts_series.min()
    logger.info("Global t0 (earliest timestamp): %s", t0)

    combined["timestamp_h"] = (ts_series - t0).dt.total_seconds() / 3600.0
    combined["timestamp_iso"] = ts_series.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Severity rank for sort
    combined["_sev_rank"] = combined["severity"].apply(_severity_rank)

    # MD5 tiebreak key
    combined["_md5_key"] = combined.apply(
        lambda r: _md5_tiebreak(r["source_ip"], r["event_type"]), axis=1
    )

    # Apply deterministic tie-breaking sort
    combined = combined.sort_values(
        by=["timestamp_h", "_sev_rank", "_md5_key"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    # Select output columns
    out_df = combined[_OUTPUT_COLUMNS].copy()

    # Write output
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info(
        "Wrote %d events to %s (dropped %d malformed rows)",
        len(out_df), out_path, total_dropped,
    )

    return out_df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "data/processed/events.csv"
    df = parse_logs(input_dir, output_csv)
    print(df.head())
