"""
Tests for src/alerting.py — calibration, hysteresis, cooldown, payload schema.
"""
import json
import os
import numpy as np
import pytest
from src.alerting import (
    calibrate_threshold_quantile,
    calibrate_threshold_fbeta,
    should_alert,
    build_alert_payload,
    send_alert,
)


def test_calibrate_threshold_quantile():
    """Quantile threshold matches numpy percentile."""
    counts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    q_thresh = calibrate_threshold_quantile(counts, quantile=0.90)
    expected = float(np.quantile(counts, 0.90))
    assert abs(q_thresh - expected) < 1e-6


def test_should_alert_hysteresis():
    """Once in_alert, state stays True until count drops below threshold * (1 - hysteresis)."""
    threshold = 10.0
    hysteresis_pct = 0.20  # reset band at 8.0
    state = {"last_alert_time": 0.0, "in_alert": False, "alert_start_count": 0.0}

    # Step 1: Count reaches 11.0 -> alert fires
    fire, state = should_alert(11.0, threshold, state, cooldown_s=0.0, hysteresis_pct=hysteresis_pct)
    assert fire is True
    assert state["in_alert"] is True

    # Step 2: Count drops to 9.0 (below 10.0 but above 8.0) -> still in_alert, no new fire without cooldown
    fire, state = should_alert(9.0, threshold, state, cooldown_s=100.0, hysteresis_pct=hysteresis_pct)
    assert state["in_alert"] is True

    # Step 3: Count drops to 7.5 (below 8.0) -> in_alert resets to False
    fire, state = should_alert(7.5, threshold, state, cooldown_s=0.0, hysteresis_pct=hysteresis_pct)
    assert state["in_alert"] is False


def test_should_alert_cooldown():
    """Alert does not re-fire within cooldown window."""
    threshold = 10.0
    state = {"last_alert_time": 0.0, "in_alert": False, "alert_start_count": 0.0}

    fire1, state = should_alert(12.0, threshold, state, cooldown_s=60.0)
    assert fire1 is True

    # Immediate second check -> suppressed by cooldown
    fire2, state = should_alert(15.0, threshold, state, cooldown_s=60.0)
    assert fire2 is False


def test_build_alert_payload_schema():
    """Alert payload contains all required SOC advisory keys."""
    mc_results = {
        "quantiles": {0.5: 8.0},
        "mean_count": 8.5,
        "p_exceed": {10: 0.25},
        "gap_mc_vs_poisson": {10: 0.15},
    }
    hawkes_fit = {
        "mu": 0.5,
        "alpha": 0.6,
        "beta": 1.2,
        "branching_ratio": 0.5,
        "converged": True,
    }
    payload = build_alert_payload(
        t_now=50.0,
        horizon_h=2.0,
        expected_count=12.0,
        mc_results=mc_results,
        hawkes_fit=hawkes_fit,
        threshold=10.0,
        recent_event_types=["4625_failed_login", "firewall_block_port_445"],
    )

    required_keys = [
        "timestamp", "horizon_h", "expected_count", "threshold",
        "confidence", "branching_ratio", "recommended_actions"
    ]
    for k in required_keys:
        assert k in payload, f"Missing key {k} in alert payload"

    assert "revoke_credentials" in payload["recommended_actions"]
    assert "block_smb" in payload["recommended_actions"]


def test_send_alert_appends_log(tmp_path):
    """send_alert with webhook_url=None appends valid JSON line to log file."""
    log_file = tmp_path / "alerts.log"
    payload = {"test_alert": True, "value": 42}
    ok = send_alert(payload, webhook_url=None, alerts_log_path=str(log_file))
    assert ok is True
    assert log_file.exists()

    with open(log_file, "r") as f:
        data = json.loads(f.readline().strip())
    assert data["test_alert"] is True
    assert data["value"] == 42

