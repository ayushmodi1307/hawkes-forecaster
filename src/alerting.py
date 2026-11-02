"""
src/alerting.py — Threshold calibration, alert generation, and delivery.

Threshold calibration (on VALIDATION data only, never test):
  Method 1 (quantile): threshold = quantile(historical_expected_counts, q)
  Method 2 (F-beta): sweep thresholds on validation set, pick the one
                     maximising F_β(precision, recall) where β>1 weights recall.

Alert payload schema (all fields required):
  timestamp, horizon_h, expected_count, peak_intensity, threshold,
  p_burst_mc, naive_poisson_p_value, confidence,
  mu, alpha, beta, branching_ratio, fano_implied, recommended_actions, why

confidence rule:
  - "high"   : margin > 0.3 * threshold AND hawkes converged AND n_events > 50
  - "medium" : margin > 0.1 * threshold OR  hawkes converged
  - "low"    : otherwise

recommended_actions: always list from {"isolate_subnet","block_smb","revoke_credentials"}
  - include "block_smb" if any firewall_block on port 445/139 in recent window
  - include "revoke_credentials" if 4625_failed_login events spike in recent window
  - include "isolate_subnet" if 4688_process_creation events spike or lateral movement
  Default (no context): all three.

send_alert: POST to webhook_url if configured, else append JSON line to alerts.log.
"""
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import scipy.stats as stats
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold_quantile(
    historical_counts: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """Set alert threshold as a quantile of historical expected counts.

    Parameters
    ----------
    historical_counts : np.ndarray
        Array of expected (or observed) counts over a validation/historical
        window. Should NOT include test-set data.
    quantile : float
        Quantile level (default 0.95 → 95th percentile).

    Returns
    -------
    float : threshold value.
    """
    if len(historical_counts) == 0:
        raise ValueError("historical_counts is empty.")
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}.")
    threshold = float(np.quantile(historical_counts, quantile))
    logger.info("Quantile threshold (q=%.3f): %.4f", quantile, threshold)
    return threshold


def calibrate_threshold_fbeta(
    val_counts: np.ndarray,
    val_labels: np.ndarray,
    fbeta_beta: float = 2.0,
) -> Tuple[float, float]:
    """Select threshold maximising F_β score on a validation set.

    Sweeps candidate thresholds from min to max of val_counts (100 steps).
    β > 1 weights recall more heavily than precision, which is appropriate
    for security alerting where missed attacks are costlier than false alarms.

    Parameters
    ----------
    val_counts : np.ndarray
        Model-predicted expected counts on validation windows.
    val_labels : np.ndarray
        Binary ground-truth burst labels (1=burst, 0=normal) aligned with
        val_counts.
    fbeta_beta : float
        β parameter for F_β. Default 2.0 (weights recall 2× more than
        precision).

    Returns
    -------
    (best_threshold, best_fbeta) : Tuple[float, float]
    """
    val_counts = np.asarray(val_counts, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    if len(val_counts) != len(val_labels):
        raise ValueError("val_counts and val_labels must have the same length.")

    candidates = np.linspace(val_counts.min(), val_counts.max(), 100)
    best_thresh = candidates[0]
    best_fb = -1.0

    b2 = fbeta_beta ** 2

    for thresh in candidates:
        preds = (val_counts >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (val_labels == 1)))
        fp = int(np.sum((preds == 1) & (val_labels == 0)))
        fn = int(np.sum((preds == 0) & (val_labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = b2 * precision + recall
        fb = (1 + b2) * precision * recall / denom if denom > 0 else 0.0

        if fb > best_fb:
            best_fb = fb
            best_thresh = float(thresh)

    logger.info(
        "F-beta calibration (beta=%.1f): threshold=%.4f  F_beta=%.4f",
        fbeta_beta, best_thresh, best_fb,
    )
    return best_thresh, best_fb


# ---------------------------------------------------------------------------
# Alert logic with hysteresis and cooldown
# ---------------------------------------------------------------------------

def should_alert(
    expected_count: float = 0.0,
    threshold: float = 0.0,
    current_state: Optional[dict] = None,
    cooldown_s: float = 300.0,
    hysteresis_pct: float = 0.10,
    intensity: Optional[float] = None,
    cooldown_remaining_h: Optional[float] = None,
) -> Tuple[bool, Union[dict, str]]:
    """Determine whether to fire an alert, applying hysteresis and cooldown.

    Supports both SOC state dictionary tracking and dashboard instant evaluation.
    """
    # Check if caller passed (intensity, expected_count, threshold, cooldown) positionally
    if isinstance(current_state, (int, float)):
        intensity = float(expected_count)
        expected_count = float(threshold)
        threshold = float(current_state)
        if cooldown_s is not None and isinstance(cooldown_s, (int, float)):
            cooldown_remaining_h = float(cooldown_s)
        current_state = None
        is_dashboard_call = True
    else:
        is_dashboard_call = (intensity is not None and current_state is None)

    if current_state is None:
        state = {"last_alert_time": 0.0, "in_alert": False, "alert_start_count": 0.0}
    else:
        state = dict(current_state)

    now_ts = time.time()
    in_alert = state.get("in_alert", False)
    last_alert_time = state.get("last_alert_time", 0.0)
    alert_start_count = state.get("alert_start_count", 0.0)

    lower_band = threshold * (1.0 - hysteresis_pct)
    if cooldown_remaining_h is not None:
        cooldown_elapsed = (cooldown_remaining_h <= 0.0)
    else:
        cooldown_elapsed = (now_ts - last_alert_time) >= cooldown_s

    val_to_check = max(expected_count, intensity if intensity is not None else 0.0)
    fire_alert = False

    if not in_alert:
        if val_to_check >= threshold and cooldown_elapsed:
            fire_alert = True
            in_alert = True
            alert_start_count = val_to_check
            last_alert_time = now_ts
    else:
        if val_to_check < lower_band:
            in_alert = False
            alert_start_count = 0.0
        else:
            if cooldown_elapsed:
                fire_alert = True
                last_alert_time = now_ts

    state["in_alert"] = in_alert
    state["last_alert_time"] = last_alert_time
    state["alert_start_count"] = alert_start_count

    if is_dashboard_call:
        if fire_alert:
            reason = f"Forecast/intensity ({val_to_check:.2f}) exceeded threshold ({threshold:.2f}). Threat wave imminent."
        else:
            reason = f"Observed count ({val_to_check:.2f}) below threshold ({threshold:.2f})."
        return fire_alert, reason

    return fire_alert, state


# ---------------------------------------------------------------------------
# Alert payload builder
# ---------------------------------------------------------------------------

def _derive_confidence(
    expected_count: float,
    threshold: float,
    hawkes_fit: dict,
    n_events: int,
) -> str:
    """Compute confidence label per the documented rules."""
    margin = expected_count - threshold
    converged = hawkes_fit.get("converged", True)  # assume converged unless flagged

    if margin > 0.3 * threshold and converged and n_events > 50:
        return "high"
    if margin > 0.1 * threshold or converged:
        return "medium"
    return "low"


def _derive_recommended_actions(
    recent_events: Optional[np.ndarray],
    recent_event_types: Optional[list],
) -> List[str]:
    """Derive recommended actions from recent event context.

    Falls back to returning all three actions if no context is provided.
    """
    ALL_ACTIONS = ["isolate_subnet", "block_smb", "revoke_credentials"]

    if recent_event_types is None or len(recent_event_types) == 0:
        return ALL_ACTIONS

    event_types = [str(e).lower() for e in recent_event_types]
    actions = []

    # block_smb: firewall_block on port 445 or 139
    smb_keywords = ["firewall_block", "port_445", "port_139", "smb", "445", "139"]
    if any(any(kw in et for kw in smb_keywords) for et in event_types):
        actions.append("block_smb")

    # revoke_credentials: 4625 failed login spikes
    if any("4625" in et or "failed_login" in et or "login_fail" in et for et in event_types):
        actions.append("revoke_credentials")

    # isolate_subnet: 4688 process creation or lateral movement
    lateral_keywords = ["4688", "process_creation", "lateral", "move", "wmi", "psexec"]
    if any(any(kw in et for kw in lateral_keywords) for et in event_types):
        actions.append("isolate_subnet")

    # Default fallback
    if not actions:
        actions = ALL_ACTIONS

    return sorted(set(actions))


def build_alert_payload(
    t_now: float,
    horizon_h: float,
    expected_count: float,
    mc_results: dict,
    hawkes_fit: dict,
    threshold: float,
    recent_events: Optional[np.ndarray] = None,
    recent_event_types: Optional[list] = None,
) -> dict:
    """Build a complete alert payload conforming to the documented schema.

    All schema fields are included:
      timestamp, horizon_h, expected_count, peak_intensity, threshold,
      p_burst_mc, naive_poisson_p_value, confidence,
      mu, alpha, beta, branching_ratio, fano_implied, recommended_actions, why

    Parameters
    ----------
    t_now : float
        Current Unix timestamp (evaluation time).
    horizon_h : float
        Forecast horizon.
    expected_count : float
        Closed-form expected count E[N|H_t].
    mc_results : dict
        Output of forecast.monte_carlo_forecast().
    hawkes_fit : dict
        Dict with keys: mu, alpha, beta, branching_ratio (and optionally
        n_events, converged, peak_intensity).
    threshold : float
        Calibrated alert threshold.
    recent_events : np.ndarray, optional
        Timestamps of recent events (for context).
    recent_event_types : list of str, optional
        Event-type labels aligned with recent_events.

    Returns
    -------
    dict : fully-populated alert payload.
    """
    mu = float(hawkes_fit.get("mu", 0.0))
    alpha = float(hawkes_fit.get("alpha", 0.0))
    beta = float(hawkes_fit.get("beta", 1.0))
    n_ratio = float(hawkes_fit.get("branching_ratio", alpha / beta if beta > 0 else 0.0))
    n_events = int(hawkes_fit.get("n_events", 0))
    peak_intensity = float(hawkes_fit.get("peak_intensity", mu / max(1.0 - n_ratio, 1e-9)))

    # Fano factor implied by branching ratio: 1/(1−n)²
    if 0.0 <= n_ratio < 1.0:
        fano_implied = 1.0 / (1.0 - n_ratio) ** 2
    else:
        fano_implied = float("inf")

    # P(burst) from MC: P(N ≥ 1)
    p_burst_mc = float(mc_results.get("p_exceed", {}).get(1, 0.0))

    # Naive Poisson tail probability: P(Poisson(threshold) ≥ ceil(expected_count))
    k_alert = max(1, int(math.ceil(threshold)))
    lam_naive = max(expected_count, 1e-9)
    naive_poisson_p_value = float(1.0 - stats.poisson.cdf(k_alert - 1, lam_naive))

    confidence = _derive_confidence(expected_count, threshold, hawkes_fit, n_events)
    actions = _derive_recommended_actions(recent_events, recent_event_types)

    margin = expected_count - threshold
    why = (
        f"Expected count ({expected_count:.2f}) exceeds threshold ({threshold:.2f}) "
        f"by {margin:.2f} ({100*margin/max(threshold,1e-9):.1f}%). "
        f"MC P(burst)={p_burst_mc:.3f}. "
        f"Hawkes branching ratio n={n_ratio:.3f} "
        f"({'clustered' if n_ratio > 0.1 else 'near-Poisson'}). "
        f"Confidence: {confidence}."
    )

    payload = {
        "timestamp": datetime.fromtimestamp(t_now, tz=timezone.utc).isoformat(),
        "horizon_h": horizon_h,
        "expected_count": expected_count,
        "peak_intensity": peak_intensity,
        "threshold": threshold,
        "p_burst_mc": p_burst_mc,
        "naive_poisson_p_value": naive_poisson_p_value,
        "confidence": confidence,
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": n_ratio,
        "fano_implied": fano_implied,
        "recommended_actions": actions,
        "why": why,
        # Extra enrichment fields (not in the minimal schema but useful)
        "mc_quantiles": mc_results.get("quantiles", {}),
        "mc_mean_count": mc_results.get("mean_count", expected_count),
        "p_exceed": mc_results.get("p_exceed", {}),
        "gap_mc_vs_poisson": mc_results.get("gap_mc_vs_poisson", {}),
    }
    return payload


# ---------------------------------------------------------------------------
# Alert delivery
# ---------------------------------------------------------------------------

def send_alert(
    payload: dict,
    webhook_url: Optional[str] = None,
    alerts_log_path: str = "alerts.log",
    timeout_s: int = 5,
    retries: int = 3,
) -> bool:
    """Send an alert payload via HTTP POST or append to a log file.

    Tries the webhook up to `retries` times with exponential back-off.
    Falls back to file logging on all failures or when webhook_url is None.

    Parameters
    ----------
    payload : dict
        Alert payload (from build_alert_payload).
    webhook_url : str, optional
        If provided, POSTs JSON to this URL.
    alerts_log_path : str
        Path to JSONL log file used when webhook is absent or fails.
    timeout_s : int
        HTTP request timeout in seconds.
    retries : int
        Maximum number of POST attempts.

    Returns
    -------
    bool : True if delivery succeeded (webhook or file), False on error.
    """
    line = json.dumps(payload, default=str)

    if webhook_url:
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    webhook_url,
                    data=line,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout_s,
                )
                if resp.ok:
                    logger.info("Alert delivered via webhook (attempt %d).", attempt)
                    return True
                else:
                    logger.warning(
                        "Webhook returned HTTP %d on attempt %d.", resp.status_code, attempt
                    )
            except requests.RequestException as exc:
                logger.warning("Webhook attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s …

        logger.error("All %d webhook attempts failed; falling back to log file.", retries)

    # File fallback
    try:
        os.makedirs(os.path.dirname(os.path.abspath(alerts_log_path)), exist_ok=True)
        with open(alerts_log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info("Alert appended to %s.", alerts_log_path)
        return True
    except OSError as exc:
        logger.error("Failed to write alert log: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Alerting loop
# ---------------------------------------------------------------------------

def run_alerting_loop(
    events_df,
    hawkes_fit: dict,
    cfg: dict,
) -> List[dict]:
    """Sliding-window alerting loop over an events DataFrame.

    Walks through the event stream in steps of `eval_step_h` hours,
    evaluates the forecast for each horizon, and fires alerts when the
    expected count exceeds the calibrated threshold.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must have a 'timestamp' column (float seconds or datetime).
    hawkes_fit : dict
        Fitted Hawkes parameters: mu, alpha, beta, branching_ratio.
        Optionally: n_events, converged, peak_intensity.
    cfg : dict
        Configuration. Expected keys:
          alerting.threshold          (float) — pre-calibrated threshold
          alerting.horizons           (list)  — forecast horizons in hours
          alerting.eval_step_h        (float) — evaluation step (default 1.0)
          alerting.K_mc               (int)   — MC paths (default 500)
          alerting.cooldown_s         (float) — alert cooldown seconds
          alerting.webhook_url        (str|None)
          alerting.alerts_log_path    (str)
          alerting.recent_window_h    (float) — window for event-type context

    Returns
    -------
    List[dict] : list of fired alert payloads (in chronological order).
    """
    import pandas as pd
    from src.forecast import forecast as run_forecast

    alert_cfg = cfg.get("alerting", {})
    threshold = float(alert_cfg.get("threshold", 10.0))
    horizons = alert_cfg.get("horizons", [1.0, 6.0, 24.0])
    eval_step_h = float(alert_cfg.get("eval_step_h", 1.0))
    K_mc = int(alert_cfg.get("K_mc", 500))
    cooldown_s = float(alert_cfg.get("cooldown_s", 300.0))
    webhook_url = alert_cfg.get("webhook_url", None)
    alerts_log_path = alert_cfg.get("alerts_log_path", "alerts.log")
    recent_window_h = float(alert_cfg.get("recent_window_h", 1.0))

    mu = float(hawkes_fit["mu"])
    alpha = float(hawkes_fit["alpha"])
    beta = float(hawkes_fit["beta"])

    # Parse timestamps
    ts_col = events_df["timestamp"]
    if hasattr(ts_col, "dt"):
        event_times = ts_col.astype(np.int64).values / 1e9
    else:
        event_times = ts_col.astype(float).values
    event_times = np.sort(event_times)

    # Event types column (optional)
    has_types = "event_type" in events_df.columns

    T_start = event_times[0]
    T_end = event_times[-1]
    eval_step_s = eval_step_h * 3600.0

    alert_state = {"last_alert_time": 0.0, "in_alert": False, "alert_start_count": 0.0}
    fired_alerts = []

    t_now = T_start + eval_step_s
    while t_now <= T_end:
        past_events = event_times[event_times <= t_now]
        if len(past_events) == 0:
            t_now += eval_step_s
            continue

        # Recent events for context
        recent_cutoff = t_now - recent_window_h * 3600.0
        recent_mask = (event_times > recent_cutoff) & (event_times <= t_now)
        recent_ev = event_times[recent_mask]
        recent_types = None
        if has_types:
            recent_types = events_df["event_type"].values[recent_mask].tolist()

        # Forecast for all horizons
        try:
            fc_results = run_forecast(
                event_times=past_events,
                t_now=t_now,
                horizons=horizons,
                mu=mu,
                alpha=alpha,
                beta=beta,
                K=K_mc,
                rng_seed=int(t_now) % (2**31),
            )
        except Exception as exc:
            logger.warning("Forecast failed at t=%.1f: %s", t_now, exc)
            t_now += eval_step_s
            continue

        # Use the first (shortest) horizon as the primary alert trigger
        primary_h = float(horizons[0])
        fc = fc_results[primary_h]
        ec = fc["expected_count"]

        fire, alert_state = should_alert(
            ec, threshold, alert_state, cooldown_s=cooldown_s
        )

        if fire:
            hawkes_fit_aug = dict(hawkes_fit)
            hawkes_fit_aug["n_events"] = int(len(past_events))
            hawkes_fit_aug["peak_intensity"] = fc["peak_intensity"]

            payload = build_alert_payload(
                t_now=t_now,
                horizon_h=primary_h,
                expected_count=ec,
                mc_results=fc["mc_results"],
                hawkes_fit=hawkes_fit_aug,
                threshold=threshold,
                recent_events=recent_ev,
                recent_event_types=recent_types,
            )
            sent = send_alert(
                payload,
                webhook_url=webhook_url,
                alerts_log_path=alerts_log_path,
            )
            if sent:
                fired_alerts.append(payload)
                logger.info(
                    "Alert fired at t=%.1f  horizon=%.1fh  E[N]=%.2f  threshold=%.2f",
                    t_now, primary_h, ec, threshold,
                )

        t_now += eval_step_s

    logger.info("Alerting loop complete. Total alerts fired: %d", len(fired_alerts))
    return fired_alerts
