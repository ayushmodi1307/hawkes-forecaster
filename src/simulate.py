"""
src/simulate.py — Event-stream simulators and SIEM log emitter.

Theoretical basis
-----------------
Homogeneous Poisson
  α = 0 collapses Hawkes to λ(t)=μ; gaps ~ Exp(μ), CV=1, Fano=1.

Hawkes (exponential kernel)
  λ(t) = μ + Σ_{tᵢ<t} α exp(−β(t−tᵢ))
  Stationary if n = α/β < 1.
  Mean rate   : μ/(1−n)
  Fano factor : 1/(1−n)² (asymptotic over window Δ→∞)

Poisson cluster view
  Immigrants ~ Poisson(μ); each spawns Poisson(n) children, delays ~ Exp(β).
  Cluster size S ~ Borel(n): P(S=k)=exp(−nk)(nk)^(k−1)/k!, E[S]=1/(1−n).

Ogata thinning (Lewis & Shedler 1979)
  Propose candidates from Poisson(λ_max); accept with prob λ(t)/λ_max.

Kill-chain family
  Staged cascade: failed logins → process creation → lateral movement → encryption.
  Each stage delay ~ Exp(1/mean_delay); burst intensity decays as t^(−γ).
"""
import json
import logging
import math
import os
import pathlib
import random
import hashlib
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ScenarioResult(tuple):
    """
    Result container that acts as both a 2-tuple (event_times, wave_labels)
    and a mapping with keys 'event_times' and 'wave_labels'.
    """
    def __new__(cls, event_times, wave_labels):
        return super().__new__(cls, (event_times, wave_labels))

    @property
    def event_times(self) -> np.ndarray:
        return self[0]

    @property
    def wave_labels(self) -> list:
        return self[1]

    def __getitem__(self, item):
        if item == "event_times":
            return self[0]
        if item == "wave_labels":
            return self[1]
        return super().__getitem__(item)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the project config dict from config.yaml."""
    cfg_path = pathlib.Path(config_path)
    if not cfg_path.exists():
        # Try relative to this file's parent
        cfg_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", cfg_path)
    return cfg


# ---------------------------------------------------------------------------
# Homogeneous Poisson
# ---------------------------------------------------------------------------

def simulate_homogeneous_poisson(mu: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a homogeneous Poisson process with rate *mu* on [0, T].

    Uses the exact gap-based method: inter-arrivals ~ Exp(mu).

    Parameters
    ----------
    mu  : float  — constant arrival rate (events per hour)
    T   : float  — observation horizon (hours)
    rng : np.random.Generator

    Returns
    -------
    np.ndarray — sorted event times in [0, T]
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    times: List[float] = []
    t = 0.0
    while True:
        gap = rng.exponential(1.0 / mu)
        t += gap
        if t >= T:
            break
        times.append(t)
    return np.array(times, dtype=float)


# ---------------------------------------------------------------------------
# Diurnal (inhomogeneous) Poisson — Ogata thinning
# ---------------------------------------------------------------------------

def _diurnal_rate(t: float, mu0: float, amplitude: float, phase_h: float) -> float:
    """Sinusoidal rate: mu0 * (1 + amplitude * cos(2π(t mod 24 − phase_h)/24))."""
    return mu0 * (1.0 + amplitude * math.cos(2.0 * math.pi * (t % 24.0 - phase_h) / 24.0))


def simulate_diurnal_poisson(
    mu0: float,
    amplitude: float,
    phase_h: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate an inhomogeneous Poisson process with sinusoidal (diurnal) rate
    via Ogata thinning.

    λ(t) = mu0 * (1 + amplitude * cos(2π(t mod 24 − phase_h) / 24))

    The upper bound λ_max = mu0 * (1 + amplitude) is used for the proposal.

    Parameters
    ----------
    mu0       : float — base rate
    amplitude : float — relative amplitude in (0, 1)
    phase_h   : float — peak hour (0–24)
    T         : float — horizon (hours)
    rng       : np.random.Generator

    Returns
    -------
    np.ndarray — sorted event times in [0, T]
    """
    if not (0.0 <= amplitude < 1.0):
        raise ValueError(f"amplitude must be in [0, 1), got {amplitude}")
    lambda_max = mu0 * (1.0 + amplitude)
    times: List[float] = []
    t = 0.0
    while True:
        gap = rng.exponential(1.0 / lambda_max)
        t += gap
        if t >= T:
            break
        lam_t = _diurnal_rate(t, mu0, amplitude, phase_h)
        if rng.uniform() <= lam_t / lambda_max:
            times.append(t)
    return np.array(times, dtype=float)


# ---------------------------------------------------------------------------
# Hawkes process — Ogata thinning
# ---------------------------------------------------------------------------

def simulate_hawkes_ogata(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a Hawkes process with exponential kernel via Ogata thinning.

    λ(t) = μ + Σ_{tᵢ<t} α exp(−β(t−tᵢ))

    Stationarity requires branching ratio n = α/β < 1.

    Parameters
    ----------
    mu    : float — background rate
    alpha : float — excitation magnitude
    beta  : float — exponential decay rate
    T     : float — horizon (hours)
    rng   : np.random.Generator

    Returns
    -------
    np.ndarray — sorted event times in [0, T]
    """
    n = alpha / beta if beta > 0 else 0.0
    if n >= 1.0:
        raise ValueError(f"Branching ratio α/β = {n:.4f} ≥ 1 (non-stationary).")

    times: List[float] = []
    t = 0.0
    lambda_star = mu

    while True:
        if lambda_star <= 0:
            lambda_star = max(mu, 1e-6)
        gap = rng.exponential(1.0 / lambda_star)
        t += gap
        if t >= T:
            break

        # Evaluate true rate at candidate time t
        if times and alpha > 0:
            decay_sum = sum(math.exp(-beta * (t - ti)) for ti in times)
            lambda_t = mu + alpha * decay_sum
        else:
            lambda_t = mu

        u = rng.uniform()
        if u <= lambda_t / lambda_star:
            times.append(t)
            lambda_star = lambda_t + alpha
        else:
            lambda_star = lambda_t

    return np.array(times, dtype=float)


# ---------------------------------------------------------------------------
# Hawkes process — Branching / cluster construction
# ---------------------------------------------------------------------------

def simulate_hawkes_branching(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a Hawkes process via the Poisson-cluster (branching) representation.

    Immigrants ~ HomPoisson(mu, T).
    Each event (immigrant or offspring) independently generates a Poisson(alpha/beta)
    number of children; each child arrives after an Exp(beta) delay.
    Events outside [0, T] are discarded.

    This exactly matches the stationary distribution of the Hawkes process.

    Parameters
    ----------
    mu    : float — background (immigrant) rate
    alpha : float — excitation magnitude
    beta  : float — exponential decay rate
    T     : float — horizon (hours)
    rng   : np.random.Generator

    Returns
    -------
    np.ndarray — sorted event times in [0, T]
    """
    n = alpha / beta
    if n >= 1.0:
        raise ValueError(f"Branching ratio α/β = {n:.4f} ≥ 1 (non-stationary).")

    # Step 1: Generate immigrants
    immigrants = simulate_homogeneous_poisson(mu, T, rng)
    all_times: List[float] = list(immigrants)

    # Step 2: BFS over generations
    generation = list(immigrants)
    while generation:
        next_gen: List[float] = []
        for parent_t in generation:
            n_children = rng.poisson(n)
            for _ in range(n_children):
                delay = rng.exponential(1.0 / beta)
                child_t = parent_t + delay
                if child_t < T:
                    all_times.append(child_t)
                    next_gen.append(child_t)
        generation = next_gen

    return np.sort(np.array(all_times, dtype=float))


# ---------------------------------------------------------------------------
# Borel distribution PMF
# ---------------------------------------------------------------------------

def borel_pmf(k: np.ndarray, n: float) -> np.ndarray:
    """
    Vectorized Borel distribution PMF.

    P(S = k) = exp(−n k) (n k)^(k−1) / k!

    Arises as the cluster-size distribution in the Hawkes branching representation.

    Parameters
    ----------
    k : array-like of positive integers
    n : float — branching ratio in (0, 1)

    Returns
    -------
    np.ndarray of probabilities (same shape as k)
    """
    k = np.asarray(k, dtype=float)
    if np.any(k < 1):
        raise ValueError("k must be ≥ 1 for the Borel distribution.")
    if not (0.0 < n < 1.0):
        raise ValueError(f"n must be in (0, 1), got {n}")
    log_pmf = -n * k + (k - 1.0) * np.log(n * k) - _log_factorial_vec(k)
    return np.exp(log_pmf)


def _log_factorial_vec(k: np.ndarray) -> np.ndarray:
    """Vectorised log-factorial via scipy.special.gammaln."""
    from scipy.special import gammaln  # lazy import
    return gammaln(k + 1.0)


# ---------------------------------------------------------------------------
# Hawkes scenario generator
# ---------------------------------------------------------------------------

def generate_hawkes_scenario(
    T: float = 168.0,
    mu: float = 0.5,
    alpha: float = 0.6,
    beta: float = 1.2,
    wave_count: int = 5,
    wave_boost: float = 4.0,
    wave_duration: float = 2.0,
    n_waves_rng: Optional[np.random.Generator] = None,
    rng: Optional[np.random.Generator] = None,
) -> ScenarioResult:
    if rng is None:
        rng = np.random.default_rng(42)
    if n_waves_rng is None:
        n_waves_rng = rng
    """
    Generate a Hawkes-process event stream with injected "attack waves".

    During each wave the background rate is boosted by *wave_boost* ×.
    Waves are placed uniformly at random in [0, T] with the constraint that
    they do not overlap.

    Parameters
    ----------
    T              : float — horizon
    mu             : float — baseline background rate
    alpha          : float — Hawkes excitation
    beta           : float — Hawkes decay
    wave_count     : int   — number of attack waves
    wave_boost     : float — multiplicative rate boost during waves
    wave_duration  : float — duration of each wave (hours)
    n_waves_rng    : np.random.Generator — used to place waves
    rng            : np.random.Generator — used for simulation

    Returns
    -------
    event_times : np.ndarray
    wave_labels : List[Tuple[float, float]] — (start, end) of each wave
    """
    # Place waves with random starts, ensuring no overlap
    wave_labels: List[Tuple[float, float]] = []
    attempts = 0
    while len(wave_labels) < wave_count and attempts < 10_000:
        attempts += 1
        start = n_waves_rng.uniform(0.0, T - wave_duration)
        end = start + wave_duration
        overlap = any(not (end <= ws or start >= we) for ws, we in wave_labels)
        if not overlap:
            wave_labels.append((start, end))
    wave_labels.sort(key=lambda x: x[0])

    # Simulate piecewise: alternate between baseline and boosted segments
    # Build a time-varying background schedule and run branching sim
    def mu_at(t: float) -> float:
        for ws, we in wave_labels:
            if ws <= t < we:
                return mu * wave_boost
        return mu

    # Simulate via branching with effective mean rate, then accept/reject
    # For correctness: run a Hawkes simulation in segments, restarting the
    # excitation kernel properly between wave transitions.
    all_times: List[float] = []
    excitation_history: List[float] = []  # accumulated event times for kernel

    t_prev = 0.0
    segments: List[Tuple[float, float, float]] = []
    breakpoints = sorted(set([0.0, T] + [ws for ws, _ in wave_labels] + [we for _, we in wave_labels]))
    for i in range(len(breakpoints) - 1):
        seg_start = breakpoints[i]
        seg_end = breakpoints[i + 1]
        seg_mu = mu_at(0.5 * (seg_start + seg_end))
        segments.append((seg_start, seg_end, seg_mu))

    # Run Ogata thinning across entire [0,T] with piecewise mu
    times: List[float] = []
    t = 0.0
    lambda_star = mu * wave_boost + 0.0  # initial upper bound (conservative)

    def current_mu(t_: float) -> float:
        return mu_at(t_)

    def excitation_at(t_: float, event_list: List[float]) -> float:
        return sum(alpha * math.exp(-beta * (t_ - ti)) for ti in event_list if ti < t_)

    # Upper bound: max(mu)*wave_boost + excitation can grow; use adaptive bound
    lambda_star = mu * wave_boost
    while t < T:
        gap = rng.exponential(1.0 / max(lambda_star, 1e-10))
        t_cand = t + gap
        if t_cand >= T:
            break
        exc = excitation_at(t_cand, times)
        lam_t = current_mu(t_cand) + exc
        lambda_new = current_mu(t_cand) * wave_boost + exc + alpha  # upper bound after accept
        if rng.uniform() <= lam_t / max(lambda_star, lam_t):
            times.append(t_cand)
            lambda_star = lam_t + alpha  # after event: excitation jumps by alpha
        else:
            lambda_star = max(lam_t, lambda_star * math.exp(-beta * gap))
        t = t_cand

    return ScenarioResult(np.sort(np.array(times, dtype=float)), wave_labels)


# ---------------------------------------------------------------------------
# Kill-chain scenario generator
# ---------------------------------------------------------------------------

def generate_killchain_scenario(
    T: float = 168.0,
    mu_background: float = 0.5,
    wave_count: int = 5,
    stage_delays: Optional[List[float]] = None,
    power_law_exp: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> ScenarioResult:
    if stage_delays is None:
        stage_delays = [0.5, 1.0, 1.5]
    if rng is None:
        rng = np.random.default_rng(42)
    """
    Generate a staged kill-chain event stream.

    Stages (4):
      0  failed_login  — initial brute-force
      1  process_creation — attacker establishes foothold
      2  lateral_movement — spread across hosts
      3  encryption_start — ransomware/exfil payload

    For each wave:
      - Wave start is drawn uniformly from [0, T - sum(stage_delays)]
      - Each stage begins after Exp(1/stage_delay[i]) after the previous
      - Burst events within a stage follow a power-law decay: rate ~ (t - t_stage)^(-gamma)
        implemented as thinned Poisson

    Background events are homogeneous Poisson(mu_background) throughout.

    Parameters
    ----------
    T              : float
    mu_background  : float
    wave_count     : int
    stage_delays   : List[float] — mean delay (hours) between consecutive stages
    power_law_exp  : float       — exponent γ for power-law burst decay
    rng            : np.random.Generator

    Returns
    -------
    event_times : np.ndarray
    wave_labels : List[Tuple[float, float]] — (wave_start, wave_end)
    """
    n_stages = 4
    if len(stage_delays) < n_stages - 1:
        raise ValueError(f"Need {n_stages - 1} stage_delays, got {len(stage_delays)}")

    # Background
    bg_times = list(simulate_homogeneous_poisson(mu_background, T, rng))

    wave_labels: List[Tuple[float, float]] = []
    all_wave_times: List[float] = []

    total_delay = sum(stage_delays)
    max_start = T - total_delay - 1.0
    if max_start <= 0:
        logger.warning("T too short for kill-chain scenario; adjusting.")
        max_start = T * 0.1

    for wave_idx in range(wave_count):
        wave_start = rng.uniform(0.0, max(max_start, 0.01))
        stage_start = wave_start

        wave_event_times: List[float] = [wave_start]

        for stage_i in range(n_stages):
            # Delay to next stage
            if stage_i > 0:
                delay = rng.exponential(stage_delays[stage_i - 1])
                stage_start = stage_start + delay

            if stage_start >= T:
                break

            # Burst duration proportional to remaining stage_delay (or 1h)
            burst_dur = stage_delays[stage_i] if stage_i < len(stage_delays) else 1.0
            burst_end = min(stage_start + burst_dur, T)

            # Power-law thinned Poisson burst
            # Base rate (at t0 of stage) — thinned by (t - stage_start + 0.1)^(-gamma)
            burst_base_rate = mu_background * 5.0  # 5x background during burst
            t_burst = stage_start
            while t_burst < burst_end:
                gap = rng.exponential(1.0 / burst_base_rate)
                t_cand = t_burst + gap
                if t_cand >= burst_end:
                    break
                dt = max(t_cand - stage_start, 0.01)
                accept_prob = min(1.0, dt ** (-power_law_exp))
                if rng.uniform() <= accept_prob:
                    all_wave_times.append(t_cand)
                    wave_event_times.append(t_cand)
                t_burst = t_cand

        wave_end = max(wave_event_times) if wave_event_times else wave_start
        wave_labels.append((wave_start, min(wave_end, T)))

    combined = np.sort(np.array(bg_times + all_wave_times, dtype=float))
    combined = combined[combined < T]
    return ScenarioResult(combined, wave_labels)


# ---------------------------------------------------------------------------
# SIEM log emitter
# ---------------------------------------------------------------------------

# Benign field pools — NO malware names, NO exploit strings
_WINDOWS_EVENT_IDS = [4624, 4625, 4688, 7045]
_PROCESS_NAMES = ["powershell.exe", "svchost.exe", "net.exe", "cmd.exe", "wscript.exe"]
_USERNAMES = ["jsmith", "adm_svc", "SYSTEM", "Network Service", "Local Service",
              "bob.k", "alice.t", "svc_backup", "helpdesk01"]
_HOSTNAMES = [f"WS-{i:04d}" for i in range(1, 51)] + [f"SRV-{i:02d}" for i in range(1, 11)]
_SEVERITY_MAP = {4624: "Low", 4625: "High", 4688: "Medium", 7045: "High"}
_TZ_OFFSETS = ["+00:00", "-05:00", "-08:00", "+05:30", "+00:00"]  # UTC, EST, PST, IST, UTC

def _random_ip(rng: np.random.Generator, private: bool = True) -> str:
    if private:
        prefixes = ["10.", "192.168.", "172.16."]
        p = rng.choice(prefixes)
        if p == "10.":
            return f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"
        elif p == "192.168.":
            return f"192.168.{rng.integers(0,255)}.{rng.integers(1,254)}"
        else:
            return f"172.16.{rng.integers(0,31)}.{rng.integers(1,254)}"
    else:
        return f"{rng.integers(1,223)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"


def _hours_to_iso(t_h: float, tz_offset: str = "+00:00") -> str:
    """Convert float hours to an ISO8601 timestamp string (anchored at 2024-01-01T00:00:00)."""
    base_epoch = 1704067200.0  # 2024-01-01T00:00:00Z in UNIX seconds
    ts_unix = base_epoch + t_h * 3600.0
    dt = pd.Timestamp(ts_unix, unit="s", tz="UTC")
    if tz_offset != "+00:00":
        # Convert to offset-aware string
        sign = 1 if tz_offset[0] == "+" else -1
        parts = tz_offset[1:].split(":")
        offset_h = sign * (int(parts[0]) + int(parts[1]) / 60.0)
        dt = dt + pd.Timedelta(hours=offset_h)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + tz_offset


def emit_siem_logs(
    event_times: np.ndarray,
    split_name: str,
    output_dir: str,
    rng: np.random.Generator,
) -> None:
    """
    Write realistic benign-looking SIEM logs for the given event times.

    Outputs (under output_dir / split_name /):
      - windows_events_{split}.jsonl  — Windows Security JSON lines
      - edr_{split}.csv               — EDR process-creation events
      - firewall_{split}.csv          — Firewall blocked-connection events

    Noise injections:
      - 2%  duplicate timestamps
      - 5%  out-of-order rows (shuffle a random 5% window)
      - 3%  malformed/truncated JSON lines (windows only)
      - 5%  rows with non-UTC timezone offsets

    Parameters
    ----------
    event_times : np.ndarray — sorted event times in hours
    split_name  : str        — e.g. "train", "val", "test"
    output_dir  : str        — base directory (e.g. "data/raw")
    rng         : np.random.Generator
    """
    split_dir = pathlib.Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(event_times)
    if n == 0:
        logger.warning("emit_siem_logs called with empty event_times for split=%s", split_name)
        return

    # Assign ~60% to windows, ~20% edr, ~20% firewall
    assign = rng.choice(["win", "edr", "fw"], size=n, p=[0.60, 0.20, 0.20])

    # --- Windows Security Events ---
    win_times = event_times[assign == "win"]
    win_path = split_dir / f"windows_events_{split_name}.jsonl"
    win_records: List[dict] = []
    for t in win_times:
        eid = int(rng.choice(_WINDOWS_EVENT_IDS))
        tz = _TZ_OFFSETS[0]  # default UTC
        if rng.uniform() < 0.05:  # 5% non-UTC tz
            tz = rng.choice(_TZ_OFFSETS[1:])
        record = {
            "EventID": eid,
            "TimeCreated": _hours_to_iso(t, tz),
            "Computer": rng.choice(_HOSTNAMES),
            "SubjectUserName": rng.choice(_USERNAMES),
            "SubjectDomainName": "CORP",
            "IpAddress": _random_ip(rng),
            "ProcessName": rng.choice(_PROCESS_NAMES),
            "Severity": _SEVERITY_MAP.get(eid, "Low"),
            "LogonType": int(rng.integers(2, 11)),
            "TaskCategory": f"Category{int(rng.integers(1, 20))}",
        }
        win_records.append(record)

    # Inject 2% duplicate timestamps
    n_dup = max(1, int(0.02 * len(win_records)))
    dup_idxs = rng.choice(len(win_records), size=n_dup, replace=False) if len(win_records) > 0 else []
    for idx in dup_idxs:
        dup = dict(win_records[idx])
        win_records.append(dup)

    # Inject 5% out-of-order: shuffle a random 5% contiguous slice
    if len(win_records) > 20:
        n_oos = max(2, int(0.05 * len(win_records)))
        start_idx = int(rng.integers(0, len(win_records) - n_oos))
        chunk = win_records[start_idx: start_idx + n_oos]
        rng.shuffle(chunk)
        win_records[start_idx: start_idx + n_oos] = chunk

    with open(win_path, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(win_records):
            # Inject 3% malformed/truncated JSON
            line = json.dumps(rec)
            if rng.uniform() < 0.03:
                # Truncate at a random position
                trunc_at = max(1, int(rng.integers(5, max(6, len(line) - 1))))
                line = line[:trunc_at]
            fh.write(line + "\n")

    logger.info("Wrote %d windows event records to %s", len(win_records), win_path)

    # --- EDR CSV ---
    edr_times = event_times[assign == "edr"]
    edr_path = split_dir / f"edr_{split_name}.csv"
    edr_rows = []
    for t in edr_times:
        tz = _TZ_OFFSETS[0]
        if rng.uniform() < 0.05:
            tz = rng.choice(_TZ_OFFSETS[1:])
        edr_rows.append({
            "timestamp": _hours_to_iso(t, tz),
            "event_type": "process_create",
            "host": rng.choice(_HOSTNAMES),
            "source_ip": _random_ip(rng),
            "process_name": rng.choice(_PROCESS_NAMES),
            "parent_process": rng.choice(_PROCESS_NAMES),
            "user": rng.choice(_USERNAMES),
            "pid": int(rng.integers(100, 65535)),
            "ppid": int(rng.integers(100, 65535)),
            "severity": "Medium",
            "command_line_length": int(rng.integers(10, 512)),
        })

    # Inject 5% out-of-order
    if len(edr_rows) > 10:
        n_oos = max(1, int(0.05 * len(edr_rows)))
        s = int(rng.integers(0, len(edr_rows) - n_oos))
        chunk = edr_rows[s: s + n_oos]
        rng.shuffle(chunk)
        edr_rows[s: s + n_oos] = chunk

    pd.DataFrame(edr_rows).to_csv(edr_path, index=False)
    logger.info("Wrote %d EDR records to %s", len(edr_rows), edr_path)

    # --- Firewall CSV ---
    fw_times = event_times[assign == "fw"]
    fw_path = split_dir / f"firewall_{split_name}.csv"
    fw_ports = [445, 3389, 139, 22, 80, 443, 8080]
    fw_rows = []
    for t in fw_times:
        tz = _TZ_OFFSETS[0]
        if rng.uniform() < 0.05:
            tz = rng.choice(_TZ_OFFSETS[1:])
        fw_rows.append({
            "timestamp": _hours_to_iso(t, tz),
            "action": "BLOCK",
            "src_ip": _random_ip(rng, private=False),
            "dst_ip": _random_ip(rng, private=True),
            "src_port": int(rng.integers(1024, 65535)),
            "dst_port": int(rng.choice(fw_ports)),
            "protocol": rng.choice(["TCP", "UDP"]),
            "bytes_in": int(rng.integers(40, 65000)),
            "bytes_out": int(rng.integers(0, 1500)),
            "severity": "High",
            "rule_id": f"FW-RULE-{int(rng.integers(1, 200)):04d}",
        })

    # Inject 5% out-of-order
    if len(fw_rows) > 10:
        n_oos = max(1, int(0.05 * len(fw_rows)))
        s = int(rng.integers(0, len(fw_rows) - n_oos))
        chunk = fw_rows[s: s + n_oos]
        rng.shuffle(chunk)
        fw_rows[s: s + n_oos] = chunk

    pd.DataFrame(fw_rows).to_csv(fw_path, index=False)
    logger.info("Wrote %d firewall records to %s", len(fw_rows), fw_path)


# ---------------------------------------------------------------------------
# Full scenario orchestration
# ---------------------------------------------------------------------------

def run_full_scenario(
    config_path: str = "config.yaml",
    rng: Optional[np.random.Generator] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Orchestrate end-to-end:
      1. Load config
      2. Generate Hawkes and kill-chain scenarios for train/val/test splits
      3. Emit SIEM logs per split
      4. Save data/raw/labels.csv

    All configuration comes from config.yaml — no hardcoded values.
    """
    cfg = load_config(config_path)

    sim_cfg = cfg.get("simulation", {})
    T = float(sim_cfg.get("T_hours", sim_cfg.get("horizon_hours", 168.0)))
    seed = int(cfg.get("seeds", {}).get("simulate", sim_cfg.get("seed", 42)))
    if output_dir is None:
        output_dir = cfg.get("paths", {}).get("raw_data", sim_cfg.get("output_dir", "data/raw"))
    labels_path = os.path.join(output_dir, "labels.csv")

    hawkes_cfg = cfg.get("hawkes", {})
    mu = float(hawkes_cfg.get("mu", sim_cfg.get("mu_background", 0.5)))
    alpha = float(hawkes_cfg.get("alpha", sim_cfg.get("hawkes_alpha", 0.6)))
    beta = float(hawkes_cfg.get("beta", sim_cfg.get("hawkes_beta", 1.2)))
    wave_boost = float(hawkes_cfg.get("wave_boost", sim_cfg.get("wave_mu_boost", 4.0)))
    wave_duration = float(hawkes_cfg.get("wave_duration_h", sim_cfg.get("wave_duration_h", 2.0)))
    wave_count = int(hawkes_cfg.get("wave_count", sim_cfg.get("wave_count", 5)))

    kc_cfg = cfg.get("killchain", {})
    mu_bg = float(kc_cfg.get("mu_background", sim_cfg.get("mu_background", 0.5)))
    kc_wave_count = int(kc_cfg.get("wave_count", sim_cfg.get("wave_count", 5)))
    delays_val = kc_cfg.get("stage_delays_h", sim_cfg.get("killchain_stage_delays", [0.5, 1.0, 1.5]))
    if isinstance(delays_val, dict):
        stage_delays = [float(v) for v in delays_val.values()]
    else:
        stage_delays = [float(d) for d in delays_val]
    power_law_exp = float(kc_cfg.get("power_law_exp", sim_cfg.get("power_law_exponent", 1.5)))

    splits = ["train", "val", "test"]

    master_rng = rng if rng is not None else np.random.default_rng(seed)

    all_label_rows: List[dict] = []

    for split in splits:
        split_seed = int(master_rng.integers(0, 2**31))
        split_rng = np.random.default_rng(split_seed)
        wave_rng = np.random.default_rng(int(master_rng.integers(0, 2**31)))

        # --- Hawkes family ---
        logger.info("Generating Hawkes scenario for split=%s", split)
        h_times, h_waves = generate_hawkes_scenario(
            T=T,
            mu=mu,
            alpha=alpha,
            beta=beta,
            wave_count=wave_count,
            wave_boost=wave_boost,
            wave_duration=wave_duration,
            n_waves_rng=wave_rng,
            rng=split_rng,
        )
        emit_siem_logs(h_times, f"hawkes_{split}", output_dir, split_rng)
        for wi, (ws, we) in enumerate(h_waves):
            all_label_rows.append({
                "scenario_family": "hawkes",
                "split": split,
                "wave_idx": wi,
                "wave_start_h": ws,
                "wave_end_h": we,
                "timestamp_h": ws,
                "label": f"hawkes_{split}_{wi}",
            })

        # --- Kill-chain family ---
        logger.info("Generating kill-chain scenario for split=%s", split)
        kc_times, kc_waves = generate_killchain_scenario(
            T=T,
            mu_background=mu_bg,
            wave_count=kc_wave_count,
            stage_delays=stage_delays,
            power_law_exp=power_law_exp,
            rng=split_rng,
        )
        emit_siem_logs(kc_times, f"killchain_{split}", output_dir, split_rng)
        for wi, (ws, we) in enumerate(kc_waves):
            all_label_rows.append({
                "scenario_family": "killchain",
                "split": split,
                "wave_idx": wi,
                "wave_start_h": ws,
                "wave_end_h": we,
                "timestamp_h": ws,
                "label": f"killchain_{split}_{wi}",
            })

    # Save labels CSV
    pathlib.Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    labels_df = pd.DataFrame(all_label_rows, columns=[
        "scenario_family", "split", "wave_idx", "wave_start_h", "wave_end_h", "timestamp_h", "label"
    ])
    labels_df.to_csv(labels_path, index=False)
    logger.info("Saved labels to %s (%d rows)", labels_path, len(labels_df))


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
simulate_poisson_homogeneous = simulate_homogeneous_poisson
simulate_poisson_diurnal = simulate_diurnal_poisson


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_scenario()
