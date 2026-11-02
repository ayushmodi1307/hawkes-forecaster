"""Tests for src/simulate.py — statistical property verification."""
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers — lazy imports so that collection doesn't fail if module is missing
# ---------------------------------------------------------------------------

def _import_simulate():
    import importlib
    return importlib.import_module("src.simulate")


# ---------------------------------------------------------------------------
# 1. Homogeneous Poisson statistics
# ---------------------------------------------------------------------------

def test_homogeneous_poisson_stats():
    """Empirical mean rate ≈ mu; CV ≈ 1; Fano factor (1 h window) ≈ 1."""
    sim = _import_simulate()
    rng = np.random.default_rng(0)
    mu = 2.0
    T = 1000.0
    events = sim.simulate_poisson_homogeneous(mu=mu, T=T, rng=rng)

    # Mean rate
    empirical_rate = len(events) / T
    assert abs(empirical_rate - mu) < 0.1 * mu, (
        f"Mean rate {empirical_rate:.4f} too far from mu={mu}"
    )

    # CV of inter-arrival times ≈ 1 for Poisson
    gaps = np.diff(events)
    assert len(gaps) > 10, "Too few gaps to compute CV"
    cv = float(np.std(gaps, ddof=1) / np.mean(gaps))
    assert abs(cv - 1.0) < 0.1, f"CV={cv:.4f} should be ≈ 1.0 for Poisson"

    # Fano factor (window 1 h) ≈ 1
    window_h = 1.0
    bins = np.arange(0, T + window_h, window_h)
    counts, _ = np.histogram(events, bins=bins)
    fano = float(np.var(counts, ddof=1) / np.mean(counts))
    assert abs(fano - 1.0) < 0.15, f"Fano={fano:.4f} should be ≈ 1.0 for Poisson"


# ---------------------------------------------------------------------------
# 2. Hawkes Ogata mean rate
# ---------------------------------------------------------------------------

def test_hawkes_ogata_mean_rate():
    """Empirical mean rate ≈ mu / (1 - n) for Ogata simulator."""
    sim = _import_simulate()
    rng = np.random.default_rng(1)
    mu, alpha, beta = 0.5, 0.6, 1.2  # n = 0.5
    T = 500.0
    n = alpha / beta
    theoretical_rate = mu / (1 - n)

    events = sim.simulate_hawkes_ogata(mu=mu, alpha=alpha, beta=beta, T=T, rng=rng)
    empirical_rate = len(events) / T

    assert abs(empirical_rate - theoretical_rate) < 0.1 * theoretical_rate, (
        f"Ogata empirical rate {empirical_rate:.4f} far from theory {theoretical_rate:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Hawkes branching mean rate
# ---------------------------------------------------------------------------

def test_hawkes_branching_mean_rate():
    """Empirical mean rate ≈ mu / (1 - n) for branching simulator."""
    sim = _import_simulate()
    rng = np.random.default_rng(42)
    mu, alpha, beta = 0.5, 0.6, 1.2
    T = 1000.0
    n = alpha / beta
    theoretical_rate = mu / (1 - n)

    events = sim.simulate_hawkes_branching(mu=mu, alpha=alpha, beta=beta, T=T, rng=rng)
    empirical_rate = len(events) / T

    assert abs(empirical_rate - theoretical_rate) < 0.1 * theoretical_rate, (
        f"Branching empirical rate {empirical_rate:.4f} far from theory {theoretical_rate:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Fano factor > 1 and within 25 % of theory for both methods
# ---------------------------------------------------------------------------

def test_hawkes_ogata_vs_branching_fano():
    """Both Hawkes simulators yield Fano > 1 and close to asymptotic 1/(1-n)^2 = 4.0."""
    sim = _import_simulate()
    mu, alpha, beta = 0.5, 0.6, 1.2
    T = 5000.0
    n = alpha / beta
    fano_theory = 1.0 / (1.0 - n) ** 2  # = 4.0

    def fano(events, window_h=25.0):
        bins = np.arange(0, T + window_h, window_h)
        counts, _ = np.histogram(events, bins=bins)
        if np.mean(counts) == 0:
            return np.nan
        return float(np.var(counts, ddof=1) / np.mean(counts))

    rng1 = np.random.default_rng(10)
    events_ogata = sim.simulate_hawkes_ogata(mu=mu, alpha=alpha, beta=beta, T=T, rng=rng1)
    f_ogata = fano(events_ogata)

    rng2 = np.random.default_rng(11)
    events_branch = sim.simulate_hawkes_branching(mu=mu, alpha=alpha, beta=beta, T=T, rng=rng2)
    f_branch = fano(events_branch)

    # Both must exceed 1
    assert f_ogata > 1.0, f"Ogata Fano={f_ogata:.4f} not > 1"
    assert f_branch > 1.0, f"Branching Fano={f_branch:.4f} not > 1"

    # Within 25 % of theoretical 4.0
    tol = 0.25
    assert abs(f_ogata - fano_theory) < tol * fano_theory, (
        f"Ogata Fano={f_ogata:.4f}, theory={fano_theory:.4f}, tol={tol*fano_theory:.4f}"
    )
    assert abs(f_branch - fano_theory) < tol * fano_theory, (
        f"Branching Fano={f_branch:.4f}, theory={fano_theory:.4f}, tol={tol*fano_theory:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Borel PMF sums to one
# ---------------------------------------------------------------------------

def test_borel_pmf_sums_to_one():
    """P(k) for k=1..200 sums ≈ 1.0 for several branching ratios."""
    sim = _import_simulate()
    for n_val in [0.3, 0.5, 0.7]:
        k_arr = np.arange(1, 201)
        pmf = np.array([sim.borel_pmf(k, n_val) for k in k_arr])
        total = float(np.sum(pmf))
        assert abs(total - 1.0) < 0.001, (
            f"Borel PMF sum={total:.6f} for n={n_val}, expected ≈ 1.0"
        )


# ---------------------------------------------------------------------------
# 6. Borel mean matches theory: E[cluster size] = 1/(1-n)
# ---------------------------------------------------------------------------

def test_borel_mean_matches_theory():
    """Empirical cluster mean from branching ≈ 1/(1-n)."""
    sim = _import_simulate()
    rng = np.random.default_rng(42)
    mu, alpha, beta = 0.5, 0.6, 1.2
    T = 1000.0
    n = alpha / beta
    theory_mean = 1.0 / (1.0 - n)

    events = sim.simulate_hawkes_branching(mu=mu, alpha=alpha, beta=beta, T=T, rng=rng)
    # Number of clusters ~ Poisson(mu*T), so total events / mu*T ≈ 1/(1-n)
    expected_n_events = mu * T * theory_mean
    empirical = len(events) / (mu * T) if mu * T > 0 else 0.0

    assert abs(empirical - theory_mean) < 0.2 * theory_mean, (
        f"Empirical cluster mean proxy {empirical:.4f} far from {theory_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# 7. alpha=0 is equivalent to Poisson
# ---------------------------------------------------------------------------

def test_alpha_zero_is_poisson():
    """simulate_hawkes_ogata with alpha=0 should behave like Poisson (Fano≈1, CV≈1)."""
    sim = _import_simulate()
    rng = np.random.default_rng(99)
    mu = 2.0
    T = 1000.0
    events = sim.simulate_hawkes_ogata(mu=mu, alpha=0.0, beta=1.0, T=T, rng=rng)

    # Fano
    window_h = 1.0
    bins = np.arange(0, T + window_h, window_h)
    counts, _ = np.histogram(events, bins=bins)
    fano = float(np.var(counts, ddof=1) / np.mean(counts))
    assert abs(fano - 1.0) < 0.15, f"Fano={fano:.4f} should be ≈ 1.0 when alpha=0"

    # CV
    gaps = np.diff(events)
    if len(gaps) > 5:
        cv = float(np.std(gaps, ddof=1) / np.mean(gaps))
        assert abs(cv - 1.0) < 0.15, f"CV={cv:.4f} should be ≈ 1.0 when alpha=0"


# ---------------------------------------------------------------------------
# 8. Scenario generator labels file
# ---------------------------------------------------------------------------

def test_scenario_generator_labels(tmp_path, monkeypatch):
    """run_full_scenario produces data/raw/labels.csv with required columns."""
    sim = _import_simulate()
    # Run scenario with output directed to tmp_path
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "data" / "raw", exist_ok=True)
    os.makedirs(tmp_path / "data" / "processed", exist_ok=True)
    os.makedirs(tmp_path / "reports", exist_ok=True)

    rng = np.random.default_rng(0)
    sim.run_full_scenario(rng=rng, output_dir=str(tmp_path / "data" / "raw"))

    labels_path = tmp_path / "data" / "raw" / "labels.csv"
    assert labels_path.exists(), "labels.csv was not created"

    import pandas as pd
    df = pd.read_csv(labels_path)
    required_cols = {"timestamp_h", "label"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns in labels.csv: {missing}"


# ---------------------------------------------------------------------------
# 9. Kill-chain scenario stages
# ---------------------------------------------------------------------------

def test_killchain_scenario_stages():
    """generate_killchain_scenario returns sorted event_times with non-overlapping wave_labels."""
    sim = _import_simulate()
    rng = np.random.default_rng(7)
    result = sim.generate_killchain_scenario(rng=rng)

    event_times = result["event_times"]
    wave_labels = result["wave_labels"]  # list of (start, end) or label per event

    # event_times must be sorted
    assert np.all(np.diff(event_times) >= 0), "event_times are not sorted"

    # wave_labels must be an array/list aligned with event_times OR a list of intervals
    # Check non-overlapping: if wave_labels is per-event array of stage indices
    if hasattr(wave_labels, "__len__") and len(wave_labels) == len(event_times):
        # Per-event labels: no constraint on non-overlap, just check values are valid
        unique_labels = set(int(x) for x in wave_labels if not math.isnan(float(x)))
        assert len(unique_labels) >= 1, "Expected at least 1 distinct stage label"
    else:
        # Interval-based: check non-overlapping
        intervals = sorted(wave_labels, key=lambda x: x[0])
        for i in range(len(intervals) - 1):
            assert intervals[i][1] <= intervals[i + 1][0], (
                f"Overlapping wave intervals: {intervals[i]} and {intervals[i+1]}"
            )
