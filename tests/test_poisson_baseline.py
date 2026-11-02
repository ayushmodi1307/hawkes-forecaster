"""
Tests for src/poisson_baseline.py — Rung 0, Rung 1, detectors, dispersion stats.
"""
import numpy as np
import pytest
from src.poisson_baseline import (
    fit_poisson,
    fit_poisson_diurnal,
    poisson_tail_alert,
    compute_cv,
    compute_fano_factor,
    ewma_rate_detector,
    sliding_window_detector,
)


def test_fit_poisson_mle():
    """Poisson MLE mu must equal N / T exactly."""
    events = np.array([1.0, 3.5, 6.0, 8.0, 10.0])
    T = 12.0
    res = fit_poisson(events, T)
    assert abs(res["mu"] - 5.0 / 12.0) < 1e-6
    assert "log_likelihood" in res
    assert "aic" in res
    assert "bic" in res
    assert res["converged"] is True


def test_fit_poisson_empty():
    """Empty events return zero mu and valid dict."""
    events = np.array([])
    T = 10.0
    res = fit_poisson(events, T)
    assert res["mu"] == 0.0
    assert res["converged"] is True


def test_compute_cv_poisson():
    """CV of exponential gaps from Poisson process is approx 1.0."""
    rng = np.random.default_rng(123)
    gaps = rng.exponential(scale=2.0, size=5000)
    events = np.cumsum(gaps)
    cv = compute_cv(events)
    assert abs(cv - 1.0) < 0.08


def test_compute_fano_poisson():
    """Fano factor for homogeneous Poisson window counts is approx 1.0."""
    rng = np.random.default_rng(456)
    T = 2000.0
    mu = 1.5
    times = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / mu)
        if t >= T:
            break
        times.append(t)
    events = np.array(times)
    fano = compute_fano_factor(events, T, window_h=1.0)
    assert abs(fano - 1.0) < 0.15


def test_fit_poisson_diurnal():
    """Diurnal fit produces 24 rate bins."""
    rng = np.random.default_rng(789)
    events = np.sort(rng.uniform(0.0, 72.0, 300))
    res = fit_poisson_diurnal(events, T=72.0, n_bins=24)
    assert len(res["bin_rates"]) == 24
    assert np.all(np.array(res["bin_rates"]) >= 0)


def test_poisson_tail_alert():
    """Tail alert returns True when observed count is extreme."""
    # Under mu=0.5, count=15 in 1h is extreme
    res = poisson_tail_alert(k_obs=15, mu_hat=0.5, window_h=1.0, alpha_level=0.01)
    assert res["is_alert"] is True
    assert res["p_value"] < 0.01

    # Normal count should not alert
    res_norm = poisson_tail_alert(k_obs=1, mu_hat=0.5, window_h=1.0, alpha_level=0.01)
    assert res_norm["is_alert"] is False


def test_ewma_and_sliding_detectors():
    """Detectors run without errors and return expected indicators."""
    events = np.array([1.0, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0])
    ewma_rate = ewma_rate_detector(events, t_now=2.0, half_life_h=1.0)
    assert isinstance(ewma_rate, float)
    assert ewma_rate > 0

    sw_res = sliding_window_detector(events, t_now=2.0, window_h=1.0, threshold_count=3)
    assert isinstance(sw_res["is_alert"], (bool, np.bool_))
    assert sw_res["count"] >= 3
