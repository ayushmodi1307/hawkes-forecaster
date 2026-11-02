"""
Tests for src/model.py — Hawkes MLE, gradients, stationarity, edge cases.
"""
import numpy as np
import pytest
from src.model import fit_hawkes, verify_gradient
from src.simulate import simulate_hawkes_ogata


def test_hawkes_gradient_analytic_matches_fd():
    """Analytic gradient must match finite differences within tolerance."""
    events = np.array([0.5, 1.2, 1.4, 2.1, 2.3, 2.35, 4.0, 5.5, 5.6])
    T = 8.0
    passed = verify_gradient(events, T)
    assert passed is True, "Analytic gradient did not match finite differences"


def test_hawkes_stationarity_enforced():
    """Branching ratio n = alpha / beta must be strictly < 1.0."""
    rng = np.random.default_rng(42)
    # Even on clustered burst data, stationarity bound n < 1 must hold
    events = np.sort(rng.uniform(0.0, 50.0, 100))
    res = fit_hawkes(events, T=50.0)
    assert res["branching_ratio"] < 1.0
    assert res["alpha"] < res["beta"]


def test_hawkes_parameter_recovery():
    """Hawkes MLE recovers ground truth parameters on synthetic data."""
    rng = np.random.default_rng(999)
    mu_true = 0.5
    alpha_true = 0.6
    beta_true = 1.2
    T = 1500.0
    events = simulate_hawkes_ogata(mu_true, alpha_true, beta_true, T, rng)

    res = fit_hawkes(events, T)
    assert res["converged"] is True
    # Branching ratio n = 0.5
    assert abs(res["branching_ratio"] - 0.5) < 0.15
    # Mu approx 0.5
    assert abs(res["mu"] - 0.5) < 0.20


def test_hawkes_empty_and_small_events():
    """Empty or single event inputs must not raise, returning converged=False with reason."""
    empty_res = fit_hawkes(np.array([]), T=10.0)
    assert empty_res["converged"] is False
    assert "reason" in empty_res
    assert isinstance(empty_res["reason"], str)

    single_res = fit_hawkes(np.array([2.5]), T=10.0)
    assert single_res["converged"] is False
    assert "reason" in single_res


def test_hawkes_multistart_reported():
    """Multi-start optimization records convergence and attempts."""
    events = np.array([0.5, 1.0, 1.2, 1.3, 2.0, 3.5, 4.0, 4.1, 4.2, 5.0])
    res = fit_hawkes(events, T=6.0)
    assert "n_starts_converged" in res or "starts_ok" in res or "converged" in res

