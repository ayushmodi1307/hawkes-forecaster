"""
Tests for src/forecast.py — closed-form forecast, Monte Carlo continuations, bounds.
"""
import numpy as np
import pytest
from src.forecast import (
    current_intensity,
    expected_intensity_curve,
    expected_count,
    no_new_events_lower_bound,
    monte_carlo_forecast,
)


def test_current_intensity_at_event():
    """current_intensity matches analytical sum."""
    events = np.array([1.0, 2.0])
    t = 2.5
    mu, alpha, beta = 0.5, 0.4, 1.0
    lam = current_intensity(events, t, mu, alpha, beta)
    expected_lam = mu + alpha * (np.exp(-beta * (2.5 - 1.0)) + np.exp(-beta * (2.5 - 2.0)))
    assert abs(lam - expected_lam) < 1e-9


def test_closed_form_matches_mc_mean():
    """Closed-form expected count matches Monte Carlo continuation mean."""
    events = np.array([1.0, 1.8, 2.1, 2.3, 2.4])
    t_now = 2.5
    delta = 2.0
    mu, alpha, beta = 0.5, 0.6, 1.2
    cf_count = expected_count(events, t_now, delta, mu, alpha, beta)

    mc_res = monte_carlo_forecast(
        events, t_now, delta, mu, alpha, beta, K=1500, seed=42
    )
    mc_mean = float(np.mean(mc_res["counts"]))

    # Standard error of MC mean: std / sqrt(K)
    mc_se = float(np.std(mc_res["counts"]) / np.sqrt(1500))
    assert abs(cf_count - mc_mean) < 3.0 * mc_se + 0.15


def test_no_new_events_lower_bound():
    """Deterministic no-new-events lower bound <= expected intensity curve."""
    events = np.array([1.0, 1.5, 1.9])
    t_now = 2.0
    mu, alpha, beta = 0.5, 0.6, 1.2
    s_grid = np.linspace(0.1, 4.0, 20)
    lam_now = current_intensity(events, t_now, mu, alpha, beta)

    exp_curve = expected_intensity_curve(lam_now, s_grid, mu, alpha, beta)
    lb_curve = no_new_events_lower_bound(lam_now, s_grid, mu, alpha, beta)

    assert np.all(lb_curve <= exp_curve + 1e-9)


def test_alpha_zero_expected_count():
    """When alpha=0, expected count collapses exactly to mu * delta."""
    events = np.array([1.0, 2.0, 3.0])
    t_now = 4.0
    delta = 3.0
    mu = 0.7
    cnt = expected_count(events, t_now, delta, mu=mu, alpha=0.0, beta=1.0)
    assert abs(cnt - mu * delta) < 1e-9

