"""
src/forecast.py — Hawkes conditional intensity forecasting.

Conditional intensity at time t (given history):
  λ(t) = μ + Σ_{tᵢ<t} α exp(−β(t−tᵢ))

Steady-state mean: η = μβ/(β−α)  [= μ/(1−n) since β−α = β(1−n)]

Conditional forecast (exact closed form):
  E[λ(t+s)|H_t] = η + (λ(t) − η) exp(−(β−α)s)

Expected count over horizon [t, t+Δ]:
  E[N(t,t+Δ)|H_t] = ηΔ + (λ(t)−η)/(β−α) · (1 − exp(−(β−α)Δ))

Deterministic no-new-events lower bound:
  λ_lb(t+s) = μ + Σ_{tᵢ<t} α exp(−β(t+s−tᵢ))   [no new events after t]

Monte Carlo:
  Forward-simulate K Hawkes paths from history H_t, record N(t,t+Δ) for each.
  P(N≥k) estimated from empirical distribution.
  Also compute naive Poisson tail: P(Poisson(E[N])≥k) for comparison.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core intensity helpers
# ---------------------------------------------------------------------------

def current_intensity(
    event_times: np.ndarray,
    t: float,
    mu: float,
    alpha: float,
    beta: float,
) -> float:
    """Compute conditional intensity λ(t) given history.

    λ(t) = μ + Σ_{tᵢ < t} α exp(−β(t − tᵢ))

    Uses O(n) vectorised computation over past events.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted array of past event timestamps (all must be < t).
    t : float
        Query time.
    mu, alpha, beta : float
        Hawkes parameters (mu>0, alpha>0, beta>alpha for stationarity).

    Returns
    -------
    float
        λ(t) ≥ mu.
    """
    event_times = np.asarray(event_times, dtype=float)
    past = event_times[event_times < t]
    excitation = float(np.sum(alpha * np.exp(-beta * (t - past))))
    return mu + excitation


def _steady_state_intensity(mu: float, alpha: float, beta: float) -> float:
    """Steady-state (unconditional) mean intensity η = μ / (1 − α/β)."""
    n = alpha / beta if beta > 0 else 0.999
    if n >= 1.0:
        n = 0.999
    return mu / (1.0 - n)


# ---------------------------------------------------------------------------
# Closed-form conditional forecast
# ---------------------------------------------------------------------------

def expected_intensity_curve(
    t_now: float,
    horizon_h: Union[float, np.ndarray] = 2.0,
    n_steps: int = 50,
    event_times: Optional[np.ndarray] = None,
    mu: float = 0.5,
    alpha: float = 0.6,
    beta: float = 1.2,
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Closed-form conditional intensity forecast curve.

    E[λ(t_now+s)|H_{t_now}] = η + (λ(t_now) − η) exp(−(β−α)s)
    """
    # Check if called as expected_intensity_curve(lam_now, s_grid, mu=..., alpha=..., beta=...)
    if isinstance(horizon_h, np.ndarray):
        lam_now = float(t_now)
        s_vals = horizon_h
        mu_val = kwargs.get("mu", mu)
        alpha_val = kwargs.get("alpha", alpha)
        beta_val = kwargs.get("beta", beta)
        eta = _steady_state_intensity(mu_val, alpha_val, beta_val)
        decay = beta_val - alpha_val if beta_val > alpha_val else 1e-4
        intensities = eta + (lam_now - eta) * np.exp(-decay * s_vals)
        return intensities

    eta = _steady_state_intensity(mu, alpha, beta)
    if event_times is not None:
        lam_now = current_intensity(event_times, t_now, mu, alpha, beta)
    else:
        lam_now = mu
    decay = beta - alpha if beta > alpha else 1e-4

    s_vals = np.linspace(0.0, float(horizon_h), int(n_steps))
    intensities = eta + (lam_now - eta) * np.exp(-decay * s_vals)
    times = t_now + s_vals
    return times, intensities


def expected_count(
    t_now: Union[float, np.ndarray],
    horizon_h: float = 2.0,
    event_times: Optional[Union[float, np.ndarray]] = None,
    mu: float = 0.5,
    alpha: float = 0.6,
    beta: float = 1.2,
    delta: Optional[float] = None,
) -> float:
    """Closed-form expected event count over [t_now, t_now + horizon_h].

    E[N(t_now, t_now+Δ)|H_{t_now}]
        = η·Δ + (λ(t_now) − η) / (β−α) · (1 − exp(−(β−α)·Δ))
    """
    # Check if called as expected_count(event_times, t_now, delta, mu, alpha, beta)
    if isinstance(t_now, (np.ndarray, list)):
        ev_arr = np.asarray(t_now, dtype=float)
        real_t_now = float(horizon_h)
        real_delta = float(event_times) if event_times is not None else (float(delta) if delta is not None else 2.0)
    else:
        real_t_now = float(t_now)
        real_delta = float(delta if delta is not None else horizon_h)
        ev_arr = np.asarray(event_times, dtype=float) if event_times is not None else np.array([])

    if alpha == 0.0 or beta <= alpha:
        return float(mu * real_delta)

    eta = _steady_state_intensity(mu, alpha, beta)
    lam_now = current_intensity(ev_arr, real_t_now, mu, alpha, beta)
    decay = beta - alpha

    mean_count = eta * real_delta + (lam_now - eta) / decay * (1.0 - math.exp(-decay * real_delta))
    return float(max(mean_count, 0.0))


def no_new_events_lower_bound(
    t_now: float,
    horizon_h: Union[float, np.ndarray] = 2.0,
    n_steps: int = 50,
    event_times: Optional[np.ndarray] = None,
    mu: float = 0.5,
    alpha: float = 0.6,
    beta: float = 1.2,
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Deterministic lower-bound intensity assuming no new events after t_now."""
    # Check if called as no_new_events_lower_bound(lam_now, s_grid, mu=..., alpha=..., beta=...)
    if isinstance(horizon_h, np.ndarray):
        lam_now = float(t_now)
        s_vals = horizon_h
        mu_val = kwargs.get("mu", mu)
        beta_val = kwargs.get("beta", beta)
        exc_now = max(0.0, lam_now - mu_val)
        return mu_val + exc_now * np.exp(-beta_val * s_vals)

    event_times = np.asarray(event_times, dtype=float) if event_times is not None else np.array([])
    past = event_times[event_times <= t_now] if len(event_times) > 0 else np.array([])
    exc_now = float(np.sum(alpha * np.exp(-beta * (t_now - past)))) if len(past) > 0 else 0.0

    s_vals = np.linspace(0.0, float(horizon_h), int(n_steps))
    intensities = mu + exc_now * np.exp(-beta * s_vals)
    times = t_now + s_vals
    return times, intensities


# ---------------------------------------------------------------------------
# Monte Carlo forward simulation
# ---------------------------------------------------------------------------

def _simulate_hawkes_from_history(
    event_times: np.ndarray,
    t_now: float,
    horizon_h: float,
    mu: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ogata thinning for Hawkes continuation from existing history.

    Starts from the conditional intensity at t_now and simulates new events
    in (t_now, t_now + horizon_h].

    Parameters
    ----------
    event_times : np.ndarray
        Past event times (all ≤ t_now).
    t_now : float
        Start of the forecast window.
    horizon_h : float
        Length of forecast window.
    mu, alpha, beta : float
        Hawkes parameters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of new event times in (t_now, t_now + horizon_h].
    """
    T_end = t_now + horizon_h
    past = list(event_times[event_times <= t_now])

    # Upper bound on intensity
    lam_star = mu + sum(alpha * math.exp(-beta * (t_now - s)) for s in past)
    lam_star = max(lam_star, mu)

    new_events = []
    t = t_now

    while t < T_end:
        # Inter-arrival from intensity upper bound
        u1 = rng.uniform()
        w = -math.log(u1) / lam_star
        t_cand = t + w

        if t_cand > T_end:
            break

        # Compute true intensity at t_cand
        all_past = past + new_events
        lam_cand = mu + sum(alpha * math.exp(-beta * (t_cand - s)) for s in all_past)

        u2 = rng.uniform()
        if u2 <= lam_cand / lam_star:
            new_events.append(t_cand)
            lam_star = lam_cand + alpha
        else:
            lam_star = max(lam_cand, mu)

        t = t_cand

    return np.array(new_events)


def monte_carlo_forecast(
    t_now: Union[float, np.ndarray],
    horizon_h: float = 2.0,
    event_times: Optional[Union[float, np.ndarray]] = None,
    mu: float = 0.5,
    alpha: float = 0.6,
    beta: float = 1.2,
    K: int = 1000,
    rng_seed: int = 42,
    delta: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    """Monte Carlo forecast for count distribution over [t_now, t_now+horizon_h]."""
    if seed is not None:
        rng_seed = seed
    if isinstance(t_now, (np.ndarray, list)):
        ev_arr = np.asarray(t_now, dtype=float)
        real_t_now = float(horizon_h)
        real_horizon = float(event_times) if event_times is not None else (float(delta) if delta is not None else 2.0)
    else:
        real_t_now = float(t_now)
        real_horizon = float(delta if delta is not None else horizon_h)
        ev_arr = np.asarray(event_times, dtype=float) if event_times is not None else np.array([])

    rng = np.random.default_rng(rng_seed)
    count_samples = np.zeros(K, dtype=int)

    for i in range(K):
        new_ev = _simulate_hawkes_from_history(
            ev_arr, real_t_now, real_horizon, mu, alpha, beta, rng
        )
        count_samples[i] = len(new_ev)

    mean_count = float(np.mean(count_samples))

    quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles = {q: float(np.quantile(count_samples, q)) for q in quantile_levels}

    # Empirical exceedance probabilities
    p_exceed = {}
    naive_poisson_p_exceed = {}
    gap_mc_vs_poisson = {}
    lam_naive = max(mean_count, 1e-9)
    for k in range(1, 21):
        p_mc = float(np.mean(count_samples >= k))
        p_pois = float(1.0 - stats.poisson.cdf(k - 1, lam_naive))
        p_exceed[k] = p_mc
        naive_poisson_p_exceed[k] = p_pois
        gap_mc_vs_poisson[k] = round(p_mc - p_pois, 6)

    return {
        "count_samples": count_samples,
        "counts": count_samples,
        "mean_count": mean_count,
        "quantiles": quantiles,
        "p_exceed": p_exceed,
        "naive_poisson_p_exceed": naive_poisson_p_exceed,
        "gap_mc_vs_poisson": gap_mc_vs_poisson,
    }


# ---------------------------------------------------------------------------
# High-level forecast entry point
# ---------------------------------------------------------------------------

def forecast(
    event_times: np.ndarray,
    t_now: float,
    horizons: list,
    mu: float,
    alpha: float,
    beta: float,
    K: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """Run forecasts for multiple horizons.

    For each horizon Δ in `horizons`, returns a sub-dict containing:
      - expected_count      : closed-form E[N(t_now, t_now+Δ)|H_{t_now}]
      - current_intensity   : λ(t_now)
      - peak_intensity      : max of the conditional intensity curve
      - mc_results          : output of monte_carlo_forecast()
      - lower_bound_count   : integral of no-new-events curve (numerical)
      - intensity_curve     : (times, intensities) from expected_intensity_curve()

    Parameters
    ----------
    event_times : np.ndarray
        Past event times (sorted, ≤ t_now).
    t_now : float
        Conditioning time.
    horizons : list of float
        Forecast horizons Δ.
    mu, alpha, beta : float
        Hawkes parameters.
    K : int
        Monte Carlo sample count.
    rng_seed : int
        Random seed.

    Returns
    -------
    dict keyed by horizon value.
    """
    event_times = np.asarray(event_times, dtype=float)
    lam_now = current_intensity(event_times, t_now, mu, alpha, beta)

    results = {}
    for h in horizons:
        h = float(h)

        # Closed-form expected count
        ec = expected_count(t_now, h, event_times, mu, alpha, beta)

        # Intensity curve (100 steps)
        n_steps = max(100, int(h * 10))
        times_curve, int_curve = expected_intensity_curve(
            t_now, h, n_steps, event_times, mu, alpha, beta
        )
        peak_intensity = float(np.max(int_curve))

        # No-new-events lower bound count (trapezoidal integral)
        times_lb, int_lb = no_new_events_lower_bound(
            t_now, h, n_steps, event_times, mu, alpha, beta
        )
        lower_bound_count = float(np.trapz(int_lb, times_lb))

        # Monte Carlo
        mc = monte_carlo_forecast(
            t_now, h, event_times, mu, alpha, beta, K=K, rng_seed=rng_seed
        )

        results[h] = {
            "expected_count": ec,
            "current_intensity": lam_now,
            "peak_intensity": peak_intensity,
            "mc_results": mc,
            "lower_bound_count": lower_bound_count,
            "intensity_curve": (times_curve, int_curve),
        }

    return results
