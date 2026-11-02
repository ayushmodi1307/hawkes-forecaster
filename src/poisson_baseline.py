"""
src/poisson_baseline.py — Rung 0 and Rung 1 Poisson baseline models.

Rung 0: Homogeneous Poisson, μ̂ = N/T
  logL = N log μ̂ − μ̂ T
  CV(inter-arrivals) = 1, Fano(window Δ) = 1 for all Δ.

Rung 1: Diurnal inhomogeneous Poisson, 24 hourly bins with L2 regularization.
  The bin rates μ_h are estimated by penalised ML:
    argmin −logL + λ_reg Σ μ_h²

Poisson tail alert (with Bonferroni correction):
  p = P(Poisson(μ̂ Δ) ≥ k_obs) = 1 − CDF(k_obs − 1)
  Corrected threshold: α_adj = α / n_tests
"""
import logging
import math
from typing import Optional

import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rung 0: Homogeneous Poisson MLE
# ---------------------------------------------------------------------------

def fit_poisson(event_times: np.ndarray, T: float) -> dict:
    """
    Fit a homogeneous Poisson process (Rung 0) via MLE.

    MLE solution: μ̂ = N / T.

    Goodness-of-fit diagnostics:
      - CV   (coefficient of variation of inter-arrival times) should be ≈ 1
      - Fano factor over window Δ=1h should be ≈ 1

    Parameters
    ----------
    event_times : np.ndarray — sorted event times in [0, T] (hours)
    T           : float      — observation horizon (hours)

    Returns
    -------
    dict with keys:
        mu              : float — MLE rate estimate
        log_likelihood  : float
        aic             : float — Akaike Information Criterion (k=1)
        bic             : float — Bayesian Information Criterion
        cv              : float — CV of inter-arrivals
        fano            : float — empirical Fano factor (Δ=1h)
        n_events        : int
    """
    times = np.asarray(event_times, dtype=float)
    N = len(times)

    if N == 0:
        return dict(mu=0.0, log_likelihood=-np.inf, aic=np.inf, bic=np.inf,
                    cv=np.nan, fano=np.nan, n_events=0, converged=True)

    mu_hat = N / T
    log_lik = N * math.log(mu_hat) - mu_hat * T  # homogeneous Poisson log-likelihood

    k = 1  # one free parameter
    aic = 2 * k - 2 * log_lik
    bic = k * math.log(N) - 2 * log_lik if N > 0 else np.inf

    cv = compute_cv(times)
    fano = compute_fano_factor(times, T, window_h=1.0)

    return dict(
        mu=mu_hat,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
        cv=cv,
        fano=fano,
        n_events=N,
        converged=True,
    )


# ---------------------------------------------------------------------------
# Rung 1: Diurnal inhomogeneous Poisson (penalised MLE)
# ---------------------------------------------------------------------------

def fit_poisson_diurnal(
    event_times: np.ndarray,
    T: float,
    n_bins: int = 24,
    reg_lambda: float = 0.01,
) -> dict:
    """
    Fit a diurnal inhomogeneous Poisson process (Rung 1) with penalised MLE.

    The observation window is divided into *n_bins* bins of width Δ = T / n_bins.
    Each bin h ∈ {0, …, n_bins−1} has rate μ_h (events per hour).

    Penalised log-likelihood:
        ℓ_pen = Σ_h [k_h log(μ_h Δ) − μ_h Δ] − λ_reg Σ_h μ_h²

    Optimised via L-BFGS-B in log-space (θ_h = log μ_h) to enforce μ_h > 0.

    Parameters
    ----------
    event_times : np.ndarray — sorted event times in [0, T]
    T           : float      — observation horizon (hours)
    n_bins      : int        — number of time bins (default 24 = hourly)
    reg_lambda  : float      — L2 regularisation strength

    Returns
    -------
    dict with keys:
        bin_rates      : np.ndarray (len n_bins) — μ_h for each bin
        log_likelihood : float — unpenalised log-likelihood at solution
        aic            : float — AIC (k = n_bins)
        bic            : float — BIC
    """
    times = np.asarray(event_times, dtype=float)
    N = len(times)
    bin_width = T / n_bins  # hours per bin

    # Assign each event to a bin index
    bin_indices = np.floor(times / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Event counts per bin
    k_h = np.bincount(bin_indices, minlength=n_bins).astype(float)

    # Penalised negative log-likelihood in θ = log(μ_h) space
    def neg_penalised_loglik(theta: np.ndarray) -> float:
        mu_h = np.exp(theta)
        # Poisson log-likelihood contribution per bin: k_h log(μ_h Δ) - μ_h Δ
        # Handle k_h = 0 carefully (0 * log(μ_h Δ) = 0)
        with np.errstate(divide="ignore"):
            log_terms = np.where(k_h > 0, k_h * np.log(mu_h * bin_width), 0.0)
        ll = np.sum(log_terms - mu_h * bin_width)
        penalty = reg_lambda * np.sum(mu_h ** 2)
        return -(ll - penalty)

    def neg_penalised_loglik_grad(theta: np.ndarray) -> np.ndarray:
        mu_h = np.exp(theta)
        # d/d_theta_h [-ll + penalty] = d/d_mu_h * mu_h
        # d(ll)/d(mu_h) = k_h / mu_h - Δ
        # d(penalty)/d(mu_h) = 2 λ mu_h
        # Chain rule: multiply by mu_h (Jacobian of exp)
        grad_mu = -(k_h / np.maximum(mu_h, 1e-12) - bin_width - 2 * reg_lambda * mu_h)
        return grad_mu * mu_h  # chain rule: dL/d_theta = dL/d_mu * mu

    # Initial guess: uniform rate from MLE
    mu0 = N / T if N > 0 else 1.0
    theta0 = np.full(n_bins, math.log(max(mu0, 1e-6)))

    result = optimize.minimize(
        neg_penalised_loglik,
        theta0,
        jac=neg_penalised_loglik_grad,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    mu_h_opt = np.exp(result.x)

    # Compute unpenalised log-likelihood at solution
    with np.errstate(divide="ignore"):
        log_terms = np.where(k_h > 0, k_h * np.log(mu_h_opt * bin_width), 0.0)
    log_lik = float(np.sum(log_terms - mu_h_opt * bin_width))

    k_params = n_bins
    aic = 2 * k_params - 2 * log_lik
    bic = k_params * math.log(max(N, 1)) - 2 * log_lik

    return dict(
        bin_rates=mu_h_opt,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
    )


# ---------------------------------------------------------------------------
# Poisson tail alert with Bonferroni correction
# ---------------------------------------------------------------------------

def poisson_tail_alert(
    k_obs: int,
    mu_hat: float,
    window_h: float,
    alpha_level: float = 0.05,
    n_tests: int = 1,
) -> dict:
    """
    One-sided Poisson tail alert: test whether k_obs events in a window of
    length *window_h* hours is extreme under H₀: count ~ Poisson(μ̂ Δ).

    With Bonferroni correction for multiple simultaneous windows:
        α_adj = α / n_tests
        p = P(Poisson(μ̂ Δ) ≥ k_obs) = 1 − CDF(k_obs − 1)

    Parameters
    ----------
    k_obs       : int   — observed event count in window
    mu_hat      : float — estimated rate (events per hour)
    window_h    : float — window length (hours)
    alpha_level : float — nominal significance level (default 0.05)
    n_tests     : int   — number of simultaneous windows tested (Bonferroni)

    Returns
    -------
    dict with keys:
        p_value        : float — tail probability
        corrected_alpha: float — Bonferroni-corrected threshold
        is_alert       : bool  — True if p_value < corrected_alpha
        expected_count : float — μ̂ × Δ
    """
    expected = mu_hat * window_h
    # P(X >= k) = 1 - P(X <= k-1) = sf(k-1) for Poisson
    p_value = float(stats.poisson.sf(max(k_obs - 1, 0), expected))
    corrected_alpha = alpha_level / max(n_tests, 1)
    is_alert = p_value < corrected_alpha

    return dict(
        p_value=p_value,
        corrected_alpha=corrected_alpha,
        is_alert=is_alert,
        expected_count=expected,
    )


# ---------------------------------------------------------------------------
# EWMA rate estimator
# ---------------------------------------------------------------------------

def ewma_rate_detector(
    event_times: np.ndarray,
    t_now: float,
    half_life_h: float = 1.0,
) -> float:
    """
    Estimate the current event rate using Exponentially Weighted Moving Average (EWMA).

    The EWMA rate at t_now is computed as:
        λ_ewma(t_now) = Σ_{tᵢ ≤ t_now} w_i  /  Σ_{tᵢ ≤ t_now} (1/decay_rate)

    Implementation: kernel density estimate with exponential kernel
        λ̂(t) = Σ_{tᵢ < t} λ_decay · exp(−λ_decay · (t − tᵢ))
    where λ_decay = log(2) / half_life_h.

    Parameters
    ----------
    event_times : np.ndarray — sorted event times (hours)
    t_now       : float      — evaluation time (hours)
    half_life_h : float      — EWMA half-life (hours)

    Returns
    -------
    float — EWMA rate estimate at t_now (events per hour)
    """
    times = np.asarray(event_times, dtype=float)
    past_events = times[times < t_now]

    if len(past_events) == 0:
        return 0.0

    decay_rate = math.log(2.0) / max(half_life_h, 1e-9)
    deltas = t_now - past_events  # all positive
    weights = decay_rate * np.exp(-decay_rate * deltas)
    return float(np.sum(weights))


# ---------------------------------------------------------------------------
# Sliding window detector
# ---------------------------------------------------------------------------

def sliding_window_detector(
    event_times: np.ndarray,
    t_now: float,
    window_h: float,
    threshold_count: int,
) -> dict:
    """
    Count events in the sliding window [t_now − window_h, t_now] and
    flag an alert if the count exceeds *threshold_count*.

    Parameters
    ----------
    event_times     : np.ndarray — sorted event times (hours)
    t_now           : float      — right edge of the window (hours)
    window_h        : float      — window width (hours)
    threshold_count : int        — alert threshold

    Returns
    -------
    dict with keys:
        count        : int
        is_alert     : bool
        window_start : float
        window_end   : float
    """
    times = np.asarray(event_times, dtype=float)
    window_start = t_now - window_h
    count = int(np.sum((times >= window_start) & (times <= t_now)))
    is_alert = count >= threshold_count

    return dict(
        count=count,
        is_alert=is_alert,
        window_start=window_start,
        window_end=t_now,
    )


# ---------------------------------------------------------------------------
# Empirical Fano factor
# ---------------------------------------------------------------------------

def compute_fano_factor(
    event_times: np.ndarray,
    T: float,
    window_h: Optional[float] = None,
    window_size_h: Optional[float] = None,
) -> float:
    """
    Compute the empirical Fano factor (variance / mean) for event counts in
    non-overlapping windows of size *window_h*.

    For a homogeneous Poisson process, Fano ≈ 1.
    Hawkes processes show Fano > 1 (super-Poisson).

    Parameters
    ----------
    event_times : np.ndarray — sorted event times in [0, T]
    T           : float      — observation horizon
    window_h    : float      — bin width (hours)
    window_size_h : float    — alias for window_h

    Returns
    -------
    float — Fano factor (Var[N_Δ] / E[N_Δ]); NaN if mean ≈ 0
    """
    if window_h is None:
        window_h = window_size_h if window_size_h is not None else 1.0

    times = np.asarray(event_times, dtype=float)
    n_bins = max(1, int(T / window_h))
    edges = np.linspace(0.0, T, n_bins + 1)
    counts, _ = np.histogram(times, bins=edges)

    mean_count = float(np.mean(counts))
    if mean_count < 1e-10:
        return float("nan")

    var_count = float(np.var(counts, ddof=1)) if len(counts) > 1 else 0.0
    return var_count / mean_count


# ---------------------------------------------------------------------------
# Coefficient of Variation of inter-arrival times
# ---------------------------------------------------------------------------

def compute_cv(event_times: np.ndarray) -> float:
    """
    Compute the coefficient of variation (std / mean) of inter-arrival times.

    For a homogeneous Poisson process, CV = 1.
    Hawkes processes tend to exhibit CV > 1 (bursty arrivals).

    Parameters
    ----------
    event_times : np.ndarray — sorted event times (hours)

    Returns
    -------
    float — CV of inter-arrivals; NaN if fewer than 2 events
    """
    times = np.asarray(event_times, dtype=float)
    if len(times) < 2:
        return float("nan")

    gaps = np.diff(np.sort(times))
    mean_gap = float(np.mean(gaps))
    if mean_gap < 1e-12:
        return float("nan")

    std_gap = float(np.std(gaps, ddof=1))
    return std_gap / mean_gap


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Quick smoke test
    rng = np.random.default_rng(42)
    T = 100.0
    mu_true = 2.0
    gaps = rng.exponential(1.0 / mu_true, size=200)
    times = np.cumsum(gaps)
    times = times[times < T]

    result = fit_poisson(times, T)
    print("Rung 0 fit:", result)

    diurnal = fit_poisson_diurnal(times, T, n_bins=24, reg_lambda=0.01)
    print("Rung 1 fit: AIC=", diurnal["aic"])

    alert = poisson_tail_alert(k_obs=15, mu_hat=2.0, window_h=1.0, alpha_level=0.05, n_tests=10)
    print("Alert:", alert)

    ewma = ewma_rate_detector(times, t_now=50.0, half_life_h=1.0)
    print(f"EWMA rate at t=50h: {ewma:.3f}")

    sw = sliding_window_detector(times, t_now=50.0, window_h=1.0, threshold_count=5)
    print("Sliding window:", sw)

    fano = compute_fano_factor(times, T, window_h=1.0)
    print(f"Fano factor: {fano:.3f}")

    cv = compute_cv(times)
    print(f"CV of inter-arrivals: {cv:.3f}")
