"""
src/analysis.py — Statistical analysis and goodness-of-fit diagnostics.

Inter-arrival analysis
  CV = std(gaps)/mean(gaps); CV ≈ 1 for Poisson, CV > 1 for clustering.

Fano factor
  F(Δ) = Var[N(Δ)] / E[N(Δ)].  Poisson: F=1 for all Δ.
  Hawkes asymptote: F → 1/(1−n)² as Δ → ∞.
  Chi-square dispersion test: T = (N−1)·F(Δ) ~ χ²(N−1) under Poisson.

Parametric bootstrap LRT (Self & Liang 1987; boundary issue)
  Under H0: alpha=0, beta is unidentified => standard χ²(1)/2 mixture.
  We use parametric bootstrap: simulate N_boot datasets from fitted H0 Poisson,
  refit both H0 and H1 on each, build empirical null distribution of 2Δloglik.

Time-rescaling theorem (Papangelou 1972; Brown et al. 2002)
  Under correct model: τᵢ = Λ(tᵢ) − Λ(tᵢ₋₁) ~ i.i.d. Exp(1).
  KS test on τ against Exp(1) and Q-Q plot.
"""
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inter-arrival statistics
# ---------------------------------------------------------------------------

def compute_interarrival_stats(event_times: np.ndarray) -> dict:
    """Return mean, std, CV, and gaps array for inter-arrival times.

    CV ≈ 1 indicates Poisson-like process; CV > 1 suggests clustering.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted 1-D array of event timestamps.

    Returns
    -------
    dict with keys: mean, std, cv, gaps, n_events
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    if len(event_times) < 2:
        logger.warning("compute_interarrival_stats: fewer than 2 events; returning NaNs.")
        return {"mean": np.nan, "std": np.nan, "cv": np.nan, "gaps": np.array([]), "n_events": len(event_times)}

    gaps = np.diff(event_times)
    mu = float(np.mean(gaps))
    sigma = float(np.std(gaps, ddof=1))
    cv = sigma / mu if mu > 0 else np.nan
    return {
        "mean": mu,
        "std": sigma,
        "cv": cv,
        "gaps": gaps,
        "n_events": len(event_times),
    }


# ---------------------------------------------------------------------------
# Fano factor
# ---------------------------------------------------------------------------

def compute_fano_factor(
    event_times: np.ndarray,
    T: float,
    window_h: Optional[float] = None,
    window_size_h: Optional[float] = None,
) -> float:
    """Compute Fano factor F(Δ) = Var[N(Δ)] / E[N(Δ)] for a given window size.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted event timestamps in [0, T].
    T : float
        Total observation duration.
    window_h : float, optional
        Window size Δ (same units as event_times).
    window_size_h : float, optional
        Alias for window_h.

    Returns
    -------
    float
        Fano factor; returns NaN if fewer than 2 non-empty windows.
    """
    if window_h is None:
        window_h = window_size_h if window_size_h is not None else 1.0

    event_times = np.sort(np.asarray(event_times, dtype=float))
    bins = np.arange(0, T + window_h, window_h)
    counts, _ = np.histogram(event_times, bins=bins)
    if len(counts) < 2:
        return np.nan
    mean_count = float(np.mean(counts))
    if mean_count == 0:
        return np.nan
    return float(np.var(counts, ddof=1) / mean_count)


def fano_vs_windows(
    event_times: np.ndarray,
    T: float,
    window_sizes: list,
    n: float = None,
) -> dict:
    """Compute Fano factor across multiple window sizes.

    Also runs chi-square dispersion tests for each window.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted event timestamps.
    T : float
        Observation duration.
    window_sizes : list of float
        Window sizes Δ to evaluate.
    n : float, optional
        Branching ratio (α/β) of fitted Hawkes model. If provided, the
        theoretical Hawkes asymptotic Fano factor 1/(1−n)² is added.

    Returns
    -------
    dict with keys: windows, fano_empirical, fano_hawkes_theoretical,
                    dispersion_tests
    """
    windows = np.asarray(window_sizes, dtype=float)
    fano_emp = np.full(len(windows), np.nan)
    disp_tests = []

    for i, wh in enumerate(windows):
        fano_emp[i] = compute_fano_factor(event_times, T, wh)
        disp_tests.append(chi_square_dispersion_test(event_times, T, wh))

    fano_hawkes = None
    if n is not None and 0.0 <= n < 1.0:
        fano_hawkes = np.full(len(windows), 1.0 / (1.0 - n) ** 2)

    return {
        "windows": windows,
        "fano_empirical": fano_emp,
        "fano_hawkes_theoretical": fano_hawkes,
        "dispersion_tests": disp_tests,
    }


# ---------------------------------------------------------------------------
# Chi-square dispersion test
# ---------------------------------------------------------------------------

def chi_square_dispersion_test(
    event_times: np.ndarray,
    T: float,
    window_h: float,
    alpha_level: float = 0.05,
) -> dict:
    """Chi-square dispersion test for Poisson vs. over-dispersed counts.

    Test statistic T = (N−1) · F(Δ) ~ χ²(N−1) under H0: Poisson.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted event timestamps.
    T : float
        Observation duration.
    window_h : float
        Window size for binning.
    alpha_level : float
        Significance level for rejection.

    Returns
    -------
    dict with keys: fano, test_stat, p_value, reject_poisson, n_windows
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    bins = np.arange(0, T + window_h, window_h)
    counts, _ = np.histogram(event_times, bins=bins)
    N = len(counts)

    fano = compute_fano_factor(event_times, T, window_h)
    if np.isnan(fano) or N < 2:
        return {
            "fano": fano,
            "test_stat": np.nan,
            "p_value": np.nan,
            "reject_poisson": False,
            "n_windows": N,
        }

    test_stat = (N - 1) * fano
    # Upper-tail p-value (over-dispersion)
    p_value = float(1.0 - stats.chi2.cdf(test_stat, df=N - 1))
    return {
        "fano": float(fano),
        "test_stat": float(test_stat),
        "p_value": float(p_value),
        "reject_poisson": bool(p_value < alpha_level),
        "n_windows": int(N),
    }


# ---------------------------------------------------------------------------
# Parametric bootstrap LRT
# ---------------------------------------------------------------------------

def _poisson_loglik(mu: float, n_events: int, T: float) -> float:
    """Log-likelihood of homogeneous Poisson process."""
    if mu <= 0:
        return -np.inf
    return n_events * math.log(mu) - mu * T


def _hawkes_loglik(
    event_times: np.ndarray,
    T: float,
    mu: float,
    alpha: float,
    beta: float,
) -> float:
    """Log-likelihood of exponential Hawkes process (O(n) recursion)."""
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return -np.inf

    n = len(event_times)
    if n == 0:
        return -mu * T

    # Compensator integral: μT + α/β * Σ(1 − exp(−β(T−tᵢ)))
    compensator = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - event_times)))

    # Log-intensity sum with O(n) recursion
    log_sum = 0.0
    R = 0.0  # recursive sum of past excitations
    for i, ti in enumerate(event_times):
        if i == 0:
            lam_i = mu
        else:
            R = math.exp(-beta * (ti - event_times[i - 1])) * (1.0 + R)
            lam_i = mu + alpha * R
        if lam_i <= 0:
            return -np.inf
        log_sum += math.log(lam_i)

    return log_sum - compensator


def _fit_poisson_mle(event_times: np.ndarray, T: float) -> float:
    """MLE for homogeneous Poisson: μ̂ = N/T."""
    return len(event_times) / T if T > 0 else 0.0


def _fit_hawkes_mle(
    event_times: np.ndarray,
    T: float,
    mu0: float = None,
    alpha0: float = 0.3,
    beta0: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Simple gradient-free MLE for Hawkes via scipy.optimize.minimize.

    Returns (mu, alpha, beta, loglik).
    """
    from scipy.optimize import minimize

    if mu0 is None:
        mu0 = max(_fit_poisson_mle(event_times, T) * 0.5, 1e-4)

    def neg_loglik(params):
        mu, alpha, beta = params
        return -_hawkes_loglik(event_times, T, mu, alpha, beta)

    x0 = np.array([mu0, alpha0, beta0])
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]
    # Also enforce alpha < beta (stationarity)
    constraints = [{"type": "ineq", "fun": lambda p: p[2] - p[1] - 1e-6}]

    result = minimize(
        neg_loglik,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )
    if result.success or result.fun < neg_loglik(x0):
        mu_h, alpha_h, beta_h = result.x
        ll = float(-result.fun)
    else:
        mu_h, alpha_h, beta_h = x0
        ll = float(-neg_loglik(x0))
    return float(mu_h), float(alpha_h), float(beta_h), ll


def _simulate_poisson(mu: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate homogeneous Poisson process on [0, T]."""
    n = rng.poisson(mu * T)
    return np.sort(rng.uniform(0, T, n))


def parametric_bootstrap_lrt(
    event_times: np.ndarray,
    T: float,
    n_boot: int = 499,
    rng_seed: int = 7,
    alpha_level: float = 0.05,
) -> dict:
    """Parametric bootstrap likelihood ratio test: Poisson vs. Hawkes.

    Under H0: alpha=0, β is unidentified, so the standard χ²(1)/2 mixture is
    unreliable. We build the null distribution empirically.

    Parameters
    ----------
    event_times : np.ndarray
        Observed event times.
    T : float
        Observation duration.
    n_boot : int
        Number of bootstrap replications.
    rng_seed : int
        Random seed.
    alpha_level : float
        Significance level.

    Returns
    -------
    dict with keys: observed_lrt, p_value, null_lrt_samples, reject_poisson,
                    mu_h0, mu_h1, alpha_h1, beta_h1
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    rng = np.random.default_rng(rng_seed)

    # Fit H0: Poisson
    mu_h0 = _fit_poisson_mle(event_times, T)
    ll_h0 = _poisson_loglik(mu_h0, len(event_times), T)

    # Fit H1: Hawkes
    mu_h1, alpha_h1, beta_h1, ll_h1 = _fit_hawkes_mle(event_times, T, mu0=mu_h0)

    observed_lrt = float(2.0 * (ll_h1 - ll_h0))
    observed_lrt = max(observed_lrt, 0.0)

    # Bootstrap null distribution
    null_lrt_samples = np.zeros(n_boot)
    for b in range(n_boot):
        sim = _simulate_poisson(mu_h0, T, rng)
        mu_b = _fit_poisson_mle(sim, T)
        ll_b_h0 = _poisson_loglik(mu_b, len(sim), T)
        _, _, _, ll_b_h1 = _fit_hawkes_mle(sim, T, mu0=mu_b)
        null_lrt_samples[b] = max(2.0 * (ll_b_h1 - ll_b_h0), 0.0)

    p_value = float(np.mean(null_lrt_samples >= observed_lrt))

    return {
        "observed_lrt": observed_lrt,
        "p_value": p_value,
        "null_lrt_samples": null_lrt_samples,
        "reject_poisson": bool(p_value < alpha_level),
        "mu_h0": float(mu_h0),
        "mu_h1": float(mu_h1),
        "alpha_h1": float(alpha_h1),
        "beta_h1": float(beta_h1),
    }


# ---------------------------------------------------------------------------
# Time-rescaling test
# ---------------------------------------------------------------------------

def _hawkes_compensator(
    event_times: np.ndarray,
    T: float,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Compute compensator Λ(tᵢ) = ∫₀^tᵢ λ(s) ds for each event time."""
    n = len(event_times)
    compensators = np.zeros(n)
    R = 0.0  # recursive term
    for i in range(n):
        ti = event_times[i]
        if i == 0:
            # Λ(t₀) = μt₀  (no past events)
            excitation = 0.0
        else:
            dt = ti - event_times[i - 1]
            R = math.exp(-beta * dt) * (1.0 + R)
            # Λ(tᵢ) − Λ(tᵢ₋₁) integral of excitation part
            # = α/β * [R_prev*(1 − exp(−β*dt)) ] already summed below
            excitation = (alpha / beta) * R * (1.0 - math.exp(-beta * dt)) if False else 0.0
        compensators[i] = mu * ti + (alpha / beta) * sum(
            1.0 - math.exp(-beta * (ti - event_times[j]))
            for j in range(i)
        )
    return compensators


def _hawkes_compensator_fast(
    event_times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """O(n) compensator for Hawkes process at each event time."""
    n = len(event_times)
    Lambda = np.zeros(n)
    A = 0.0  # running: Σ_{j<i} exp(−β(tᵢ − tⱼ))
    for i in range(n):
        ti = event_times[i]
        if i > 0:
            dt = ti - event_times[i - 1]
            A = math.exp(-beta * dt) * (A + 1.0)
        # Λ(tᵢ) = μtᵢ + (α/β) Σ_{j<i} (1 − exp(−β(tᵢ−tⱼ)))
        #        = μtᵢ + (α/β)(i − Σ_{j<i} exp(−β(tᵢ−tⱼ)))
        #   Σ_{j<i} exp(−β(tᵢ−tⱼ)) tracked via A
        Lambda[i] = mu * ti + (alpha / beta) * (i - A)
    return Lambda


def time_rescaling_test(
    event_times: np.ndarray,
    *args,
    T: Optional[float] = None,
    mu: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    alpha_level: float = 0.05,
    **kwargs,
) -> dict:
    """Time-rescaling goodness-of-fit test (Brown et al. 2002).

    Rescaled gaps τᵢ = Λ(tᵢ) − Λ(tᵢ₋₁) ~ i.i.d. Exp(1) under correct model.
    KS test against Exp(1).

    Supports both (event_times, T, mu, alpha, beta) and
    (event_times, mu, alpha, beta, T) calling conventions.
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    if len(event_times) < 2:
        return {
            "rescaled_gaps": np.array([]),
            "rescaled_interarrivals": np.array([]),
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "p_value": np.nan,
            "reject_model": False,
        }

    # Parse flexible positional arguments if kwargs not fully supplied
    if len(args) == 4:
        a0, a1, a2, a3 = [float(x) for x in args]
        # Check if caller passed (mu, alpha, beta, T) vs (T, mu, alpha, beta)
        if a3 > a0 and a3 > a2:  # last arg is large time horizon T
            mu = mu if mu is not None else a0
            alpha = alpha if alpha is not None else a1
            beta = beta if beta is not None else a2
            T = T if T is not None else a3
        else:  # standard (T, mu, alpha, beta)
            T = T if T is not None else a0
            mu = mu if mu is not None else a1
            alpha = alpha if alpha is not None else a2
            beta = beta if beta is not None else a3
    elif len(args) == 1 and T is None:
        T = float(args[0])

    if mu is None:
        mu = 1.0
    if alpha is None:
        alpha = 0.2
    if beta is None:
        beta = 1.0

    Lambda = _hawkes_compensator_fast(event_times, mu, alpha, beta)
    rescaled_gaps = np.diff(Lambda)
    rescaled_gaps = rescaled_gaps[rescaled_gaps > 0]

    if len(rescaled_gaps) == 0:
        return {
            "rescaled_gaps": np.array([]),
            "rescaled_interarrivals": np.array([]),
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "p_value": np.nan,
            "reject_model": False,
        }

    ks_stat, ks_pvalue = stats.kstest(rescaled_gaps, "expon", args=(0, 1))
    return {
        "rescaled_gaps": rescaled_gaps,
        "rescaled_interarrivals": rescaled_gaps,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "p_value": float(ks_pvalue),
        "reject_model": bool(ks_pvalue < alpha_level),
    }


# ---------------------------------------------------------------------------
# Q-Q exponential
# ---------------------------------------------------------------------------

def qq_exponential(gaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute theoretical vs. empirical quantiles for Exp(1) Q-Q plot.

    Parameters
    ----------
    gaps : np.ndarray
        Observed (rescaled) inter-arrival gaps.

    Returns
    -------
    (theoretical_quantiles, empirical_quantiles)
    """
    gaps = np.sort(np.asarray(gaps, dtype=float))
    n = len(gaps)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = -np.log(1.0 - probs)  # Exp(1) quantile function
    return theoretical, gaps


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_interarrival_histogram(
    event_times: np.ndarray,
    output_path: str,
    mu: float = None,
    title: str = "",
) -> str:
    """Plot inter-arrival time histogram with optional Exponential(μ) overlay.

    Parameters
    ----------
    event_times : np.ndarray
        Sorted event timestamps.
    output_path : str
        File path to save the figure.
    mu : float, optional
        Poisson rate; if given, overlays the fitted Exp(1/μ) density.
    title : str
        Figure title.

    Returns
    -------
    str : output_path
    """
    gaps = np.diff(np.sort(np.asarray(event_times, dtype=float)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gaps, bins=30, density=True, alpha=0.6, color="steelblue", label="Empirical")

    if mu is not None and mu > 0:
        x = np.linspace(0, gaps.max() * 1.2, 300)
        ax.plot(x, mu * np.exp(-mu * x), "r-", lw=2, label=f"Exp(λ={mu:.3f})")

    # Fit Exp to data and overlay
    loc_fit, scale_fit = stats.expon.fit(gaps, floc=0)
    x = np.linspace(0, gaps.max() * 1.2, 300)
    ax.plot(x, stats.expon.pdf(x, loc=loc_fit, scale=scale_fit),
            "g--", lw=2, label=f"Exp fit (λ̂={1/scale_fit:.3f})")

    cv = float(np.std(gaps, ddof=1) / np.mean(gaps)) if np.mean(gaps) > 0 else np.nan
    ax.set_xlabel("Inter-arrival time")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nCV={cv:.3f}" if title else f"Inter-arrival times  CV={cv:.3f}")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved inter-arrival histogram to %s", output_path)
    return output_path


def plot_fano_vs_windows(
    result_dict: dict,
    output_path: str,
    title: str = "",
) -> str:
    """Plot Fano factor vs. window size.

    Parameters
    ----------
    result_dict : dict
        Output of fano_vs_windows().
    output_path : str
        File path to save.
    title : str
        Figure title.

    Returns
    -------
    str : output_path
    """
    windows = result_dict["windows"]
    fano_emp = result_dict["fano_empirical"]
    fano_hawk = result_dict.get("fano_hawkes_theoretical")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(windows, fano_emp, "o-", color="steelblue", label="Empirical Fano")
    ax.axhline(1.0, color="gray", linestyle="--", label="Poisson (F=1)")

    if fano_hawk is not None:
        ax.axhline(fano_hawk[0], color="crimson", linestyle=":", lw=2,
                   label=f"Hawkes asymptote ({fano_hawk[0]:.2f})")

    ax.set_xlabel("Window size Δ")
    ax.set_ylabel("Fano factor F(Δ)")
    ax.set_title(title or "Fano Factor vs. Window Size")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved Fano plot to %s", output_path)
    return output_path


def plot_count_pmf_comparison(
    event_times: np.ndarray,
    T: float,
    mu_poisson: float,
    mu_hawkes: float = None,
    alpha: float = None,
    beta: float = None,
    window_h: float = 1.0,
    output_path: str = None,
) -> str:
    """3-way PMF comparison: empirical / Poisson(μ·Δ) / Hawkes marginal.

    The Hawkes marginal count distribution is approximated via simulation
    (1 000 paths) if (mu_hawkes, alpha, beta) are provided.

    Parameters
    ----------
    event_times : np.ndarray
        Observed event times.
    T : float
        Observation duration.
    mu_poisson : float
        Fitted Poisson rate.
    mu_hawkes, alpha, beta : float, optional
        Hawkes parameters.
    window_h : float
        Window size for binning.
    output_path : str, optional
        If None, saves to 'reports/count_pmf.png'.

    Returns
    -------
    str : output_path
    """
    if output_path is None:
        output_path = os.path.join("reports", "count_pmf.png")

    event_times = np.sort(np.asarray(event_times, dtype=float))
    bins = np.arange(0, T + window_h, window_h)
    empirical_counts, _ = np.histogram(event_times, bins=bins)
    max_k = max(empirical_counts.max() + 5, 20)
    k_vals = np.arange(0, max_k + 1)

    # Empirical PMF
    emp_pmf = np.bincount(empirical_counts, minlength=max_k + 1) / len(empirical_counts)
    emp_pmf = emp_pmf[: max_k + 1]

    # Poisson PMF
    lam_p = mu_poisson * window_h
    poisson_pmf = stats.poisson.pmf(k_vals, lam_p)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.3
    ax.bar(k_vals - width, emp_pmf, width=width, label="Empirical", alpha=0.7)
    ax.bar(k_vals, poisson_pmf, width=width, label=f"Poisson(λ={lam_p:.2f})", alpha=0.7)

    # Hawkes simulated PMF
    if mu_hawkes is not None and alpha is not None and beta is not None:
        rng = np.random.default_rng(0)
        sim_counts = []
        for _ in range(1000):
            sim = _simulate_hawkes(mu_hawkes, alpha, beta, T, rng)
            c, _ = np.histogram(sim, bins=bins)
            sim_counts.extend(c.tolist())
        sim_arr = np.array(sim_counts)
        hawk_pmf = np.bincount(np.clip(sim_arr, 0, max_k), minlength=max_k + 1) / len(sim_arr)
        hawk_pmf = hawk_pmf[: max_k + 1]
        ax.bar(k_vals + width, hawk_pmf, width=width, label="Hawkes (simulated)", alpha=0.7)

    ax.set_xlabel(f"Count in window Δ={window_h}")
    ax.set_ylabel("Probability")
    ax.set_title("Count PMF Comparison")
    ax.legend()
    ax.set_xlim(-1, min(max_k, 30))
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved PMF comparison to %s", output_path)
    return output_path


def plot_time_rescaling_qq(
    rescaled_gaps: np.ndarray,
    output_path: str,
    title: str = "",
) -> str:
    """Q-Q plot of rescaled gaps vs. Exp(1).

    Parameters
    ----------
    rescaled_gaps : np.ndarray
        Output of time_rescaling_test()["rescaled_gaps"].
    output_path : str
        File path to save.
    title : str
        Figure title.

    Returns
    -------
    str : output_path
    """
    theoretical, empirical = qq_exponential(rescaled_gaps)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical, empirical, s=10, alpha=0.6, color="steelblue", label="Rescaled gaps")
    lim = max(theoretical.max(), empirical.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x (Exp(1))")
    ax.set_xlabel("Theoretical Exp(1) quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title(title or "Time-Rescaling Q-Q Plot")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved time-rescaling Q-Q plot to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal Hawkes simulator (used by PMF comparison)
# ---------------------------------------------------------------------------

def _simulate_hawkes(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Ogata thinning algorithm for exponential Hawkes process."""
    events = []
    t = 0.0
    lam_bar = mu  # initial upper bound

    while t < T:
        # Draw candidate
        u = rng.uniform()
        w = -math.log(u) / lam_bar
        t += w
        if t > T:
            break

        # Compute true intensity at t
        lam_t = mu + alpha * sum(math.exp(-beta * (t - s)) for s in events)

        # Accept/reject
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)
            lam_bar = lam_t + alpha  # new upper bound after event
        else:
            lam_bar = lam_t

    return np.array(events)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_clustering(events: pd.DataFrame, cfg: dict = None) -> dict:
    """Run full clustering analysis and goodness-of-fit diagnostics.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain a 'timestamp' column (numeric seconds or datetime).
    cfg : dict, optional
        Configuration dict (yaml-loaded). Expected keys:
          analysis.window_sizes, analysis.n_bootstrap,
          poisson_baseline.mu, model.mu, model.alpha, model.beta,
          paths.reports_dir

    Returns
    -------
    dict with keys: is_clustered, stats, figure_paths, poisson_fit, hawkes_fit
    """
    cfg = cfg or {}
    reports_dir = cfg.get("paths", {}).get("reports_dir", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # --- Extract event times ---
    if "timestamp" not in events.columns:
        raise ValueError("events DataFrame must have a 'timestamp' column.")
    ts = events["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        event_times = ts.astype(np.int64).values / 1e9  # ns → s
    else:
        event_times = ts.astype(float).values
    event_times = np.sort(event_times)
    T = float(event_times[-1] - event_times[0]) if len(event_times) > 1 else 1.0
    # Normalise to start at 0
    event_times = event_times - event_times[0]

    # --- Poisson and Hawkes parameters (from cfg or fit) ---
    mu_poisson = cfg.get("poisson_baseline", {}).get("mu", None)
    if mu_poisson is None:
        mu_poisson = _fit_poisson_mle(event_times, T)
    poisson_fit = {"mu": float(mu_poisson)}

    mu_h = cfg.get("model", {}).get("mu", None)
    alpha_h = cfg.get("model", {}).get("alpha", None)
    beta_h = cfg.get("model", {}).get("beta", None)
    if any(v is None for v in [mu_h, alpha_h, beta_h]):
        logger.info("Hawkes parameters not in cfg — fitting MLE.")
        mu_h, alpha_h, beta_h, _ = _fit_hawkes_mle(event_times, T, mu0=mu_poisson)
    n_branching = alpha_h / beta_h if beta_h > 0 else 0.0
    hawkes_fit = {"mu": float(mu_h), "alpha": float(alpha_h), "beta": float(beta_h),
                  "branching_ratio": float(n_branching)}

    # --- Inter-arrival stats ---
    ia_stats = compute_interarrival_stats(event_times)

    # --- Fano analysis ---
    window_sizes = cfg.get("analysis", {}).get(
        "window_sizes", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0]
    )
    window_sizes = [w for w in window_sizes if w < T]
    if not window_sizes:
        window_sizes = [T / 10.0]
    fano_result = fano_vs_windows(event_times, T, window_sizes, n=n_branching)

    # --- Bootstrap LRT ---
    n_boot = cfg.get("analysis", {}).get("n_bootstrap", 299)
    lrt_result = parametric_bootstrap_lrt(event_times, T, n_boot=n_boot)

    # --- Time-rescaling test ---
    tr_result = time_rescaling_test(event_times, T, mu_h, alpha_h, beta_h)

    # --- Figures ---
    fig_paths = {}

    # Clustering analysis figure (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Inter-arrival histogram
    if len(ia_stats["gaps"]) > 0:
        ax = axes[0, 0]
        ax.hist(ia_stats["gaps"], bins=30, density=True, alpha=0.6, color="steelblue")
        x = np.linspace(0, ia_stats["gaps"].max() * 1.2, 300)
        ax.plot(x, mu_poisson * np.exp(-mu_poisson * x), "r-", lw=2,
                label=f"Exp(λ={mu_poisson:.3f})")
        ax.set_title(f"Inter-arrival times  CV={ia_stats['cv']:.3f}")
        ax.set_xlabel("Gap (s)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # 2. Fano vs windows
    ax = axes[0, 1]
    ax.plot(fano_result["windows"], fano_result["fano_empirical"], "o-", color="steelblue")
    ax.axhline(1.0, color="gray", linestyle="--", label="Poisson")
    if fano_result["fano_hawkes_theoretical"] is not None:
        ax.axhline(fano_result["fano_hawkes_theoretical"][0], color="crimson", linestyle=":",
                   label=f"Hawkes ({fano_result['fano_hawkes_theoretical'][0]:.2f})")
    ax.set_title("Fano Factor vs. Window Size")
    ax.set_xlabel("Window Δ")
    ax.set_ylabel("F(Δ)")
    ax.legend(fontsize=8)

    # 3. Bootstrap null LRT
    ax = axes[1, 0]
    ax.hist(lrt_result["null_lrt_samples"], bins=30, density=True, alpha=0.6, color="gray",
            label="Bootstrap null")
    ax.axvline(lrt_result["observed_lrt"], color="crimson", lw=2,
               label=f"Observed LRT={lrt_result['observed_lrt']:.2f}")
    ax.set_title(f"Bootstrap LRT  p={lrt_result['p_value']:.4f}")
    ax.set_xlabel("2Δ log-likelihood")
    ax.legend(fontsize=8)

    # 4. Time-rescaling Q-Q
    ax = axes[1, 1]
    if len(tr_result["rescaled_gaps"]) > 0:
        th, em = qq_exponential(tr_result["rescaled_gaps"])
        ax.scatter(th, em, s=8, alpha=0.5, color="steelblue")
        lim = max(th.max(), em.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1.5)
    ax.set_title(f"Time-rescaling Q-Q  KS p={tr_result['ks_pvalue']:.4f}")
    ax.set_xlabel("Theoretical Exp(1)")
    ax.set_ylabel("Empirical")

    fig.suptitle("Clustering Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    clustering_fig = os.path.join(reports_dir, "clustering_analysis.png")
    fig.savefig(clustering_fig, dpi=130)
    plt.close(fig)
    fig_paths["clustering_analysis"] = clustering_fig

    # GoF summary figure
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    metrics = {
        "CV": ia_stats["cv"],
        f"Fano(Δ={window_sizes[len(window_sizes)//2]:.1f})": fano_result["fano_empirical"][len(window_sizes)//2],
        "Bootstrap LRT p": lrt_result["p_value"],
        "KS p (rescaled)": tr_result["ks_pvalue"],
    }
    colors = []
    for k, v in metrics.items():
        if k.endswith(" p") or "LRT" in k or "KS" in k:
            colors.append("crimson" if v < 0.05 else "steelblue")
        else:
            colors.append("steelblue")
    ax2.barh(list(metrics.keys()), list(metrics.values()), color=colors)
    ax2.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("GoF Diagnostics Summary")
    ax2.set_xlabel("Value")
    fig2.tight_layout()
    gof_fig = os.path.join(reports_dir, "gof.png")
    fig2.savefig(gof_fig, dpi=120)
    plt.close(fig2)
    fig_paths["gof"] = gof_fig

    # Determine overall clustering verdict
    reject_poisson = (
        ia_stats["cv"] > 1.15
        or lrt_result["reject_poisson"]
        or any(d["reject_poisson"] for d in fano_result["dispersion_tests"])
    )

    return {
        "is_clustered": bool(reject_poisson),
        "stats": {
            "interarrival": ia_stats,
            "fano": fano_result,
            "lrt": lrt_result,
            "time_rescaling": tr_result,
        },
        "figure_paths": fig_paths,
        "poisson_fit": poisson_fit,
        "hawkes_fit": hawkes_fit,
    }
