"""
src/model.py — Hawkes process MLE engine.

Log-likelihood (O(n) recursion, Ozaki 1979):
  A_i = exp(−β(t_i − t_{i−1}))(1 + A_{i−1}),  A_1 = 0
  logL = Σ log(μ + α A_i) − μ T − (α/β) Σ (1 − exp(−β(T − t_i)))

Parameterisation for bounded optimisation (Amendment A3):
  θ = (log μ, logit((n − n_min)/(n_max − n_min)), log β)
  where n = α/β,  α = n β,  so α < β iff n < 1.

Multi-start: n_starts ≥ 5 initialisations including:
  • near-Poisson: n≈0.01, β≈1.0, μ≈N/T
  • high-clustering: n≈0.9, β≈5.0, μ≈0.1
  • random draws from interior of parameter space

Returns dict with keys:
  mu, alpha, beta, branching_ratio, log_likelihood, aic, bic,
  converged, reason, n_events, n_starts_converged
"""
import logging
import math
from typing import Optional, Tuple

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default optimizer configuration
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "n_starts": 5,
    "n_min": 1e-4,       # minimum branching ratio
    "n_max": 0.9999,     # maximum branching ratio (< 1 for stationarity)
    "mu_min": 1e-6,
    "mu_max": 1e4,
    "beta_min": 1e-4,
    "beta_max": 1e4,
    "max_iter": 2000,
    "ftol": 1e-14,
    "gtol": 1e-8,
}


# ---------------------------------------------------------------------------
# Logit / sigmoid helpers
# ---------------------------------------------------------------------------

def _logit(x: float) -> float:
    """
    Logit function: log(x / (1 − x)).
    Numerically stable for x ∈ (0, 1).
    """
    x = float(np.clip(x, 1e-15, 1.0 - 1e-15))
    return math.log(x) - math.log(1.0 - x)


def _sigmoid(x: float) -> float:
    """
    Sigmoid function: 1 / (1 + exp(−x)).
    Numerically stable for large |x|.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


# ---------------------------------------------------------------------------
# O(n) Hawkes log-likelihood (Ozaki 1979 recursion)
# ---------------------------------------------------------------------------

def _hawkes_loglik(
    mu: float,
    alpha: float,
    beta: float,
    times: np.ndarray,
    T: float,
) -> float:
    """
    Compute the negative Hawkes log-likelihood using the O(n) Ozaki recursion.

    Recursion:
        A_0 = 0
        A_i = exp(−β (t_i − t_{i−1})) * (1 + A_{i−1})   for i ≥ 1

    Log-likelihood:
        logL = Σ_i log(μ + α A_i)
               − μ T
               − (α/β) Σ_i (1 − exp(−β (T − t_i)))

    Parameters
    ----------
    mu    : float > 0
    alpha : float > 0
    beta  : float > 0
    times : np.ndarray — sorted event times in [0, T]
    T     : float — observation horizon

    Returns
    -------
    float — negative log-likelihood (for minimisation)
    """
    n = len(times)
    if n == 0:
        return mu * T  # only the compensator term remains; return neg ll = mu * T

    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e18  # infeasible

    log_intensity_sum = 0.0
    A_prev = 0.0
    t_prev = 0.0

    for i in range(n):
        t_i = times[i]
        if i == 0:
            A_i = 0.0
        else:
            dt = t_i - t_prev
            A_i = math.exp(-beta * dt) * (1.0 + A_prev)

        intensity = mu + alpha * A_i
        if intensity <= 0.0:
            return 1e18

        log_intensity_sum += math.log(intensity)
        A_prev = A_i
        t_prev = t_i

    # Compensator: −(α/β) Σ_i (1 − exp(−β(T − t_i)))
    compensator = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))

    log_lik = log_intensity_sum - mu * T - compensator
    return -log_lik  # return negative for minimisation


# ---------------------------------------------------------------------------
# Analytic gradient of the Hawkes log-likelihood
# ---------------------------------------------------------------------------

def _hawkes_loglik_grad(
    mu: float,
    alpha: float,
    beta: float,
    times: np.ndarray,
    T: float,
) -> Tuple[float, np.ndarray]:
    """
    Compute the negative Hawkes log-likelihood AND its analytic gradient
    w.r.t. (μ, α, β) using the Ozaki recursion.

    Gradient derivation
    -------------------
    Let λ_i = μ + α A_i.

    ∂logL/∂μ = Σ (1/λ_i) − T
    ∂logL/∂α = Σ (A_i / λ_i) − (1/β) Σ (1 − exp(−β(T−t_i)))
    ∂logL/∂β = Σ (α ∂A_i/∂β / λ_i) + (α/β²) Σ (1 − exp(−β(T−t_i)))
                − (α/β) Σ (T−t_i) exp(−β(T−t_i))

    where ∂A_i/∂β satisfies the recursion:
        B_0 = 0
        B_i = exp(−β Δt_i) * (−Δt_i * (1 + A_{i−1}) + B_{i−1})

    Parameters
    ----------
    mu, alpha, beta : float
    times : np.ndarray — sorted event times
    T     : float

    Returns
    -------
    (neg_log_lik, grad) where grad = [∂(-logL)/∂μ, ∂(-logL)/∂α, ∂(-logL)/∂β]
    """
    n = len(times)
    if n == 0:
        return mu * T, np.array([T, 0.0, 0.0])

    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e18, np.zeros(3)

    log_lik = 0.0
    d_mu = 0.0
    d_alpha = 0.0
    d_beta = 0.0

    A_prev = 0.0
    B_prev = 0.0  # ∂A/∂β
    t_prev = 0.0

    A_arr = np.zeros(n)
    B_arr = np.zeros(n)  # dA/d_beta

    for i in range(n):
        t_i = times[i]
        if i == 0:
            A_i = 0.0
            B_i = 0.0
        else:
            dt = t_i - t_prev
            exp_term = math.exp(-beta * dt)
            A_i = exp_term * (1.0 + A_prev)
            B_i = exp_term * (-dt * (1.0 + A_prev) + B_prev)

        intensity = mu + alpha * A_i
        if intensity <= 0.0:
            return 1e18, np.zeros(3)

        log_lik += math.log(intensity)

        inv_lam = 1.0 / intensity
        d_mu    += inv_lam
        d_alpha += A_i * inv_lam
        d_beta  += alpha * B_i * inv_lam

        A_arr[i] = A_i
        B_arr[i] = B_i
        A_prev = A_i
        B_prev = B_i
        t_prev = t_i

    # Compensator terms
    residuals = T - times  # shape (n,)
    exp_res = np.exp(-beta * residuals)

    compensator = (alpha / beta) * np.sum(1.0 - exp_res)
    log_lik = log_lik - mu * T - compensator

    # Gradient of compensator w.r.t. parameters
    # d(compensator)/d_mu = 0
    # d(compensator)/d_alpha = (1/β) Σ(1 - exp(−β(T−t_i)))
    # d(compensator)/d_beta = −(α/β²) Σ(1 − exp(−β(T−t_i))) + (α/β) Σ (T−t_i) exp(−β(T−t_i))
    dc_d_mu = 0.0
    dc_d_alpha = (1.0 / beta) * float(np.sum(1.0 - exp_res))
    dc_d_beta = (
        -(alpha / beta**2) * float(np.sum(1.0 - exp_res))
        + (alpha / beta) * float(np.sum(residuals * exp_res))
    )

    # Full gradient of logL
    grad_mu    = d_mu    - T - dc_d_mu
    grad_alpha = d_alpha      - dc_d_alpha
    grad_beta  = d_beta       - dc_d_beta

    # Return negative (for minimisation)
    neg_log_lik = -log_lik
    neg_grad = np.array([-grad_mu, -grad_alpha, -grad_beta])

    return neg_log_lik, neg_grad


# ---------------------------------------------------------------------------
# Parameter transformations (unconstrained ↔ constrained)
# ---------------------------------------------------------------------------

def _theta_to_params(theta: np.ndarray, cfg: dict) -> Tuple[float, float, float]:
    """
    Map unconstrained θ = (log μ, logit_n_scaled, log β) to (μ, α, β).

    Transformation:
        μ = exp(θ[0])
        n = n_min + (n_max − n_min) * sigmoid(θ[1])   (branching ratio)
        β = exp(θ[2])
        α = n * β

    Parameters
    ----------
    theta : np.ndarray, shape (3,)
    cfg   : dict with keys n_min, n_max

    Returns
    -------
    (mu, alpha, beta) — all positive, branching ratio in (n_min, n_max)
    """
    n_min = cfg.get("n_min", _DEFAULT_CFG["n_min"])
    n_max = cfg.get("n_max", _DEFAULT_CFG["n_max"])

    mu   = math.exp(float(theta[0]))
    n    = n_min + (n_max - n_min) * _sigmoid(float(theta[1]))
    beta = math.exp(float(theta[2]))
    alpha = n * beta

    return mu, alpha, beta


def _params_to_theta(
    mu: float,
    alpha: float,
    beta: float,
    cfg: dict,
) -> np.ndarray:
    """
    Map (μ, α, β) to unconstrained θ (inverse of _theta_to_params).

    Parameters
    ----------
    mu, alpha, beta : float
    cfg : dict with keys n_min, n_max

    Returns
    -------
    np.ndarray, shape (3,)
    """
    n_min = cfg.get("n_min", _DEFAULT_CFG["n_min"])
    n_max = cfg.get("n_max", _DEFAULT_CFG["n_max"])

    n = alpha / beta  # branching ratio
    n_clipped = float(np.clip(n, n_min + 1e-9, n_max - 1e-9))

    theta0 = math.log(max(mu, 1e-12))
    theta1 = _logit((n_clipped - n_min) / (n_max - n_min))
    theta2 = math.log(max(beta, 1e-12))

    return np.array([theta0, theta1, theta2])


# ---------------------------------------------------------------------------
# Objective function (wraps theta → params → neg loglik + grad)
# ---------------------------------------------------------------------------

def _objective(
    theta: np.ndarray,
    times: np.ndarray,
    T: float,
    cfg: dict,
) -> Tuple[float, np.ndarray]:
    """
    Compute negative log-likelihood and gradient w.r.t. θ (unconstrained).

    Uses chain rule to propagate gradient through the parameter transformation.

    Parameters
    ----------
    theta : np.ndarray, shape (3,) — unconstrained
    times : np.ndarray
    T     : float
    cfg   : dict

    Returns
    -------
    (neg_log_lik, grad_theta)
    """
    n_min = cfg.get("n_min", _DEFAULT_CFG["n_min"])
    n_max = cfg.get("n_max", _DEFAULT_CFG["n_max"])

    mu, alpha, beta = _theta_to_params(theta, cfg)

    neg_ll, neg_grad_raw = _hawkes_loglik_grad(mu, alpha, beta, times, T)

    # neg_grad_raw = [d(-logL)/d_mu, d(-logL)/d_alpha, d(-logL)/d_beta]
    g_mu    = neg_grad_raw[0]
    g_alpha = neg_grad_raw[1]
    g_beta  = neg_grad_raw[2]

    # Jacobians of (mu, alpha, beta) w.r.t. θ
    # μ = exp(θ0)            → ∂μ/∂θ0 = μ
    # n = n_min + Δn σ(θ1)   → ∂n/∂θ1 = Δn σ(1−σ)
    # α = n β = n exp(θ2)    → ∂α/∂θ1 = β ∂n/∂θ1; ∂α/∂θ2 = n β + n ∂β/∂θ2 ... see below
    # β = exp(θ2)            → ∂β/∂θ2 = β
    # α = n β  so:
    #   ∂α/∂θ1 = β * (Δn * σ * (1−σ))
    #   ∂α/∂θ2 = n * β  (since ∂β/∂θ2 = β and n is independent of θ2)

    sig_val = _sigmoid(float(theta[1]))
    delta_n = n_max - n_min
    d_n_d_theta1 = delta_n * sig_val * (1.0 - sig_val)
    n_val = n_min + delta_n * sig_val

    d_mu_d_theta0    = mu
    d_alpha_d_theta1 = beta * d_n_d_theta1
    d_alpha_d_theta2 = n_val * beta   # ∂α/∂θ2 = n β (since ∂β/∂θ2 = β)
    d_beta_d_theta2  = beta

    # Chain rule: grad_theta[i] = Σ_j (∂(-logL)/∂param_j) * (∂param_j / ∂θ_i)
    g_theta0 = g_mu * d_mu_d_theta0
    g_theta1 = g_alpha * d_alpha_d_theta1
    g_theta2 = g_alpha * d_alpha_d_theta2 + g_beta * d_beta_d_theta2

    return neg_ll, np.array([g_theta0, g_theta1, g_theta2])


# ---------------------------------------------------------------------------
# Gradient verification
# ---------------------------------------------------------------------------

def verify_gradient(
    times: np.ndarray,
    T: float,
    eps: float = 1e-5,
    cfg: Optional[dict] = None,
) -> bool:
    """
    Verify analytic gradient against finite differences.

    Checks at a single interior test point.

    Parameters
    ----------
    times : np.ndarray — sorted event times
    T     : float
    eps   : float — finite difference step size
    cfg   : dict  — optimizer config (default: _DEFAULT_CFG)

    Returns
    -------
    bool — True if max relative error < 1e-4
    """
    if cfg is None:
        cfg = _DEFAULT_CFG

    rng = np.random.default_rng(7)
    # Pick a safe interior test point
    mu_test    = 0.3
    alpha_test = 0.4
    beta_test  = 2.0
    theta_test = _params_to_theta(mu_test, alpha_test, beta_test, cfg)

    _, grad_analytic = _objective(theta_test, times, T, cfg)

    grad_fd = np.zeros(3)
    for j in range(3):
        theta_p = theta_test.copy()
        theta_m = theta_test.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        ll_p, _ = _objective(theta_p, times, T, cfg)
        ll_m, _ = _objective(theta_m, times, T, cfg)
        grad_fd[j] = (ll_p - ll_m) / (2.0 * eps)

    rel_err = np.abs(grad_analytic - grad_fd) / (np.abs(grad_fd) + 1e-10)
    max_rel_err = float(np.max(rel_err))
    passed = max_rel_err < 1e-4
    logger.info(
        "Gradient check: max_rel_err=%.2e  %s",
        max_rel_err,
        "PASSED" if passed else "FAILED",
    )
    if not passed:
        logger.warning("Analytic grad: %s", grad_analytic)
        logger.warning("FD grad:       %s", grad_fd)
    return passed


# ---------------------------------------------------------------------------
# Multi-start L-BFGS-B Hawkes MLE
# ---------------------------------------------------------------------------

def fit_hawkes(
    event_times: np.ndarray,
    end_time: Optional[float] = None,
    cfg: Optional[dict] = None,
    T: Optional[float] = None,
) -> dict:
    if end_time is None:
        if T is not None:
            end_time = float(T)
        elif len(event_times) > 0:
            end_time = float(np.max(event_times)) + 1.0
        else:
            end_time = 1.0
    """
    Fit a Hawkes process via multi-start L-BFGS-B MLE.

    Uses the O(n) Ozaki recursion for the log-likelihood and analytic gradients.
    Optimisation is performed in unconstrained θ-space with the parameterisation:
        θ = (log μ, logit_scaled_n, log β)
    ensuring μ > 0, β > 0, and n ∈ (n_min, n_max) ⊂ (0, 1).

    Multi-start strategy (≥ 5 starts):
        1. Near-Poisson:     n≈0.01, β≈1.0,  μ≈N/T
        2. High-clustering:  n≈0.90, β≈5.0,  μ≈0.1
        3. Mid-range:        n≈0.50, β≈2.0,  μ≈N/(2T)
        4. Fast-decay:       n≈0.30, β≈10.0, μ≈N/T
        5. Slow-decay:       n≈0.70, β≈0.5,  μ≈0.05
        6+. Random draws from interior of (θ_0, θ_1, θ_2) space

    Parameters
    ----------
    event_times : np.ndarray — sorted event times in [0, end_time]
    end_time    : float      — observation horizon T
    cfg         : dict       — override for optimiser config (optional)

    Returns
    -------
    dict with keys:
        mu               : float
        alpha            : float
        beta             : float
        branching_ratio  : float — α/β
        log_likelihood   : float — (positive) log-likelihood at solution
        aic              : float
        bic              : float
        converged        : bool
        reason           : str
        n_events         : int
        n_starts_converged: int
    """
    if cfg is None:
        cfg = {}
    full_cfg = {**_DEFAULT_CFG, **cfg}

    times = np.asarray(event_times, dtype=float)
    times = np.sort(times)
    times = times[times < end_time]
    T = float(end_time)
    N = len(times)

    mu_guess = max(N / T, 1e-3) if T > 0 else 1.0

    if N < 2:
        return _build_result(
            mu=mu_guess,
            alpha=0.0,
            beta=1.0,
            neg_log_lik=mu_guess * T,
            N=N,
            converged=False,
            reason="Insufficient events (N < 2) to fit Hawkes process",
            n_starts_converged=0,
        )

    n_starts = max(5, int(full_cfg.get("n_starts", 5)))
    max_iter = int(full_cfg.get("max_iter", 2000))
    ftol = float(full_cfg.get("ftol", 1e-14))
    gtol = float(full_cfg.get("gtol", 1e-8))
    n_min = float(full_cfg.get("n_min", _DEFAULT_CFG["n_min"]))
    n_max = float(full_cfg.get("n_max", _DEFAULT_CFG["n_max"]))

    mu_guess = max(N / T, 1e-3) if T > 0 else 1.0

    # Deterministic starting points
    fixed_starts = [
        # (mu, n, beta)
        (mu_guess,         0.01, 1.0),   # near-Poisson
        (0.1,              0.90, 5.0),   # high-clustering
        (mu_guess / 2.0,   0.50, 2.0),   # mid-range
        (mu_guess,         0.30, 10.0),  # fast-decay
        (0.05,             0.70, 0.5),   # slow-decay
    ]

    rng_init = np.random.default_rng(seed=12345 + N)
    random_starts = []
    for _ in range(max(0, n_starts - len(fixed_starts))):
        mu_r    = float(rng_init.uniform(1e-3, mu_guess * 2.0 + 1e-3))
        n_r     = float(rng_init.uniform(n_min + 0.01, min(n_max - 0.01, 0.95)))
        beta_r  = float(rng_init.uniform(0.1, 20.0))
        random_starts.append((mu_r, n_r, beta_r))

    all_starts = fixed_starts + random_starts

    best_result = None
    best_neg_ll = np.inf
    n_converged = 0

    for i, (mu0, n0, beta0) in enumerate(all_starts):
        alpha0 = n0 * beta0
        try:
            theta0 = _params_to_theta(mu0, alpha0, beta0, full_cfg)
        except Exception:
            continue

        try:
            res = optimize.minimize(
                _objective,
                theta0,
                args=(times, T, full_cfg),
                method="L-BFGS-B",
                jac=True,  # _objective returns (f, g)
                options={
                    "maxiter": max_iter,
                    "ftol": ftol,
                    "gtol": gtol,
                },
            )
        except Exception as exc:
            logger.debug("Start %d raised exception: %s", i, exc)
            continue

        if not np.isfinite(res.fun):
            continue

        if res.success or res.fun < best_neg_ll:
            if res.success:
                n_converged += 1
            if res.fun < best_neg_ll:
                best_neg_ll = res.fun
                best_result = res

    # Handle degenerate case: no valid optimisation
    if best_result is None:
        logger.warning("All optimisation starts failed. Returning fallback Poisson estimate.")
        mu_fallback = mu_guess
        alpha_fallback = 1e-4
        beta_fallback = 1.0
        neg_ll_fallback = _hawkes_loglik(mu_fallback, alpha_fallback, beta_fallback, times, T)
        return _build_result(
            mu=mu_fallback,
            alpha=alpha_fallback,
            beta=beta_fallback,
            neg_log_lik=neg_ll_fallback,
            N=N,
            converged=False,
            reason="all_starts_failed",
            n_starts_converged=0,
        )

    mu_opt, alpha_opt, beta_opt = _theta_to_params(best_result.x, full_cfg)

    # Recompute final loglik cleanly
    neg_ll_final = _hawkes_loglik(mu_opt, alpha_opt, beta_opt, times, T)

    converged = bool(best_result.success)
    reason = best_result.message if hasattr(best_result, "message") else "unknown"

    logger.info(
        "Hawkes MLE: μ=%.4f α=%.4f β=%.4f n=%.4f logL=%.4f converged=%s starts_ok=%d/%d",
        mu_opt, alpha_opt, beta_opt, alpha_opt / beta_opt,
        -neg_ll_final, converged, n_converged, n_starts,
    )

    return _build_result(
        mu=mu_opt,
        alpha=alpha_opt,
        beta=beta_opt,
        neg_log_lik=neg_ll_final,
        N=N,
        converged=converged,
        reason=reason,
        n_starts_converged=n_converged,
    )


def _build_result(
    mu: float,
    alpha: float,
    beta: float,
    neg_log_lik: float,
    N: int,
    converged: bool,
    reason: str,
    n_starts_converged: int,
) -> dict:
    """Assemble the canonical result dictionary."""
    log_lik = -neg_log_lik
    k_params = 3  # (mu, alpha, beta)
    aic = 2 * k_params - 2 * log_lik
    bic = k_params * math.log(max(N, 1)) - 2 * log_lik

    return dict(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        branching_ratio=float(alpha / max(beta, 1e-12)),
        log_likelihood=float(log_lik),
        aic=float(aic),
        bic=float(bic),
        converged=converged,
        reason=str(reason),
        n_events=N,
        n_starts_converged=n_starts_converged,
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Smoke test on synthetic Hawkes data
    from simulate import simulate_hawkes_branching

    rng = np.random.default_rng(42)
    T = 200.0
    mu_true, alpha_true, beta_true = 0.5, 0.6, 2.0
    times = simulate_hawkes_branching(mu_true, alpha_true, beta_true, T, rng)
    print(f"Simulated {len(times)} events over T={T}h")

    # Gradient check
    ok = verify_gradient(times[:50], T=50.0)
    print(f"Gradient verification: {'PASS' if ok else 'FAIL'}")

    # Fit
    result = fit_hawkes(times, T)
    print("MLE result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
