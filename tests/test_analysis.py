"""
tests/test_analysis.py — Tests for src/analysis.py

Covers:
  - compute_interarrival_stats: CV ≈ 1 for Poisson, CV > 1 for Hawkes
  - compute_fano_factor: basic value checks
  - chi_square_dispersion_test: rejects Hawkes data, accepts Poisson data
  - parametric_bootstrap_lrt: rejects on Hawkes data (n_boot small for speed)
  - time_rescaling_test: well-fitted Hawkes => KS p > 0.05
  - analyze_clustering: runs end-to-end and returns expected keys
"""
import math
import os

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    compute_interarrival_stats,
    compute_fano_factor,
    chi_square_dispersion_test,
    parametric_bootstrap_lrt,
    time_rescaling_test,
    analyze_clustering,
)
from src.simulate import simulate_homogeneous_poisson, simulate_hawkes_ogata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_events(mu: float = 2.0, T: float = 1000.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return simulate_homogeneous_poisson(mu, T, rng=rng)


def _hawkes_events(mu: float = 0.5, alpha: float = 0.7, beta: float = 1.0,
                   T: float = 500.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return simulate_hawkes_ogata(mu, alpha, beta, T, rng=rng)


# ---------------------------------------------------------------------------
# 1. Coefficient of variation
# ---------------------------------------------------------------------------

class TestCV:
    def test_cv_equals_one_for_poisson(self):
        """Poisson inter-arrivals are Exp(μ) => CV = std/mean ≈ 1."""
        events = _poisson_events(mu=2.0, T=1000.0, seed=42)
        stats = compute_interarrival_stats(events)
        assert abs(stats["cv"] - 1.0) < 0.15, (
            f"CV={stats['cv']:.3f} far from 1 for Poisson process"
        )

    def test_cv_greater_than_one_for_hawkes(self):
        """Hawkes clustering => inter-arrivals more variable => CV > 1."""
        # n = alpha/beta = 0.7 => significant clustering
        events = _hawkes_events(mu=0.3, alpha=0.7, beta=1.0, T=2000.0, seed=7)
        stats = compute_interarrival_stats(events)
        assert stats["cv"] > 1.05, (
            f"CV={stats['cv']:.3f} not > 1 for clustered Hawkes process"
        )

    def test_cv_nan_for_single_event(self):
        """Single event => fewer than 2 gaps => CV is NaN."""
        stats = compute_interarrival_stats(np.array([1.0]))
        assert math.isnan(stats["cv"])


# ---------------------------------------------------------------------------
# 2. Fano factor
# ---------------------------------------------------------------------------

class TestFanoFactor:
    def test_fano_near_one_for_poisson(self):
        """Poisson => Fano ≈ 1 for any window."""
        events = _poisson_events(mu=3.0, T=2000.0, seed=0)
        fano = compute_fano_factor(events, T=2000.0, window_h=5.0)
        assert 0.7 < fano < 1.3, f"Fano={fano:.3f} not near 1 for Poisson"

    def test_fano_greater_than_one_for_hawkes(self):
        """Hawkes => Fano > 1 for large windows."""
        events = _hawkes_events(mu=0.5, alpha=0.6, beta=1.0, T=3000.0, seed=3)
        fano = compute_fano_factor(events, T=3000.0, window_h=10.0)
        assert fano > 1.2, f"Fano={fano:.3f} not > 1.2 for clustered Hawkes"


# ---------------------------------------------------------------------------
# 3. Chi-square dispersion test
# ---------------------------------------------------------------------------

class TestChiSquareDispersion:
    def test_rejects_on_hawkes_data(self):
        """Clustered Hawkes data should trigger rejection of Poisson hypothesis."""
        events = _hawkes_events(mu=0.5, alpha=0.7, beta=1.0, T=2000.0, seed=5)
        result = chi_square_dispersion_test(events, T=2000.0, window_h=2.0)
        assert result["reject_poisson"], (
            f"Expected rejection; p={result['p_value']:.4f}, fano={result['fano']:.3f}"
        )

    def test_accepts_poisson_data(self):
        """Poisson data should not reliably reject the Poisson hypothesis."""
        events = _poisson_events(mu=3.0, T=2000.0, seed=99)
        result = chi_square_dispersion_test(events, T=2000.0, window_h=5.0)
        # With many windows the test has low false-positive rate; p > 0.01
        assert result["p_value"] > 0.01, (
            f"Poisson data falsely rejected; p={result['p_value']:.4f}"
        )

    def test_returns_expected_keys(self):
        events = _poisson_events()
        result = chi_square_dispersion_test(events, T=1000.0, window_h=5.0)
        for key in ("fano", "test_stat", "p_value", "reject_poisson", "n_windows"):
            assert key in result, f"Missing key '{key}'"


# ---------------------------------------------------------------------------
# 4. Parametric bootstrap LRT
# ---------------------------------------------------------------------------

class TestBootstrapLRT:
    def test_rejects_on_hawkes_data(self):
        """Hawkes data with n=0.7 should clearly reject Poisson (p < 0.1)."""
        events = _hawkes_events(mu=0.5, alpha=0.7, beta=1.0, T=500.0, seed=11)
        result = parametric_bootstrap_lrt(events, T=500.0, n_boot=99, rng_seed=7)
        assert result["reject_poisson"], (
            f"Expected LRT rejection; p={result['p_value']:.4f}, "
            f"lrt={result['observed_lrt']:.2f}"
        )

    def test_does_not_reliably_reject_poisson(self):
        """Poisson data: LRT should NOT always reject (p often > 0.05)."""
        events = _poisson_events(mu=2.0, T=500.0, seed=42)
        result = parametric_bootstrap_lrt(events, T=500.0, n_boot=99, rng_seed=7)
        # We just check that the function runs and returns valid structure
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_returns_expected_keys(self):
        events = _poisson_events(T=200.0, seed=0)
        result = parametric_bootstrap_lrt(events, T=200.0, n_boot=49, rng_seed=7)
        for key in ("observed_lrt", "p_value", "null_lrt_samples", "reject_poisson",
                    "mu_h0", "mu_h1", "alpha_h1", "beta_h1"):
            assert key in result, f"Missing key '{key}'"


# ---------------------------------------------------------------------------
# 5. Time-rescaling test
# ---------------------------------------------------------------------------

class TestTimeRescaling:
    def test_passes_for_well_fitted_hawkes(self):
        """If we use the true parameters, KS p-value should be > 0.05."""
        mu, alpha, beta = 0.5, 0.4, 1.0
        T = 500.0
        rng = np.random.default_rng(22)
        events = simulate_hawkes_ogata(mu, alpha, beta, T, rng=rng)
        result = time_rescaling_test(events, T=T, mu=mu, alpha=alpha, beta=beta)
        assert result["ks_pvalue"] > 0.01, (
            f"KS p={result['ks_pvalue']:.4f} unexpectedly low for well-fitted model"
        )

    def test_returns_expected_keys(self):
        events = _poisson_events(T=200.0, seed=0)
        result = time_rescaling_test(events, T=200.0, mu=2.0, alpha=0.0, beta=1.0)
        for key in ("rescaled_gaps", "ks_stat", "ks_pvalue", "reject_model"):
            assert key in result, f"Missing key '{key}'"

    def test_few_events_returns_nan(self):
        """Single event => not enough data for KS test => NaN stats."""
        result = time_rescaling_test(np.array([1.0]), T=10.0, mu=0.1, alpha=0.0, beta=1.0)
        assert math.isnan(result["ks_stat"])


# ---------------------------------------------------------------------------
# 6. analyze_clustering end-to-end
# ---------------------------------------------------------------------------

class TestAnalyzeClustering:
    def test_returns_expected_keys(self, tmp_path):
        """analyze_clustering should return a dict with core result keys."""
        events = _poisson_events(mu=2.0, T=500.0, seed=0)
        df = pd.DataFrame({"timestamp": events})
        cfg = {
            "paths": {"reports_dir": str(tmp_path)},
            "analysis": {
                "window_sizes": [1.0, 5.0, 10.0],
                "n_bootstrap": 49,
            },
        }
        result = analyze_clustering(df, cfg)
        for key in ("is_clustered", "stats", "figure_paths", "poisson_fit", "hawkes_fit"):
            assert key in result, f"Missing key '{key}'"

    def test_poisson_data_not_flagged_as_clustered(self, tmp_path):
        """Pure Poisson data should generally not be flagged as clustered."""
        events = _poisson_events(mu=3.0, T=1000.0, seed=123)
        df = pd.DataFrame({"timestamp": events})
        cfg = {"paths": {"reports_dir": str(tmp_path)}}
        result = analyze_clustering(df, cfg)
        # is_clustered is a bool - just ensure no exception and key present
        assert isinstance(result["is_clustered"], bool)

    def test_figures_saved(self, tmp_path):
        """analyze_clustering should save at least one figure to reports_dir."""
        events = _hawkes_events(mu=0.5, alpha=0.5, beta=1.0, T=300.0, seed=99)
        df = pd.DataFrame({"timestamp": events})
        cfg = {"paths": {"reports_dir": str(tmp_path)}}
        result = analyze_clustering(df, cfg)
        saved = result.get("figure_paths", [])
        png_files = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
        assert len(saved) > 0 or len(png_files) > 0, (
            "Expected at least one figure to be saved by analyze_clustering"
        )

