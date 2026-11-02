"""
tests/test_backtest.py — Tests for src/backtest.py

Covers:
  - chronological_split: proportions and embargo gaps
  - merge_alert_episodes: nearby alerts are merged correctly
  - classify_episodes: hand-crafted TP/FP/miss counts
  - compute_metrics: formula correctness
  - bootstrap_ci: returns valid interval
"""
import math

import numpy as np
import pytest

from src.backtest import (
    chronological_split,
    merge_alert_episodes,
    classify_episodes,
    compute_metrics,
    bootstrap_ci,
)


# ---------------------------------------------------------------------------
# 1. chronological_split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def _make_events(self, T: float = 1000.0, rate: float = 2.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = rng.poisson(rate * T)
        return np.sort(rng.uniform(0, T, n))

    def test_proportions_without_embargo(self):
        """Train / val / test fractions add up and are in the right ranges."""
        T = 1000.0
        events = self._make_events(T=T, rate=1.0)
        wave_labels = [(200.0, 210.0), (600.0, 615.0), (850.0, 860.0)]
        split = chronological_split(events, wave_labels, T,
                                    train_frac=0.6, val_frac=0.2, embargo_h=0.0)

        # Cut-points
        assert split["t_train_end"] == pytest.approx(600.0)
        assert split["t_val_end"]   == pytest.approx(800.0)

        # Event counts in expected windows
        train_t = split["train_times"]
        val_t   = split["val_times"]
        test_t  = split["test_times"]

        assert np.all(train_t < 600.0)
        assert np.all((val_t >= 600.0) & (val_t < 800.0))
        assert np.all(test_t >= 800.0)

    def test_embargo_removes_boundary_events(self):
        """Events in embargo zone should be excluded from val and test sets."""
        T = 100.0
        embargo = 4.0
        events = np.arange(0.5, T, 1.0)  # events every 1 h
        wave_labels: list = []
        split = chronological_split(events, wave_labels, T,
                                    train_frac=0.6, val_frac=0.2, embargo_h=embargo)

        train_end = 60.0
        val_end   = 80.0

        # No val event in [60, 64)
        val_t = split["val_times"]
        assert len(val_t[val_t < train_end + embargo]) == 0

        # No test event in [80, 84)
        test_t = split["test_times"]
        assert len(test_t[test_t < val_end + embargo]) == 0

    def test_wave_labels_partitioned_correctly(self):
        """Wave labels should be assigned to correct splits."""
        T = 100.0
        events = np.array([])
        wave_labels = [
            (10.0, 20.0),   # train
            (65.0, 70.0),   # val (60–80)
            (85.0, 90.0),   # test (80+)
        ]
        split = chronological_split(events, wave_labels, T,
                                    train_frac=0.6, val_frac=0.2, embargo_h=0.0)

        assert (10.0, 20.0) in split["train_waves"]
        assert (65.0, 70.0) in split["val_waves"]
        assert (85.0, 90.0) in split["test_waves"]


# ---------------------------------------------------------------------------
# 2. merge_alert_episodes
# ---------------------------------------------------------------------------

class TestMergeAlertEpisodes:
    def test_no_alerts_returns_empty(self):
        assert merge_alert_episodes(np.array([])) == []

    def test_single_alert_becomes_one_episode(self):
        episodes = merge_alert_episodes(np.array([5.0]))
        assert len(episodes) == 1
        assert episodes[0] == (5.0, 5.0)

    def test_nearby_alerts_merged(self):
        """Alerts within cooldown_h should form one episode."""
        alerts = np.array([1.0, 1.2, 1.4])
        episodes = merge_alert_episodes(alerts, cooldown_h=0.5)
        assert len(episodes) == 1
        assert episodes[0][0] == pytest.approx(1.0)
        assert episodes[0][1] == pytest.approx(1.4)

    def test_far_alerts_separate(self):
        """Alerts separated by more than cooldown_h form distinct episodes."""
        alerts = np.array([1.0, 5.0, 9.0])
        episodes = merge_alert_episodes(alerts, cooldown_h=0.5)
        assert len(episodes) == 3

    def test_mixed_merging(self):
        """[1.0, 1.3, 5.0, 5.2] with cooldown=0.5 => 2 episodes."""
        alerts = np.array([1.0, 1.3, 5.0, 5.2])
        episodes = merge_alert_episodes(alerts, cooldown_h=0.5)
        assert len(episodes) == 2
        assert episodes[0] == (pytest.approx(1.0), pytest.approx(1.3))
        assert episodes[1] == (pytest.approx(5.0), pytest.approx(5.2))


# ---------------------------------------------------------------------------
# 3. classify_episodes (hand example)
# ---------------------------------------------------------------------------

class TestClassifyEpisodes:
    """
    Wave at (10, 20).  lead_window = 4 h.
    Episode A starts at 7 h  => in [10-4, 10] = [6, 10] => TP, lead=3h
    Episode B starts at 25 h => not in any wave window => FP
    Episode C starts at 30 h => not in any wave window => FP
    """
    def test_hand_example(self):
        episodes    = [(7.0, 8.0), (25.0, 26.0), (30.0, 31.0)]
        wave_labels = [(10.0, 20.0)]
        result = classify_episodes(episodes, wave_labels, lead_window_h=4.0)

        assert result["tps"]          == 1
        assert result["fps"]          == 2
        assert result["missed_waves"] == 0
        assert len(result["lead_times"]) == 1
        assert result["lead_times"][0] == pytest.approx(3.0)  # 10 - 7

    def test_missed_wave(self):
        """No episodes => all waves missed."""
        wave_labels = [(10.0, 20.0), (50.0, 60.0)]
        result = classify_episodes([], wave_labels)
        assert result["tps"]          == 0
        assert result["fps"]          == 0
        assert result["missed_waves"] == 2

    def test_episode_inside_wave_not_tp_not_fp(self):
        """An episode entirely inside a wave window (not a TP) should not be FP."""
        # Episode starts at 12, wave is 10–20.  start NOT in [6,10] => not TP.
        # But episode [12, 13] is inside [6, 20] => not FP either.
        episodes    = [(12.0, 13.0)]
        wave_labels = [(10.0, 20.0)]
        result = classify_episodes(episodes, wave_labels, lead_window_h=4.0)
        assert result["tps"] == 0
        assert result["fps"] == 0
        assert result["missed_waves"] == 1

    def test_two_waves_one_tp_one_miss(self):
        """Episode covers first wave; second wave has no episode."""
        episodes    = [(6.0, 7.0)]   # leads wave at t=10 by 4 h => TP
        wave_labels = [(10.0, 20.0), (50.0, 60.0)]
        result = classify_episodes(episodes, wave_labels, lead_window_h=4.0)
        assert result["tps"]          == 1
        assert result["missed_waves"] == 1


# ---------------------------------------------------------------------------
# 4. compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_detection(self):
        result = compute_metrics(tps=5, fps=0, missed_waves=0,
                                 lead_times=[2.0, 1.5, 3.0, 2.5, 1.0], T_days=10.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"]    == pytest.approx(1.0)
        assert result["f1"]        == pytest.approx(1.0)
        assert result["fp_per_day"] == pytest.approx(0.0)

    def test_no_detections(self):
        result = compute_metrics(tps=0, fps=0, missed_waves=3,
                                 lead_times=[], T_days=5.0)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"]    == pytest.approx(0.0)
        assert result["f1"]        == pytest.approx(0.0)
        assert math.isnan(result["mean_lead_h"])

    def test_precision_recall_formula(self):
        # 2 TP, 8 FP, 1 missed wave => precision=2/10=0.2, recall=2/3≈0.667
        result = compute_metrics(tps=2, fps=8, missed_waves=1,
                                 lead_times=[1.0, 2.0], T_days=20.0)
        assert result["precision"] == pytest.approx(2 / 10)
        assert result["recall"]    == pytest.approx(2 / 3)
        assert result["fp_per_day"] == pytest.approx(8 / 20)


# ---------------------------------------------------------------------------
# 5. bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_valid_interval(self):
        values = [0.5, 0.6, 0.4, 0.55, 0.45, 0.7, 0.3]
        ci = bootstrap_ci(values, n_boot=500, ci=0.95, rng_seed=7)
        assert ci["lower"] <= ci["mean"] <= ci["upper"]

    def test_ci_width_reasonable(self):
        """With n=10 samples CI should be non-zero but bounded."""
        values = list(range(10))   # 0..9, mean=4.5
        ci = bootstrap_ci(values, n_boot=1000, ci=0.95, rng_seed=42)
        assert ci["lower"] < ci["mean"] < ci["upper"]
        assert ci["upper"] - ci["lower"] < 8  # sanity: can't be wider than range

    def test_empty_values_returns_nan(self):
        ci = bootstrap_ci([], n_boot=100)
        assert math.isnan(ci["mean"])
        assert math.isnan(ci["lower"])
        assert math.isnan(ci["upper"])

    def test_all_nan_returns_nan(self):
        ci = bootstrap_ci([float("nan"), float("nan")], n_boot=50)
        assert math.isnan(ci["mean"])

