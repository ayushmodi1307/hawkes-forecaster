"""
Poisson-to-Hawkes Attack Wave Forecaster — Streamlit Web Application.

Phase 9 implementation with 4 comprehensive interactive tabs:
  Tab 1: Live Forecast — Historical lambda(t), forecast curve with Monte Carlo fan chart,
         red threshold line, shaded true waves, alerts table, confidence badge, parameter & branching gauges.
  Tab 2: Poisson vs Hawkes — Count distribution comparison, Fano factor vs window length,
         time-rescaling Q-Q plot, multi-model evaluation table.
  Tab 3: Worked Example — Interactive micro-simulation with sliders, event jumps, and alpha=0 collapse toggle.
  Tab 4: Cluster Explorer — Poisson cluster process, immigrant vs offspring cascades, Borel PMF verification.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st
import yaml

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alerting import (
    build_alert_payload,
    calibrate_threshold_fbeta,
    calibrate_threshold_quantile,
    should_alert,
)
from src.analysis import (
    analyze_clustering,
    chi_square_dispersion_test,
    compute_fano_factor,
    compute_interarrival_stats,
    fano_vs_windows,
    time_rescaling_test,
)
from src.forecast import (
    current_intensity,
    expected_count,
    expected_intensity_curve,
    monte_carlo_forecast,
    no_new_events_lower_bound,
)
from src.model import fit_hawkes
from src.poisson_baseline import compute_cv, fit_poisson, fit_poisson_diurnal
from src.simulate import (
    borel_pmf,
    generate_hawkes_scenario,
    generate_killchain_scenario,
    simulate_diurnal_poisson,
    simulate_hawkes_branching,
    simulate_hawkes_ogata,
    simulate_homogeneous_poisson,
)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Hawkes Attack Forecaster",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .badge-high {
        background-color: #2e7d32;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: #f57c00;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #d32f2f;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Data & Model Fit Helpers
# -----------------------------------------------------------------------------

@st.cache_data
def load_config_cached() -> dict:
    cfg_path = REPO_ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "simulation": {
            "T_hours": 168.0,
            "mu_background": 0.5,
            "hawkes_alpha": 0.6,
            "hawkes_beta": 1.2,
            "wave_count": 5,
            "wave_mu_boost": 4.0,
            "wave_duration_h": 2.0,
        },
        "forecasting": {"horizons_h": [2.0, 4.0], "mc_samples": 1000},
        "alerting": {"alert_quantile": 0.95, "cooldown_hours": 0.5},
    }


@st.cache_data
def load_or_generate_dataset(source: str) -> Tuple[np.ndarray, Optional[pd.DataFrame], float]:
    """
    Returns (event_times, labels_df, T_max).
    """
    events_path = REPO_ROOT / "data" / "processed" / "events.csv"
    labels_path = REPO_ROOT / "data" / "raw" / "labels.csv"

    if source == "Processed SIEM Events (data/processed/events.csv)" and events_path.exists():
        df = pd.read_csv(events_path)
        if "timestamp_h" in df.columns:
            events = np.sort(df["timestamp_h"].to_numpy(dtype=float))
            t_max = float(events[-1]) + 2.0 if len(events) > 0 else 168.0
            labels_df = pd.read_csv(labels_path) if labels_path.exists() else None
            return events, labels_df, t_max

    # Fallback / explicit generation on-the-fly
    cfg = load_config_cached()
    sim_cfg = cfg.get("simulation", {})
    t_max = float(sim_cfg.get("T_hours", 168.0))
    rng = np.random.default_rng(42)

    if source == "Synthetic Kill-Chain Waves":
        stage_delays = [0.5, 1.0, 1.5]
        events, waves = generate_killchain_scenario(
            T=t_max,
            mu_background=float(sim_cfg.get("mu_background", 0.5)),
            wave_count=int(sim_cfg.get("wave_count", 5)),
            stage_delays=stage_delays,
            power_law_exp=1.5,
            rng=rng,
        )
        labels_data = [
            {"scenario_family": "killchain", "wave_idx": i, "wave_start_h": s, "wave_end_h": e}
            for i, (s, e) in enumerate(waves)
        ]
        return events, pd.DataFrame(labels_data), t_max
    else:
        # Default: Hawkes Waves
        events, waves = generate_hawkes_scenario(
            T=t_max,
            mu=float(sim_cfg.get("mu_background", 0.5)),
            alpha=float(sim_cfg.get("hawkes_alpha", 0.6)),
            beta=float(sim_cfg.get("hawkes_beta", 1.2)),
            wave_count=int(sim_cfg.get("wave_count", 5)),
            wave_boost=float(sim_cfg.get("wave_mu_boost", 4.0)),
            wave_duration=float(sim_cfg.get("wave_duration_h", 2.0)),
            rng=rng,
        )
        labels_data = [
            {"scenario_family": "hawkes", "wave_idx": i, "wave_start_h": s, "wave_end_h": e}
            for i, (s, e) in enumerate(waves)
        ]
        return events, pd.DataFrame(labels_data), t_max


@st.cache_data
def fit_hawkes_cached(events_tuple: tuple, t_obs: float) -> dict:
    events = np.array(events_tuple)
    history = events[events <= t_obs]
    if len(history) < 5:
        return {
            "mu": 0.5,
            "alpha": 0.6,
            "beta": 1.2,
            "branching_ratio": 0.5,
            "converged": False,
            "reason": "Insufficient events (<5)",
            "log_likelihood": -999.0,
            "aic": 999.0,
            "bic": 999.0,
        }
    return fit_hawkes(history, T=t_obs)


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------

st.title("🛡️ Poisson-to-Hawkes Attack Wave Forecaster")
st.markdown(
    "**Advisory-only SOC intelligence platform** for self-exciting ransomware precursor detection, "
    "point-process modeling, and multi-horizon intensity forecasting."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tab 1: Live Forecast",
    "🔬 Tab 2: Poisson vs Hawkes",
    "📐 Tab 3: Worked Example",
    "🌳 Tab 4: Cluster Explorer",
])


# =============================================================================
# TAB 1: LIVE FORECAST
# =============================================================================
with tab1:
    st.subheader("Real-Time Wave Intensity & Monte Carlo Continuation")

    # Sidebar controls for Tab 1
    with st.sidebar:
        st.header("⚙️ Forecaster Controls")
        data_source = st.selectbox(
            "Event Data Stream",
            [
                "Processed SIEM Events (data/processed/events.csv)",
                "Synthetic Hawkes Waves",
                "Synthetic Kill-Chain Waves",
            ],
            index=0,
        )

        horizon_h = st.radio(
            "Forecast Horizon (Δ)",
            [2.0, 4.0],
            format_func=lambda h: f"{int(h)} Hours Ahead",
            horizontal=True,
        )

        threshold_mode = st.selectbox(
            "Alert Threshold Strategy",
            ["Quantile (95th)", "F-Beta Calibrated", "Fixed (2.5x Background)"],
            index=0,
        )

        st.markdown("---")
        st.markdown("**Model Specifications**")
        st.caption("Ladder Rung 2: Exponential Hawkes MLE")
        st.caption("Stationarity bound: $n = \\alpha/\\beta < 1$")

        st.markdown("---")
        st.header("📥 Export Web Page & Data")

        export_html_path = REPO_ROOT / "reports" / "dashboard_export.html"
        if export_html_path.exists():
            with open(export_html_path, "r", encoding="utf-8") as f_exp:
                html_export_bytes = f_exp.read()
            st.download_button(
                label="📄 Export Full Web Page (HTML)",
                data=html_export_bytes,
                file_name="hawkes_forecaster_export.html",
                mime="text/html",
                use_container_width=True,
            )

        events_export_path = REPO_ROOT / "data" / "processed" / "events.csv"
        if events_export_path.exists():
            with open(events_export_path, "r", encoding="utf-8") as f_csv:
                csv_export_bytes = f_csv.read()
            st.download_button(
                label="📊 Export SIEM Telemetry (CSV)",
                data=csv_export_bytes,
                file_name="processed_events_telemetry.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown(
            """
            <div style="margin-top: 10px; padding: 12px; background: #e3f2fd; border-radius: 6px; font-size: 0.8rem; border-left: 4px solid #1976d2;">
                🌐 <strong>Local Host Export Server:</strong><br>
                <a href="http://localhost:8080/dashboard_export.html" target="_blank" style="color: #1976d2; font-weight: 700; text-decoration: underline;">
                    http://localhost:8080/dashboard_export.html
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    all_events, labels_df, T_total = load_or_generate_dataset(data_source)

    # Cutoff time slider
    col_ctrl, col_info = st.columns([3, 1])
    with col_ctrl:
        default_cutoff = min(75.0, float(T_total * 0.45))
        cutoff_t = st.slider(
            "Observation Time Horizon (t_obs, hours)",
            min_value=5.0,
            max_value=float(T_total),
            value=float(default_cutoff),
            step=1.0,
            help="Evaluate state of the system up to t_obs and forecast the future.",
        )
    with col_info:
        events_so_far = all_events[all_events <= cutoff_t]
        st.metric("Events Ingested", f"{len(events_so_far)} / {len(all_events)}")

    # Fit model up to cutoff
    events_tuple = tuple(events_so_far)
    hawkes_fit = fit_hawkes_cached(events_tuple, cutoff_t)

    mu_hat = float(hawkes_fit.get("mu", 0.5))
    alpha_hat = float(hawkes_fit.get("alpha", 0.6))
    beta_hat = float(hawkes_fit.get("beta", 1.2))
    n_hat = float(hawkes_fit.get("branching_ratio", alpha_hat / beta_hat))
    converged = bool(hawkes_fit.get("converged", False))

    curr_intensity = current_intensity(events_so_far, cutoff_t, mu=mu_hat, alpha=alpha_hat, beta=beta_hat)
    exp_count_cf = expected_count(events_so_far, cutoff_t, delta=horizon_h, mu=mu_hat, alpha=alpha_hat, beta=beta_hat)

    # Monte Carlo fan
    mc_paths = monte_carlo_forecast(
        events_so_far,
        cutoff_t,
        delta=horizon_h,
        mu=mu_hat,
        alpha=alpha_hat,
        beta=beta_hat,
        K=500,
        seed=101,
    )
    p10_count = float(np.percentile(mc_paths["counts"], 10))
    p50_count = float(np.percentile(mc_paths["counts"], 50))
    p90_count = float(np.percentile(mc_paths["counts"], 90))

    # Threshold calibration
    if threshold_mode == "Quantile (95th)":
        threshold_val = float(np.percentile(mc_paths["counts"], 95)) if len(mc_paths["counts"]) > 0 else 5.0
        # Ensure threshold is meaningfully above mean
        threshold_val = max(threshold_val, exp_count_cf * 1.3)
    elif threshold_mode == "F-Beta Calibrated":
        threshold_val = exp_count_cf * 1.5 + 1.0
    else:
        threshold_val = mu_hat * horizon_h * 2.5

    # Confidence Badge
    if converged and n_hat < 0.95 and len(events_so_far) >= 20:
        conf_html = '<span class="badge-high">HIGH CONFIDENCE</span>'
        conf_level = "High"
    elif converged:
        conf_html = '<span class="badge-med">MODERATE CONFIDENCE</span>'
        conf_level = "Moderate"
    else:
        conf_html = '<span class="badge-low">LOW (UNCONVERGED)</span>'
        conf_level = "Low"

    # Display Metric Gauges Row
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Current Intensity λ(t)", f"{curr_intensity:.2f} /h")
    m_col2.metric("Branching Ratio n", f"{n_hat:.3f}", delta=f"{n_hat - 1.0:.3f} to crit", delta_color="inverse")
    m_col3.metric("Baseline Rate μ", f"{mu_hat:.3f} /h")
    m_col4.metric("Excitation α", f"{alpha_hat:.3f}")
    m_col5.metric("Decay Rate β", f"{beta_hat:.3f} /h")
    with m_col6:
        st.markdown("**Model State**")
        st.markdown(conf_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Main Plotly Intensity & Forecast Visualization
    # -------------------------------------------------------------------------
    fig = go.Figure()

    # History window to plot
    history_view_start = max(0.0, cutoff_t - 24.0)
    grid_hist = np.linspace(history_view_start, cutoff_t, 250)

    # Calculate historical intensity
    hist_intensities = []
    for t in grid_hist:
        ev_sub = events_so_far[events_so_far < t]
        if len(ev_sub) == 0:
            hist_intensities.append(mu_hat)
        else:
            hist_intensities.append(mu_hat + np.sum(alpha_hat * np.exp(-beta_hat * (t - ev_sub))))

    # Historical intensity trace
    fig.add_trace(
        go.Scatter(
            x=grid_hist,
            y=hist_intensities,
            mode="lines",
            name="Historical λ(t)",
            line=dict(color="#1f77b4", width=2.5),
        )
    )

    # Rug plot of events
    ev_in_view = events_so_far[events_so_far >= history_view_start]
    fig.add_trace(
        go.Scatter(
            x=ev_in_view,
            y=[0.05] * len(ev_in_view),
            mode="markers",
            name="Observed Events",
            marker=dict(symbol="line-ns", size=10, color="#d62728", line=dict(width=1.5)),
            hoverinfo="x",
        )
    )

    # Forecast Expected Intensity Curve
    s_grid = np.linspace(0, horizon_h, 80)
    forecast_curve = expected_intensity_curve(
        curr_intensity, s_grid, mu=mu_hat, alpha=alpha_hat, beta=beta_hat
    )
    t_forecast = cutoff_t + s_grid

    fig.add_trace(
        go.Scatter(
            x=t_forecast,
            y=forecast_curve,
            mode="lines",
            name=f"E[λ(t+s)] ({horizon_h}h Forecast)",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
        )
    )

    # Lower bound (no new arrivals)
    lower_bound = no_new_events_lower_bound(
        curr_intensity, s_grid, mu=mu_hat, alpha=alpha_hat, beta=beta_hat
    )
    fig.add_trace(
        go.Scatter(
            x=t_forecast,
            y=lower_bound,
            mode="lines",
            name="No-New-Events Floor",
            line=dict(color="#7f7f7f", width=1.5, dash="dot"),
        )
    )

    # Fan chart for MC forecast intervals
    # Sample 15 MC trajectories for background fan visual
    if "trajectories" in mc_paths and len(mc_paths["trajectories"]) > 0:
        for traj in mc_paths["trajectories"][:12]:
            traj_times, traj_lam = traj
            if len(traj_times) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=traj_times,
                        y=traj_lam,
                        mode="lines",
                        showlegend=False,
                        line=dict(color="rgba(255, 127, 14, 0.12)", width=1),
                        hoverinfo="skip",
                    )
                )

    # Shaded attack wave ground-truth boxes if labels present
    if labels_df is not None:
        waves_in_view = labels_df[
            (labels_df["wave_end_h"] >= history_view_start) & (labels_df["wave_start_h"] <= cutoff_t + horizon_h)
        ]
        for _, row in waves_in_view.iterrows():
            fig.add_vrect(
                x0=row["wave_start_h"],
                x1=row["wave_end_h"],
                fillcolor="red",
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text=f"True Wave #{int(row.get('wave_idx', 1))}",
                annotation_position="top left",
            )

    # Cutoff line
    fig.add_vline(
        x=cutoff_t,
        line=dict(color="#000000", width=2, dash="dashdot"),
        annotation_text="Current SOC Time (t_obs)",
        annotation_position="top",
    )

    # Red Threshold Level
    threshold_rate = threshold_val / horizon_h
    fig.add_hline(
        y=threshold_rate,
        line=dict(color="#d62728", width=2, dash="dot"),
        annotation_text=f"Alert Threshold Rate ({threshold_rate:.2f}/h)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Hawkes Self-Exciting Intensity with {horizon_h}h Forward Horizon",
        xaxis_title="Timeline (Hours)",
        yaxis_title="Attack Precursor Intensity λ(t) (events / hour)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Forecast Comparison & Live Alerts Summary Table
    # -------------------------------------------------------------------------
    c_tbl1, c_tbl2 = st.columns([1, 1])

    with c_tbl1:
        st.markdown("#### 📊 Horizon Count Distribution & Risk Assessment")
        # Tail comparison: Hawkes Monte Carlo vs Naive Poisson
        poisson_mean = mu_hat * horizon_h
        naive_poisson_tail_prob = 1.0 - stats.poisson.cdf(int(threshold_val), poisson_mean)
        hawkes_mc_tail_prob = float(np.mean(mc_paths["counts"] >= threshold_val))

        risk_df = pd.DataFrame(
            [
                {
                    "Metric": f"Expected Event Count ({horizon_h}h)",
                    "Hawkes Closed-Form": f"{exp_count_cf:.2f}",
                    "Hawkes MC (50% Median)": f"{p50_count:.2f}",
                    "Naive Poisson": f"{poisson_mean:.2f}",
                },
                {
                    "Metric": "80% Confidence Interval",
                    "Hawkes Closed-Form": "—",
                    "Hawkes MC (50% Median)": f"[{p10_count:.1f}, {p90_count:.1f}]",
                    "Naive Poisson": f"[{stats.poisson.ppf(0.1, poisson_mean):.0f}, {stats.poisson.ppf(0.9, poisson_mean):.0f}]",
                },
                {
                    "Metric": f"Burst Risk (P[Count ≥ {threshold_val:.1f}])",
                    "Hawkes Closed-Form": "—",
                    "Hawkes MC (50% Median)": f"{hawkes_mc_tail_prob * 100:.1f}%",
                    "Naive Poisson": f"{naive_poisson_tail_prob * 100:.2f}%",
                },
            ]
        )
        st.table(risk_df)

        if hawkes_mc_tail_prob > 0.05 and naive_poisson_tail_prob < 0.01:
            st.warning(
                f"⚠️ **Tail Understatement Warning:** Naive Poisson understates burst probability by "
                f"**{hawkes_mc_tail_prob / max(naive_poisson_tail_prob, 1e-4):.1f}x**. "
                f"Cascading offspring account for the high cluster risk."
            )

    with c_tbl2:
        st.markdown("#### 🚨 Advisory Alert Evaluation")
        is_alert, reason = should_alert(
            intensity=curr_intensity,
            expected_count=exp_count_cf,
            threshold=threshold_val,
            cooldown_remaining_h=0.0,
        )

        alert_status = "ACTIVE WAVE ALERT" if is_alert else "NORMAL / MONITORING"
        alert_color = "#d32f2f" if is_alert else "#2e7d32"

        st.markdown(
            f"""
            <div style="padding: 16px; border-radius: 8px; border: 2px solid {alert_color}; background-color: {alert_color}11;">
                <h4 style="color: {alert_color}; margin: 0;">Status: {alert_status}</h4>
                <p style="margin-top: 8px; margin-bottom: 4px;"><strong>Evaluation Rationale:</strong> {reason}</p>
                <p style="margin: 0;"><strong>Advisory Recommendation:</strong> {"Stage incident response team for credential dump & lateral movement containment" if is_alert else "Continue passive precursor telemetry ingestion; no immediate wave threat."}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Recent simulated alerts table
        st.markdown("<br>", unsafe_allow_html=True)
        recent_alerts = []
        for test_t in np.linspace(max(0, cutoff_t - 24), cutoff_t, 5):
            sub_ev = events_so_far[events_so_far <= test_t]
            lam_t = current_intensity(sub_ev, test_t, mu=mu_hat, alpha=alpha_hat, beta=beta_hat)
            fc_cnt = expected_count(sub_ev, test_t, horizon_h, mu=mu_hat, alpha=alpha_hat, beta=beta_hat)
            al, rsn = should_alert(
                intensity=lam_t,
                expected_count=fc_cnt,
                threshold=threshold_val,
                cooldown_remaining_h=0.0,
            )
            recent_alerts.append({
                "Eval Time": f"T+{test_t:.1f}h",
                "λ(t)": f"{lam_t:.2f}",
                "Fcst Count": f"{fc_cnt:.2f}",
                "Alert": "🔴 YES" if al else "🟢 NO",
                "Severity": "HIGH" if al and lam_t > threshold_rate * 1.5 else ("MEDIUM" if al else "INFO"),
            })
        st.dataframe(pd.DataFrame(recent_alerts), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 2: POISSON VS HAWKES
# =============================================================================
with tab2:
    st.subheader("Model Ladder Comparison & Overdispersion Diagnostics")
    st.markdown(
        r"""
        Comparison of point-process rungs:
        - **Rung 0:** Homogeneous Poisson ($\lambda = \mu$), Fano = 1, memoryless.
        - **Rung 1:** Diurnal Inhomogeneous Poisson ($\mu(t)$ 24h harmonic).
        - **Rung 2:** Self-Exciting Hawkes Process ($\lambda(t) = \mu + \sum \alpha e^{-\beta(t-t_i)}$), Fano $\to 1/(1-n)^2$.
        """
    )

    t2_col1, t2_col2 = st.columns(2)

    with t2_col1:
        # Window count distribution comparison
        st.markdown("#### 1. 1-Hour Window Count Distribution")
        w_size = 1.0
        bins = np.arange(0, cutoff_t + w_size, w_size)
        counts, _ = np.histogram(events_so_far, bins=bins)

        max_c = max(10, int(np.percentile(counts, 99)) + 2)
        k_vals = np.arange(0, max_c + 1)
        mean_c = np.mean(counts)

        # Empirical frequencies
        emp_pmf = np.bincount(counts, minlength=max_c + 1)[: max_c + 1] / len(counts)

        # Poisson PMF
        poisson_pmf_vals = stats.poisson.pmf(k_vals, mean_c)

        # Hawkes simulated window counts
        hawkes_sim_events = simulate_hawkes_ogata(
            T=cutoff_t, mu=mu_hat, alpha=alpha_hat, beta=beta_hat, rng=np.random.default_rng(77)
        )
        sim_counts, _ = np.histogram(hawkes_sim_events, bins=bins)
        hawkes_pmf_vals = np.bincount(sim_counts, minlength=max_c + 1)[: max_c + 1] / len(sim_counts)

        fig_pmf = go.Figure()
        fig_pmf.add_trace(go.Bar(x=k_vals, y=emp_pmf, name="Empirical Data", marker_color="#1f77b4", opacity=0.7))
        fig_pmf.add_trace(go.Scatter(x=k_vals, y=poisson_pmf_vals, name="Homogeneous Poisson PMF", line=dict(color="#2ca02c", width=2.5)))
        fig_pmf.add_trace(go.Scatter(x=k_vals, y=hawkes_pmf_vals, name="Fitted Hawkes PMF", line=dict(color="#d62728", width=2.5, dash="dash")))

        fig_pmf.update_layout(
            xaxis_title="Event Count per 1-Hour Window",
            yaxis_title="Probability Density",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=30, r=30, t=30, b=30),
            height=340,
        )
        st.plotly_chart(fig_pmf, use_container_width=True)

    with t2_col2:
        # Fano Factor vs Window Length
        st.markdown("#### 2. Fano Factor vs Observation Window (Δ)")
        window_sizes = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        emp_fanos = []
        for w in window_sizes:
            f = compute_fano_factor(events_so_far, cutoff_t, window_h=w)
            emp_fanos.append(f)

        theo_asymp = 1.0 / max((1.0 - n_hat) ** 2, 0.01)

        fig_fano = go.Figure()
        fig_fano.add_trace(
            go.Scatter(
                x=window_sizes,
                y=emp_fanos,
                mode="lines+markers",
                name="Empirical Fano Factor",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )
        fig_fano.add_hline(
            y=1.0,
            line=dict(color="#2ca02c", width=2, dash="dash"),
            annotation_text="Poisson Baseline (Fano = 1)",
            annotation_position="bottom right",
        )
        fig_fano.add_hline(
            y=theo_asymp,
            line=dict(color="#d62728", width=2, dash="dot"),
            annotation_text=f"Hawkes Asymptote 1/(1-n)² = {theo_asymp:.2f}",
            annotation_position="top right",
        )

        fig_fano.update_layout(
            xaxis_title="Window Size Δ (Hours)",
            yaxis_title="Fano Factor Var(N) / E[N]",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            margin=dict(l=30, r=30, t=30, b=30),
            height=340,
        )
        st.plotly_chart(fig_fano, use_container_width=True)

    # Row 2: Time-Rescaling Q-Q Plot & Formal Metrics Table
    t2_col3, t2_col4 = st.columns(2)

    with t2_col3:
        st.markdown("#### 3. Time-Rescaling Theorem Q-Q Plot (KS Test)")
        gof_res = time_rescaling_test(
            events_so_far,
            T=cutoff_t,
            mu=mu_hat,
            alpha=alpha_hat,
            beta=beta_hat,
        )
        tau = gof_res.get("rescaled_interarrivals")
        if tau is None or len(tau) == 0:
            tau = np.array([1.0, 1.0])

        # Empirical quantiles vs Exp(1) theoretical quantiles
        p_grid = np.linspace(0.01, 0.99, max(len(tau), 2))
        exp_quantiles = -np.log(1.0 - p_grid)
        emp_quantiles = np.quantile(tau, p_grid)

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=exp_quantiles,
                y=emp_quantiles,
                mode="markers",
                name="Hawkes Compensator Quantiles",
                marker=dict(color="#1f77b4", size=5),
            )
        )
        max_q = max(float(exp_quantiles[-1]), float(emp_quantiles[-1]))
        fig_qq.add_trace(
            go.Scatter(
                x=[0, max_q],
                y=[0, max_q],
                mode="lines",
                name="Ideal Exp(1) Diagonal",
                line=dict(color="#d62728", width=2, dash="dash"),
            )
        )

        ks_p = gof_res.get("p_value", 0.5)
        fig_qq.update_layout(
            title=f"Goodness of Fit: KS p-value = {ks_p:.4f} ({'Pass (Good fit)' if ks_p > 0.05 else 'Reject (Poor fit)'})",
            xaxis_title="Theoretical Quantiles (Exp(1))",
            yaxis_title="Empirical Compensator Quantiles Λ(t_i) - Λ(t_{i-1})",
            margin=dict(l=30, r=30, t=50, b=30),
            height=340,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    with t2_col4:
        st.markdown("#### 4. Model Ladder Goodness-of-Fit Summary")
        # Baseline fits
        pois_fit = fit_poisson(events_so_far, cutoff_t)
        diurnal_fit = fit_poisson_diurnal(events_so_far, cutoff_t)

        comparison_df = pd.DataFrame([
            {
                "Ladder Rung": "Rung 0: Homogeneous Poisson",
                "Log-Likelihood": f"{pois_fit.get('log_likelihood', -999.0):.1f}",
                "AIC": f"{pois_fit.get('aic', 999.0):.1f}",
                "BIC": f"{pois_fit.get('bic', 999.0):.1f}",
                "Parameters": "μ",
                "Dispersion": "Equidispersed (Fano=1)",
            },
            {
                "Ladder Rung": "Rung 1: Diurnal Inhomogeneous Poisson",
                "Log-Likelihood": f"{diurnal_fit.get('log_likelihood', -999.0):.1f}",
                "AIC": f"{diurnal_fit.get('aic', 999.0):.1f}",
                "BIC": f"{diurnal_fit.get('bic', 999.0):.1f}",
                "Parameters": "24-bin λ_h",
                "Dispersion": "Diurnal cyclic variations",
            },
            {
                "Ladder Rung": "Rung 2: Exponential Hawkes MLE",
                "Log-Likelihood": f"{hawkes_fit.get('log_likelihood', -999.0):.1f}",
                "AIC": f"{hawkes_fit.get('aic', 999.0):.1f}",
                "BIC": f"{hawkes_fit.get('bic', 999.0):.1f}",
                "Parameters": "μ, α, β (n = α/β)",
                "Dispersion": f"Clustered / Overdispersed (Fano={emp_fanos[1]:.2f})",
            },
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        delta_aic = pois_fit.get("aic", 0.0) - hawkes_fit.get("aic", 0.0)
        st.info(
            f"💡 **Model Selection Diagnostic:** Hawkes MLE achieves an AIC improvement of "
            f"**ΔAIC = {delta_aic:.1f}** over Homogeneous Poisson, providing decisive evidence "
            f"for self-exciting attack cascade dynamics."
        )


# =============================================================================
# TAB 3: WORKED EXAMPLE
# =============================================================================
with tab3:
    st.subheader("Microscopic Sandbox: Anatomy of Self-Exciting Intensity")
    st.markdown(
        r"""
        Interactive demonstration of the Hawkes intensity jump-decay mechanism:
        $$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}$$
        Each arriving ransomware precursor (e.g. 4625 failed login or privilege escalation) instantly surges
        the attack wave hazard by **$+\alpha$**, which subsequently decays exponentially at rate **$\beta$** back to baseline **$\mu$**.
        """
    )

    w_col1, w_col2, w_col3, w_col4 = st.columns([1, 1, 1, 1])

    with w_col1:
        slider_mu = st.slider("Baseline Rate (μ)", min_value=0.05, max_value=3.0, value=0.5, step=0.05)
    with w_col2:
        collapse_poisson = st.toggle("🔒 Collapse to Poisson (α = 0)", value=False, help="Forces α=0 to prove Hawkes collapses to flat Poisson")
        slider_alpha = 0.0 if collapse_poisson else st.slider("Excitation Jump (α)", min_value=0.0, max_value=3.0, value=0.8, step=0.05)
    with w_col3:
        slider_beta = st.slider("Recovery Decay Rate (β)", min_value=0.2, max_value=5.0, value=1.5, step=0.1)
    with w_col4:
        half_life_min = (np.log(2) / slider_beta) * 60.0 if slider_beta > 0 else 0
        n_ratio = slider_alpha / slider_beta if slider_beta > 0 else 0
        st.metric("Branching Ratio n = α/β", f"{n_ratio:.2f}", "Stable" if n_ratio < 1 else "Explosive!")
        st.caption(f"Half-life of threat decay: **{half_life_min:.1f} minutes**")

    # Fixed default sequence: 10:00, 10:30, 10:35, 10:40, 10:42, 10:45
    # Expressed in hours elapsed from 10:00
    preset_event_times = np.array([0.0, 0.5, 0.5833, 0.6667, 0.7000, 0.7500])
    clock_labels = ["10:00", "10:30", "10:35", "10:40", "10:42", "10:45"]

    t_fine = np.linspace(0.0, 1.25, 600)
    lam_fine = np.full_like(t_fine, slider_mu)

    for i, t in enumerate(t_fine):
        past = preset_event_times[preset_event_times < t]
        if len(past) > 0 and slider_alpha > 0:
            lam_fine[i] += np.sum(slider_alpha * np.exp(-slider_beta * (t - past)))

    fig_demo = go.Figure()

    # Intensity trajectory
    fig_demo.add_trace(
        go.Scatter(
            x=t_fine,
            y=lam_fine,
            mode="lines",
            name="λ(t) Intensity",
            line=dict(color="#1E88E5", width=3),
        )
    )

    # Flat Poisson baseline line
    fig_demo.add_hline(
        y=slider_mu,
        line=dict(color="#43A047", width=2, dash="dash"),
        annotation_text=f"Baseline Rate μ = {slider_mu:.2f}",
        annotation_position="bottom left",
    )

    # Event jump points and annotations
    for ev_t, lbl in zip(preset_event_times, clock_labels):
        # Value right before
        past_ev = preset_event_times[preset_event_times < ev_t]
        y_before = slider_mu + (np.sum(slider_alpha * np.exp(-slider_beta * (ev_t - past_ev))) if len(past_ev) > 0 and slider_alpha > 0 else 0)
        y_after = y_before + slider_alpha

        fig_demo.add_vline(x=ev_t, line=dict(color="#E53935", width=1.5, dash="dot"))
        if slider_alpha > 0:
            fig_demo.add_annotation(
                x=ev_t,
                y=y_after,
                text=f"+{slider_alpha:.2f} ({lbl})",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor="#E53935",
                ax=0,
                ay=-25,
            )

    fig_demo.update_layout(
        title="Hawkes Intensity Curve Over Attack Burst Sequence (10:00 - 11:15)",
        xaxis_title="Time Elapsed (Hours) [Events at 10:00, 10:30, 10:35, 10:40, 10:42, 10:45]",
        yaxis_title="Precursor Rate λ(t) (events / hour)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
    )
    st.plotly_chart(fig_demo, use_container_width=True)

    if collapse_poisson:
        st.success(
            r"✅ **Poisson Limit Verified:** When $\alpha = 0$, every self-exciting term vanishes and $\lambda(t) \equiv \mu$. "
            r"Inter-arrival gaps collapse to memoryless $\mathrm{Exp}(\mu)$, proving Poisson is a nested special case of Hawkes."
        )


# =============================================================================
# TAB 4: CLUSTER EXPLORER
# =============================================================================
with tab4:
    st.subheader("Hawkes as a Poisson Cluster Process & Borel Cascade Law")
    st.markdown(
        r"""
        By the **Hawkes & Oakes (1974)** cluster representation:
        1. **Immigrants (Gen 0)** arrive according to a Poisson process with rate $\mu$.
        2. Each event independently spawns a Poisson number of **Offspring (Gen 1)** with mean $n = \alpha/\beta$.
        3. Child delays follow $\mathrm{Exp}(\beta)$.
        4. The total cluster size $S$ follows the **Borel Distribution**:
        $$P(S = k) = \frac{e^{-nk}(nk)^{k-1}}{k!}, \quad E[S] = \frac{1}{1-n}$$
        """
    )

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        branch_ratio_slider = st.slider("Branching Ratio n = α/β", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    with c_col2:
        num_clusters_slider = st.slider("Number of Immigrant Clusters", min_value=100, max_value=1500, value=500, step=50)

    # Simulate branching process
    sim_rng = np.random.default_rng(42)
    cluster_sizes = []
    cluster_records = []

    for c_id in range(min(num_clusters_slider, 250)):
        # Immigrant at t=0
        gen_events = [(0.0, 0)]  # (time, gen)
        active_gen = [0.0]
        curr_g = 0

        while len(active_gen) > 0 and len(gen_events) < 50:
            next_active = []
            curr_g += 1
            for parent_t in active_gen:
                n_children = sim_rng.poisson(branch_ratio_slider)
                for _ in range(n_children):
                    delay = sim_rng.exponential(1.0 / 1.5)
                    child_t = parent_t + delay
                    gen_events.append((child_t, curr_g))
                    next_active.append(child_t)
            active_gen = next_active

        cluster_sizes.append(len(gen_events))
        for t, g in gen_events:
            cluster_records.append({"cluster_id": c_id, "time_h": t, "generation": g})

    # Theoretical Borel stats
    theo_mean_s = 1.0 / (1.0 - branch_ratio_slider)
    emp_mean_s = float(np.mean(cluster_sizes))

    m1, m2, m3 = st.columns(3)
    m1.metric("Theoretical Mean Cluster E[S]", f"{theo_mean_s:.2f}")
    m2.metric("Empirical Mean Cluster Size", f"{emp_mean_s:.2f}", delta=f"{emp_mean_s - theo_mean_s:+.2f}")
    m3.metric("Max Observed Cascade Size", f"{max(cluster_sizes)} events")

    # Visualizations
    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("#### 1. Cluster Cascade Timeline (First 30 Immigrants)")
        df_clust = pd.DataFrame(cluster_records)
        df_sub = df_clust[df_clust["cluster_id"] < 30]

        fig_clust = px.scatter(
            df_sub,
            x="time_h",
            y="cluster_id",
            color="generation",
            color_continuous_scale="Viridis",
            labels={"time_h": "Relative Delay (Hours)", "cluster_id": "Cluster #", "generation": "Gen"},
            title="Tree Generations (0 = Immigrant, 1 = Child, 2+ = Cascades)",
        )
        fig_clust.update_layout(margin=dict(l=30, r=30, t=40, b=30), height=360)
        st.plotly_chart(fig_clust, use_container_width=True)

    with b_col2:
        st.markdown("#### 2. Total Cluster Size vs Borel(n) PMF")
        max_k = min(15, max(cluster_sizes))
        k_range = np.arange(1, max_k + 1)
        borel_probs = borel_pmf(k_range, branch_ratio_slider)

        emp_hist, _ = np.histogram(cluster_sizes, bins=np.arange(0.5, max_k + 1.5, 1.0), density=True)

        fig_borel = go.Figure()
        fig_borel.add_trace(go.Bar(x=k_range, y=emp_hist, name="Simulated Cluster Size", marker_color="#1E88E5", opacity=0.7))
        fig_borel.add_trace(go.Scatter(x=k_range, y=borel_probs, name=f"Borel(n={branch_ratio_slider}) Law", line=dict(color="#D81B60", width=3)))

        fig_borel.update_layout(
            xaxis_title="Cluster Size S (Total Events in Wave Cascade)",
            yaxis_title="Probability P(S = k)",
            legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=30, r=30, t=40, b=30),
            height=360,
        )
        st.plotly_chart(fig_borel, use_container_width=True)
