# Poisson-to-Hawkes Attack Wave Forecaster

A **defensive, advisory-only** SOC tool that ingests SIEM-style logs, extracts
ransomware-precursor events, and forecasts attack-burst intensity 2 h and 4 h ahead
using a ladder of point-process models from homogeneous Poisson up to a self-exciting
Hawkes process.

> **Advisory only.** The tool never executes containment. All network calls require an
> explicit webhook URL in `config.yaml`.

---

## Quick Start

```bash
pip install -r requirements.txt      # or: make setup
make demo                             # generate data → parse → backtest
make dashboard                        # launch Streamlit app on localhost:8501
make test                             # run full pytest suite
```

---

## Usage

| Command | Description |
|---------|-------------|
| `make setup` | Install Python dependencies |
| `make generate` | Simulate synthetic SIEM logs |
| `make parse` | Parse raw logs → `data/processed/events.csv` |
| `make backtest` | Walk-forward backtest → `reports/backtest_results.md` |
| `make dashboard` | Launch 4-tab Streamlit dashboard |
| `make test` | Run all pytest tests |
| `make demo` | Full end-to-end pipeline |

---

## THEORY SECTION — Poisson-to-Hawkes Derivations

<!-- PLACEHOLDER — expanded below -->

### 1. Homogeneous Poisson as Hawkes with α = 0

Setting α = 0 in the Hawkes intensity λ(t) = μ + Σ α exp(−β(t−tᵢ)) collapses to
the constant λ(t) = μ. Window counts ~ Poisson(μΔ), inter-arrivals ~ Exp(μ),
CV = 1, Fano = 1.

### 2. Hawkes as a Poisson Cluster Process

Immigrants arrive ~ Poisson(μ); each event spawns Poisson(n) children with n = α/β,
child delays ~ Exp(β). Total cluster size S follows the Borel distribution:

  P(S = k) = exp(−nk) · (nk)^(k−1) / k!,  mean E[S] = 1/(1−n).

### 3. Stationary Mean Rate and Fano Factor

  E[λ] = μ/(1−n),   Fano(Δ) → 1/(1−n)²  as Δ → ∞.

Fano > 1 signals overdispersion relative to Poisson — the key diagnostic.

### 4. Time-Rescaling Theorem

Under the correct model, compensator increments
Λ(tᵢ) − Λ(tᵢ₋₁) ~ i.i.d. Exp(1).
A KS test on these rescaled gaps is the formal goodness-of-fit test used here.

### 5. Ogata Thinning

Simulate from a dominating Poisson(λ_max); accept candidate t with probability
λ(t)/λ_max. This is the simulation method in src/simulate.py.

### 6. Conditional Forecast

  E[λ(t+s) | H_t] = η + (λ(t) − η) · exp(−(β−α)s),   η = μβ/(β−α).

  E[N(t, t+Δ) | H_t] = ηΔ + (λ(t)−η)/(β−α) · (1 − exp(−(β−α)Δ)).

Naive Poisson tail using this mean understates burst risk; Monte Carlo (K=1000)
continuations are used to quantify the gap.

---

## Architecture

```
src/simulate.py        — Poisson, Hawkes, scenario generators, SIEM emitter
src/parser.py          — Multi-format log ingestion → events.csv
src/poisson_baseline.py — Rung 0/1 Poisson baselines, EWMA, sliding window
src/model.py           — Hawkes MLE (O(n) recursion, analytic gradients, L-BFGS-B)
src/analysis.py        — Clustering diagnostics, bootstrap LRT, time-rescaling
src/forecast.py        — Closed-form + Monte Carlo forecasting
src/alerting.py        — Threshold calibration, alert payload, send_alert
src/backtest.py        — Walk-forward backtest, 60/20/20 split, reports
app/dashboard.py       — Streamlit 4-tab dashboard
```

---

## Limitations

1. **False positives are possible.** Alert thresholds are calibrated on synthetic
   validation data; real environments require environment-specific tuning.
2. **Requires quality timestamped event data.** Noisy or incomplete logs degrade
   model accuracy significantly.
3. **Parameters drift over time.** Hawkes parameters (μ, α, β) must be periodically
   re-estimated as attacker behaviour and network topology evolve.
4. **Forecasts timing and intensity only.** The model does not identify attack type,
   attacker identity, or specific targets. It is not a threat-intelligence tool.
5. **Synthetic-data results do not guarantee real-world performance.** Backtests are
   conducted on simulated logs; real SIEM data may exhibit different statistical
   properties.
6. **Advisory only — no automated containment.** All recommended actions are
   suggestions for human analysts. The tool never modifies systems, blocks traffic, or
   revokes credentials.
