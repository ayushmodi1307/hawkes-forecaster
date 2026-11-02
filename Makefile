.PHONY: setup generate parse test backtest dashboard demo clean

PYTHON ?= python
PIP    ?= pip

# ── Install dependencies ──────────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt

# ── Generate synthetic data ───────────────────────────────────────────────────
generate:
	$(PYTHON) -c "from src.simulate import run_full_scenario; run_full_scenario()"

# ── Parse raw logs ────────────────────────────────────────────────────────────
parse:
	$(PYTHON) -c "from src.parser import parse_logs; import yaml; cfg=yaml.safe_load(open('config.yaml')); parse_logs(cfg['paths']['raw_data'])"

# ── Run the full test suite ───────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Run backtesting and produce reports/backtest_results.md ──────────────────
backtest:
	$(PYTHON) -m src.backtest

# ── Launch the Streamlit dashboard ───────────────────────────────────────────
dashboard:
	streamlit run app/dashboard.py

# ── Full demo pipeline ────────────────────────────────────────────────────────
demo: generate parse backtest

# ── Export static HTML report ─────────────────────────────────────────────────
export:
	$(PYTHON) -m src.export

# ── Launch local HTTP export server (port 8080) ───────────────────────────────
serve-export: export
	$(PYTHON) -m http.server 8080 --directory reports

# ── Clean generated artefacts ────────────────────────────────────────────────
clean:
	-$(PYTHON) -c "import shutil, glob; [shutil.rmtree(d, ignore_errors=True) for d in ['data/raw', 'data/processed', 'reports']]; import os; os.makedirs('data/raw', exist_ok=True); os.makedirs('data/processed', exist_ok=True); os.makedirs('reports', exist_ok=True)"
	-$(PYTHON) -c "import os; os.path.exists('alerts.log') and os.remove('alerts.log')"
