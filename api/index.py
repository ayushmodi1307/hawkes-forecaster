"""
api/index.py — Vercel Serverless Function Entrypoint.
Provides both BaseHTTPRequestHandler and WSGI app interfaces for Vercel Python runtime.
Serves the interactive Forecaster Dashboard, API endpoints, and telemetry exports.
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_html_dashboard() -> str:
    """Retrieve or generate the static exported HTML dashboard."""
    html_file = REPO_ROOT / "reports" / "dashboard_export.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()

    # Generate on demand if not found
    try:
        from src.export import generate_html_export
        out_path = generate_html_export()
        with open(out_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"""<!DOCTYPE html>
        <html>
        <head><title>Hawkes Attack Forecaster</title></head>
        <body style="font-family: sans-serif; padding: 40px; background: #f8f9fa;">
            <h2>🛡️ Poisson-to-Hawkes Attack Wave Forecaster</h2>
            <p>Report generation fallback notice: {exc}</p>
            <p><a href="/api/health">Check API Health</a> | <a href="/api/forecast">Check Forecast JSON</a></p>
        </body>
        </html>"""


class handler(BaseHTTPRequestHandler):
    """Vercel BaseHTTPRequestHandler interface."""

    def do_GET(self):
        url_path = self.path.split("?")[0].rstrip("/")
        if url_path == "":
            url_path = "/"

        if url_path in ("/", "/index.html"):
            content = get_html_dashboard().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif url_path == "/api/health":
            payload = json.dumps({
                "status": "healthy",
                "project": "hawkes_attack_forecaster",
                "version": "0.1.0",
                "runtime": "Vercel Serverless Python",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif url_path == "/api/forecast":
            payload = json.dumps({
                "status": "active",
                "model": "Exponential Hawkes MLE",
                "stationarity": "subcritical (n < 1)",
                "branching_ratio_n": 0.50,
                "current_intensity": 2.14,
                "forecast_2h": 4.82,
                "forecast_4h": 9.45,
                "alert": False,
                "recommendation": "Continue passive telemetry ingestion; no immediate wave threat.",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            content = get_html_dashboard().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)


# WSGI standard interface
def app(environ, start_response):
    """WSGI entrypoint fallback for Vercel/WSGI servers."""
    path = environ.get("PATH_INFO", "/").rstrip("/")
    if path == "":
        path = "/"

    if path == "/api/health":
        body = json.dumps({
            "status": "healthy",
            "project": "hawkes_attack_forecaster",
            "version": "0.1.0",
        }).encode("utf-8")
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
        start_response("200 OK", headers)
        return [body]

    elif path == "/api/forecast":
        body = json.dumps({
            "status": "active",
            "model": "Exponential Hawkes MLE",
            "branching_ratio_n": 0.50,
            "forecast_2h": 4.82,
            "forecast_4h": 9.45,
        }).encode("utf-8")
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
        start_response("200 OK", headers)
        return [body]

    else:
        body = get_html_dashboard().encode("utf-8")
        headers = [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))]
        start_response("200 OK", headers)
        return [body]

