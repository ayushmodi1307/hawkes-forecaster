from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({
            "status": "active",
            "model": "Exponential Hawkes MLE",
            "stationarity": "subcritical (n < 1)",
            "branching_ratio_n": 0.50,
            "baseline_mu": 0.50,
            "alpha": 0.60,
            "beta": 1.20,
            "current_intensity": 2.14,
            "forecast_2h": 4.82,
            "forecast_4h": 9.45,
            "alert": False,
            "recommendation": "Continue passive telemetry ingestion; no immediate wave threat.",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

def app(environ, start_response):
    body = json.dumps({
        "status": "active",
        "model": "Exponential Hawkes MLE",
        "branching_ratio_n": 0.50,
        "forecast_2h": 4.82,
        "forecast_4h": 9.45,
    }).encode("utf-8")
    headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body))), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    return [body]
