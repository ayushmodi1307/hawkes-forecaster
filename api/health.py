from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({
            "status": "healthy",
            "project": "hawkes_attack_forecaster",
            "version": "0.1.0",
            "runtime": "Vercel Serverless Python",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

def app(environ, start_response):
    body = json.dumps({
        "status": "healthy",
        "project": "hawkes_attack_forecaster",
        "version": "0.1.0",
    }).encode("utf-8")
    headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body))), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    return [body]
