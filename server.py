"""
server.py — Python Serverless entrypoint for Vercel and local WSGI/HTTP execution.
"""
from api.index import app, handler

if __name__ == "__main__":
    from http.server import HTTPServer
    port = 8080
    print(f"Starting Hawkes Forecaster server on http://localhost:{port}")
    server = HTTPServer(("0.0.0.0", port), handler)
    server.serve_forever()

