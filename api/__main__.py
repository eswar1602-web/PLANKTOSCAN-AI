"""Run the API (and the built frontend):  python -m api [--host 127.0.0.1] [--port 8000] [--reload]"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="PlanktoScan AI server: REST API + the built React frontend.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    a = ap.parse_args()
    print(f"PlanktoScan AI: http://{a.host}:{a.port}  (API docs at /docs; models load at start-up, about 20 s)")
    uvicorn.run("api.main:create_app", factory=True, host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
