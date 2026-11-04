"""Launch the base and instruct workers and serve the demo page.

    uv run serve.py            # then open http://127.0.0.1:8800
"""

import json
import os
import signal
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
PORT = int(os.environ.get("PORT", 8800))

PANES = [
    {
        "name": "gemma-4-26b-a4b",
        "model": "mlx-community/gemma-4-26b-a4b-4bit",
        "mode": "base",
        "port": PORT + 1,
    },
    {
        "name": "gemma-4-26b-a4b-it",
        "model": os.environ.get(
            "INSTRUCT_MODEL",
            str(Path.home() / ".lmstudio/models/mlx-community/gemma-4-26b-a4b-it-4bit"),
        ),
        "mode": "chat",
        "port": PORT + 2,
    },
]
if not Path(PANES[1]["model"]).exists():
    PANES[1]["model"] = "mlx-community/gemma-4-26b-a4b-it-4bit"

CONFIG = json.dumps(
    [{"name": p["name"], "port": p["port"], "mode": p["mode"]} for p in PANES]
).encode()


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        if self.path == "/config.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(CONFIG)))
            self.end_headers()
            self.wfile.write(CONFIG)
            return
        super().do_GET()


server = ThreadingHTTPServer(
    ("127.0.0.1", PORT), partial(Handler, directory=str(HERE / "static"))
)

procs = [
    subprocess.Popen(
        [
            sys.executable,
            str(HERE / "worker.py"),
            "--model", p["model"],
            "--mode", p["mode"],
            "--port", str(p["port"]),
        ]
    )
    for p in PANES
]


def shutdown(*_):
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

print(f"page on http://127.0.0.1:{PORT} (models load in the background)", flush=True)
try:
    server.serve_forever()
finally:
    shutdown()
