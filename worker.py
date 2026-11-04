"""Serve one MLX model over HTTP, streaming tokens as server-sent events.

POST /generate {"prompt": "..."} streams `data: {"text": "..."}` events, then
`data: {"done": true, ...}`. In "base" mode the prompt is fed as a raw
document; in "chat" mode it is wrapped as a single user turn with the model's
chat template. A new request preempts any generation already in flight.

Sampling uses the model's own generation_config.json, i.e. the defaults the
model's publisher ships with the weights.
"""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="HF repo id or local path")
parser.add_argument("--mode", choices=["base", "chat"], required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

model_path = Path(args.model)
if not model_path.exists():
    model_path = Path(snapshot_download(args.model))

gen_config = json.loads((model_path / "generation_config.json").read_text())
SAMPLING = {
    "temperature": float(gen_config.get("temperature", 1.0)),
    "top_p": float(gen_config.get("top_p", 1.0)),
    "top_k": int(gen_config.get("top_k", 0)),
}

model = tokenizer = None

# All MLX work runs on one generation thread. Handlers enqueue jobs and read
# events back from a per-request queue. A newer request bumps `latest`, so the
# running job exits at its next token.
jobs: "queue.Queue[tuple[int, str, int, queue.Queue]]" = queue.Queue()
latest = 0
latest_lock = threading.Lock()


def encode(prompt: str) -> list[int]:
    if args.mode == "base":
        return tokenizer.encode(prompt)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def generate(prompt: str, max_tokens: int, stream):
    sampler = make_sampler(
        temp=SAMPLING["temperature"], top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"]
    )
    yield from stream_generate(
        model,
        tokenizer,
        encode(prompt),
        max_tokens=max_tokens,
        sampler=sampler,
        stream=stream,
    )


def generation_loop():
    global model, tokenizer
    model, tokenizer = load(str(model_path))
    stream = mx.default_stream(mx.default_device())
    for _ in generate("Hello", 4, stream):
        pass
    ready.set()
    while True:
        job_id, prompt, max_tokens, out = jobs.get()
        if job_id != latest:
            out.put(None)
            continue
        start = time.perf_counter()
        n, finish = 0, "length"
        try:
            for resp in generate(prompt, max_tokens, stream):
                if job_id != latest:
                    finish = "preempted"
                    break
                n = resp.generation_tokens
                if resp.text:
                    out.put({"text": resp.text})
                if resp.finish_reason:
                    finish = resp.finish_reason
        except Exception as e:
            finish = f"error: {e}"
        out.put(
            {
                "done": True,
                "finish": finish,
                "tokens": n,
                "seconds": round(time.perf_counter() - start, 2),
            }
        )
        out.put(None)
        mx.clear_cache()


ready = threading.Event()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        body = json.dumps({"ok": True, "sampling": SAMPLING}).encode()
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def event(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        global latest
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        prompt = req.get("prompt", "")
        max_tokens = min(int(req.get("max_tokens", args.max_tokens)), 4096)

        with latest_lock:
            latest += 1
            my_id = latest

        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        out: queue.Queue = queue.Queue()
        jobs.put((my_id, prompt, max_tokens, out))
        try:
            while (item := out.get()) is not None:
                self.event(item)
        except (BrokenPipeError, ConnectionResetError):
            with latest_lock:
                if latest == my_id:
                    latest += 1


threading.Thread(target=generation_loop, daemon=True).start()
ready.wait()
server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
print(f"ready {args.mode} {model_path.name} on :{args.port} {SAMPLING}", flush=True)
server.serve_forever()
