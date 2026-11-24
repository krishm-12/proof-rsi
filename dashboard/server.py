"""Local read-only view of proof RSI run artifacts."""
import argparse
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
from threading import Lock
import time
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
GPU_LOCK = Lock()
TELEMETRY = True
gpu_reading = (float("-inf"), None)


@lru_cache(maxsize=48)
def load(path, modified, size):
    try:
        with path.open(encoding="utf-8") as source:
            if path.suffix == ".jsonl":
                rows = []
                for line in source:
                    if not line.endswith("\n"):
                        break
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
                return rows
            value = json.load(source)
            return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def artifact(path):
    try:
        if path.is_symlink():
            return None
        stamp = path.stat()
        return load(path, stamp.st_mtime_ns, stamp.st_size)
    except OSError:
        return None


def inventory():
    runs = []
    for folder in RUNS.iterdir() if RUNS.is_dir() else ():
        if folder.name.startswith(".") or not folder.is_dir() or folder.is_symlink():
            continue
        try:
            metrics_path = folder / "metrics.json"
            source = metrics_path if metrics_path.is_file() else folder / "attempts.jsonl"
            updated = source.stat().st_mtime
        except OSError:
            continue
        metrics = artifact(metrics_path) or {}
        runs.append(dict(id=folder.name, status=metrics.get("status"), updated=updated))
    return sorted(runs, key=lambda run: run["updated"], reverse=True)


def gpu_snapshot():
    global gpu_reading
    with GPU_LOCK:
        now = time.monotonic()
        if now - gpu_reading[0] < 4:
            return gpu_reading[1]
        reading = None
        command = shutil.which("nvidia-smi")
        if command:
            keys = ("utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw")
            try:
                output = subprocess.check_output(
                    [command, "--id=0", "--query-gpu=name," + ",".join(keys),
                     "--format=csv,noheader,nounits"],
                    timeout=2, stderr=subprocess.DEVNULL, text=True,
                )
                name, *values = output.strip().splitlines()[0].split(",")
                fields = {}
                for key, value in zip(keys, values):
                    try:
                        fields[key] = float(value.strip())
                    except ValueError:
                        fields[key] = None
                reading = dict(name=name.strip(), sampled=time.time(), fields=fields)
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        gpu_reading = now, reading
        return reading


def snapshot(run_id):
    runs = inventory()
    selected = next((run for run in runs if run["id"] == run_id), None) if run_id else None
    selected = selected or (runs[0] if runs else None)
    result = dict(runs=runs, selected=selected, metrics={}, attempts=[], loss=[], gpu=gpu_snapshot() if TELEMETRY else None)
    if selected:
        folder = RUNS / selected["id"]
        result.update(
            metrics=artifact(folder / "metrics.json") or {},
            attempts=artifact(folder / "attempts.jsonl") or [],
            loss=artifact(folder / "training_loss.jsonl") or [],
        )
    return result


class Handler(SimpleHTTPRequestHandler):
    STATIC_PATHS = frozenset(("/", "/index.html", "/styles.css", "/app.js"))

    def parse_request(self):
        if not super().parse_request():
            return False
        port = self.server.server_port
        allowed = {"localhost", "127.0.0.1", f"localhost:{port}", f"127.0.0.1:{port}"}
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in allowed:
            self.send_error(403, "Dashboard requires a loopback Host")
            return False
        return True

    def send_head(self):
        if urlsplit(self.path).path not in self.STATIC_PATHS:
            self.send_error(404)
            return None
        return super().send_head()

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/api/snapshot":
            run_id = parse_qs(url.query).get("run", [None])[0]
            body = json.dumps(snapshot(run_id), ensure_ascii=False, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            super().do_GET()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE / "dist"), **kwargs)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    global RUNS, TELEMETRY
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--no-telemetry", action="store_true")
    args = parser.parse_args()
    RUNS = args.runs_dir.resolve()
    TELEMETRY = not args.no_telemetry
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"RSI — Recursive Intelligence Lab · http://localhost:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
