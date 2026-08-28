"""In-sandbox HTTP exec server: the data-plane replacement for `eai job exec`.

Runs INSIDE a sandbox job (started by the job command, see backends/eai.py)
and executes shell commands on behalf of the trainer over pod-to-pod HTTP,
so per-command traffic never touches the EAI control plane — the component
behind every run-killing incident on this cluster (EAI_AUDIT.md).

Constraints that shape this file:
- Must run on WHATEVER Python the task image has (R2E testbeds go back to
  ~3.6): stdlib only, no f-string-adjacent newer syntax beyond 3.6, no
  dataclasses, ThreadingMixIn instead of ThreadingHTTPServer (3.7+).
- One server per sandbox; bearer-token auth (RLLM_EXEC_TOKEN env) because
  the pod network is flat within the cluster.
- Endpoints:
    GET  /health                          -> {"ok": true}
    POST /exec  {"cmd": str, "timeout": float|null}
        -> {"returncode": int, "stdout": str, "stderr": str}
  A server-side timeout kills the process group and returns returncode 124
  (mirroring coreutils `timeout`, which the CLI exec path uses).
"""

import hmac
import json
import os
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

PORT = int(os.environ.get("RLLM_EXEC_PORT", "8642"))
TOKEN = os.environ.get("RLLM_EXEC_TOKEN", "")
MAX_OUTPUT = 8 * 1024 * 1024  # per stream; CLI tails were capped anyway
DEFAULT_TIMEOUT = float(os.environ.get("RLLM_EXEC_DEFAULT_TIMEOUT", "900"))


def _run(cmd, timeout):
    # start_new_session so a timeout can kill the whole process tree, not
    # just the shell (agents spawn long pipelines).
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
        rc = 124
        err = (err or b"") + b"\n[exec_server] timeout after %ds, process group killed" % int(timeout)
    return rc, (out or b"")[:MAX_OUTPUT], (err or b"")[:MAX_OUTPUT]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        got = self.headers.get("X-RLLM-Token", "")
        return TOKEN and hmac.compare_digest(got, TOKEN)

    def do_GET(self):
        if self.path == "/health":
            # health is unauthenticated: it leaks nothing and the client
            # uses it to decide CLI-fallback before it would send a token.
            self._reply(200, {"ok": True, "pid": os.getpid(), "python": sys.version.split()[0]})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            self._reply(403, {"error": "bad token"})
            return
        if self.path != "/exec":
            self._reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            cmd = req["cmd"]
            timeout = req.get("timeout") or DEFAULT_TIMEOUT
        except (ValueError, KeyError) as exc:
            self._reply(400, {"error": "bad request: %s" % exc})
            return
        try:
            rc, out, err = _run(cmd, float(timeout))
        except Exception as exc:  # defensive: a handler crash kills the request, not the server
            self._reply(500, {"error": "exec failed: %s" % exc})
            return
        self._reply(200, {
            "returncode": rc,
            "stdout": out.decode("utf-8", "replace"),
            "stderr": err.decode("utf-8", "replace"),
        })

    def log_message(self, fmt, *args):  # quiet: one line per exec is enough
        sys.stderr.write("[exec_server] %s\n" % (fmt % args))


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    if not TOKEN:
        sys.stderr.write("[exec_server] FATAL: RLLM_EXEC_TOKEN not set\n")
        sys.exit(1)
    srv = Server(("0.0.0.0", PORT), Handler)
    sys.stderr.write("[exec_server] listening on :%d (python %s)\n" % (PORT, sys.version.split()[0]))
    srv.serve_forever()


if __name__ == "__main__":
    main()
