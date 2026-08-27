"""EAI Toolkit (yul201) sandbox backend.

Runs each sandbox as an EAI batch job executing ``sleep infinity`` on the
task's Docker image, driven through the ``eai`` CLI. Designed for clusters
where jobs are the only container primitive (no Docker daemon, no k8s API).

Cluster specifics this backend encodes (see rollout-control/EAI_AUDIT.md and
SANDBOX_VALIDATION.md in the project repo):

- Jobs run as a fixed non-root uid with no capabilities; the ``user`` argument
  to :meth:`exec` is therefore ignored (protocol permits this).
- Docker-Hub task images are re-pulled from the cluster-internal registry,
  where they carry a ``root/`` 0755 fix layer (R2E images ship ``/root`` as
  0700, hiding the uv interpreters from non-root). The hub→internal mapping
  is :func:`map_image`; images are expected to be pre-mirrored with
  ``scripts/eai/mirror_task_image.sh``.
- Uploads go through a shared VAST directory mounted read-only into every
  sandbox job, then ``cp`` inside the sandbox — there is no tar/attach API.
- Repo working copies must be made under ``/tmp`` with ``PYTHONPATH`` set to
  the copy (read-only image paths + editable-install finders otherwise win);
  that policy lives in the R2E flow/harness layer, not here.

Auth: relies on the ambient ``eai`` CLI session (``eai login`` state in the
mounted home). All driver processes (interactive container, GPU jobs that
mount the home resource) inherit it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid

logger = logging.getLogger(__name__)

EAI_PROFILE = os.environ.get("RLLM_EAI_PROFILE", "yul201")
INTERNAL_REGISTRY = os.environ.get(
    "RLLM_EAI_REGISTRY", "registry.toolkit-sp.yul201.service-now.com/snow.research.adea"
)
# Host-side path of the shared transfer dir (driver writes here) and the
# fixed mount point inside sandbox jobs (sandbox reads from here).
TRANSFER_HOST_DIR = os.environ.get(
    "RLLM_EAI_TRANSFER_HOST", "/mnt/adea/data_rw/rollout_control/transfer"
)
TRANSFER_DATA_SPEC = os.environ.get(
    "RLLM_EAI_TRANSFER_DATA",
    "snow.research.adea.data/rollout_control/transfer:/mnt/rc_transfer:ro",
)
TRANSFER_REMOTE_DIR = "/mnt/rc_transfer"

_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}

# CLI failures worth retrying. Beyond control-plane 5xx, this includes
# "token is invalid": the eai session credentials rotate periodically, and
# every process invoking the CLI during that window gets a hard auth error
# even though the session is healthy seconds later (observed 2026-08-25
# 20:26 UTC — one rotation killed a training run and four eval jobs, because
# auth errors were classified non-transient and failed on the first try).
# A genuinely expired session still fails, just after the ladder runs out.
# Lowercase: every comparison lowercases the CLI output first, so the checks
# cannot silently stop matching if the CLI changes capitalization.
_TRANSIENT_ERRS = (
    "502", "503", "504", "http: 500", "server-side error", "internal error",
    "no response", "token is invalid", "context deadline exceeded",
)

# Network-layer failures raised by the CLI's Go HTTP client: DNS lookup
# failures, refused connections, dead routes. These are strictly *connectivity*
# problems -- the request never reached the API -- so retrying is safe and the
# operation definitionally did not take effect (unlike a 502, where the server
# may well have acted before failing to answer).
#
# Kept separate from _TRANSIENT_ERRS because the two are retried under
# different conditions: control-plane calls retry on either, while `job exec`
# retries a network error only on exit 7 (transport failure before the remote
# command ran) -- a *test* that legitimately prints "connection refused" and
# exits 1 must not be re-executed.
#
# Observed 2026-08-26 ~18:30-19:00 UTC: a control-plane outage produced
# `dial tcp ...: connect: connection refused` and DNS `server misbehaving`.
# Neither was on the transient list, so sandbox submission raised on the first
# attempt and killed ALL FOUR training arms mid-run (v2@53, v3@51, v3full@4,
# rejection@3) within 30 minutes of each other.
_NETWORK_ERRS = (
    "dial tcp", "connection refused", "connection reset", "no such host",
    "server misbehaving", "i/o timeout", "network is unreachable",
    "no route to host", "temporary failure in name resolution",
    "tls handshake timeout",
    # Go renders a truncated response as `Post "...": EOF`. Matched with the
    # punctuation so it cannot hit an unrelated word containing "eof".
    ": eof", "unexpected eof",
)


def _has_cli_marker(err: str) -> bool:
    """True if `err` looks like CLI output rather than remote command output.

    The transient lists contain bare status codes ("502") and short tokens
    ("eof"), and a *remote command*'s output can easily contain those (line
    numbers, test ids, hashes). Matching them without a marker would re-run
    side-effecting agent/verifier commands that had actually executed.
    """
    low = (err or "").lower()
    return "error:" in low or "http:" in low


def _is_transient(err: str) -> bool:
    """True if `err` is a CLI/control-plane HTTP failure worth retrying."""
    if not _has_cli_marker(err):
        return False
    low = (err or "").lower()
    return any(t in low for t in _TRANSIENT_ERRS)


def _is_network_error(err: str) -> bool:
    """True if `err` is a connectivity failure that never reached the API."""
    if not _has_cli_marker(err):
        return False
    low = (err or "").lower()
    return any(t in low for t in _NETWORK_ERRS)


def _is_retryable_control_plane(err: str) -> bool:
    """Retry predicate for control-plane calls (submit / kill / ls / info).

    Superset of `_is_transient`: these calls do not execute anything inside a
    sandbox, so riding out a network outage costs nothing but latency, whereas
    giving up costs an entire multi-hour training run.
    """
    return _is_transient(err) or _is_network_error(err)

# Submission retry ladder length. 10 attempts of min(300, 10*2^n) backoff is
# ~25 min of tolerance for a control-plane outage. Env-tunable so a run can be
# made more or less patient without a code change.
SUBMIT_ATTEMPTS = max(1, int(os.environ.get("RLLM_EAI_SUBMIT_ATTEMPTS", "10")))

# Ceiling on a single `job exec` when the caller passes no timeout of its own.
# Was hardcoded at 3600s, which makes ONE hung remote command block a rollout
# for an hour -- and with retry_limit=3 the rollout can hold its batch for
# three. Observed 2026-08-27: v3-core's step-60 validation sat at 63/64 for
# 45+ minutes on `Attempt 1/3 failed: TimeoutExpired(['eai','job','exec',...])`,
# stalling the arm on the critical path.
#
# 900s still allows a genuinely long verifier test suite (episodes complete in
# well under 15 min end to end) while cutting the worst-case straggler cost 4x.
# Env-tunable so a long-horizon experiment can raise it without a code change.
EXEC_TIMEOUT_S = float(os.environ.get("RLLM_EAI_EXEC_TIMEOUT_S", "900"))

# How much time `_wait_running` may spend BLIND (control plane unreachable)
# without it counting against the startup deadline. Bounded so a permanently
# unreachable API still fails the rollout rather than hanging it forever.
WAIT_BLIND_EXTENSION = float(os.environ.get("RLLM_EAI_WAIT_BLIND_EXTENSION", "1800"))

# A sandbox's max-run-time IS the lifetime of a leaked sandbox (nothing else
# reclaims one whose owner died). Observed episodes finish in <15 min, so 2 h
# leaves a wide margin while cutting the cost of each leak 3x vs the old 6 h.
# Raise via env for long-horizon experiments (higher turn caps / slower tools).
SANDBOX_MAX_RUN_TIME = int(os.environ.get("RLLM_EAI_SANDBOX_MAX_RUN_TIME", "7200"))

# Sandboxes whose kill could not be confirmed are appended here so a reaper
# can clean them up precisely, without guessing from job age.
LEAK_LOG = os.environ.get(
    "RLLM_EAI_LEAK_LOG", "/mnt/adea/data_rw/rollout_control/leaked_sandboxes.jsonl"
)


def _record_leak(job_id: str, name: str, err: str) -> None:
    """Append a leaked-sandbox record. Best-effort: never raises into teardown."""
    try:
        os.makedirs(os.path.dirname(LEAK_LOG), exist_ok=True)
        rec = {
            "job_id": job_id,
            "name": name,
            "parent": os.environ.get("EAI_JOB_ID", ""),
            "ts": time.time(),
            "error": err[:300],
        }
        with open(LEAK_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        logger.exception("failed to record leaked sandbox %s", job_id)


def _kill_job(job_id: str, attempts: int = 4) -> tuple[bool, str]:
    """Best-effort kill with a short retry ladder. Returns (killed, last_error).

    Ladder is deliberately short (~35 s): this runs on teardown, so a long
    ladder would stall rollout throughput during a control-plane outage.
    Whatever it cannot kill is handed to the leak log for the reaper.
    """
    last_err = ""
    for attempt in range(attempts):
        try:
            proc = _eai("job", "kill", job_id, timeout=60)
        except Exception as exc:  # TimeoutExpired / OSError
            last_err = repr(exc)
        else:
            if proc.returncode == 0:
                return True, ""
            last_err = (proc.stderr or "").strip()
            low = last_err.lower()
            # Already terminal or already gone: nothing left to kill.
            if "cannot cancel a job that is in state" in low or "not found" in low:
                return True, "already terminal"
            # Network errors count too: during an outage every kill fails, and
            # giving up early is exactly how sandboxes leak. What this ladder
            # still cannot kill goes to the leak log for the reaper.
            if not _is_retryable_control_plane(last_err):
                break
        if attempt < attempts - 1:
            time.sleep(5 * 2**attempt)
    return False, last_err


def map_image(image: str) -> str:
    """Map a Docker-Hub task image to its mirrored internal-registry name.

    ``namanjain12/aiohttp_final:abc`` → ``<registry>/r2e_aiohttp_final:abc``.
    Images already pointing at the internal registry pass through unchanged.
    """
    if image.startswith(INTERNAL_REGISTRY.split("/")[0]):
        return image
    name = image.split("/", 1)[-1]
    repo, _, tag = name.partition(":")
    # r2e2_* images carry the full EAI-compat layer (writable /testbed for the
    # job uid, /tests + /logs stubs) — see mirror_task_image_v2.sh.
    mapped = f"{INTERNAL_REGISTRY}/r2e2_{repo}:{tag}" if tag else f"{INTERNAL_REGISTRY}/r2e2_{repo}"
    return mapped


def _eai(*args: str, timeout: float | None = 120, input_text: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["eai", *args]
    env = {**os.environ, "EAI_PROFILE": EAI_PROFILE}
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, input=input_text
    )


class EAISandbox:
    """Sandbox implementation backed by an EAI batch job."""

    def __init__(
        self,
        name: str,
        image: str = "python:3.11-slim",
        cpu: int = 4,
        mem: int = 16,
        max_run_time: int = SANDBOX_MAX_RUN_TIME,
        ready_timeout: float = 600.0,
        **kwargs,
    ):
        self.name = name
        self.image = map_image(image)
        self._sbx_id = uuid.uuid4().hex[:10]
        job_name = "sbx_" + re.sub(r"[^a-z0-9_]", "_", name.lower())[:40] + "_" + self._sbx_id
        self._transfer_dir = os.path.join(TRANSFER_HOST_DIR, self._sbx_id)

        # Stamp the owning trainer's job id onto the sandbox. Without it an
        # orphan is only identifiable by AGE -- and after a trainer is killed
        # its sandboxes are minutes old, so any age gate loose enough to catch
        # them is loose enough to kill live ones. 258 orphans had to be reaped
        # by hand on 2026-08-27 after a planned restart, holding ~1000 CPUs and
        # starving the very run that replaced them. With this, "parent job is
        # not alive" is an exact test.
        parent_job = os.environ.get("EAI_JOB_ID", "")
        args = [
            "job", "new", "--preemptable",
            "-i", self.image,
            "--cpu", str(cpu), "--mem", str(mem),
            "--max-run-time", str(max_run_time),
            "--data", TRANSFER_DATA_SPEC,
            "--name", job_name,
            "--env", f"RLLM_PARENT_JOB={parent_job}",
            "--field", "id", "--no-header", "--format", "csv",
            "--", "sleep", "infinity",
        ]
        proc = None
        for attempt in range(SUBMIT_ATTEMPTS):
            proc = _eai(*args, timeout=180)
            if proc.returncode == 0:
                break
            # EAI API 5xx blips and network outages are transient; back off and
            # retry. The ladder totals ~25 min (10,20,40,80,160,300,300,...) to
            # ride out a full control-plane outage, not just a flap. Blocking
            # one rollout for 25 min is far cheaper than losing the run: the
            # 2026-08-26 outage killed four multi-hour trainings because the
            # old ~10 min ladder never even engaged for network errors.
            if attempt < SUBMIT_ATTEMPTS - 1 and _is_retryable_control_plane(proc.stderr or ""):
                time.sleep(min(300, 10 * 2**attempt))
                continue
            break
        if proc.returncode != 0:
            raise RuntimeError(f"EAISandbox {name}: job submission failed: {proc.stderr.strip()[:500]}")
        self.job_id = proc.stdout.strip().splitlines()[-1].strip()
        if not re.fullmatch(r"[0-9a-f-]{36}", self.job_id):
            # A job may well have been created; we just cannot address it.
            # Record by NAME so the reaper's age sweep can still find it.
            _record_leak("", job_name, f"unparseable job id: {proc.stdout[:200]!r}")
            raise RuntimeError(f"EAISandbox {name}: unexpected job id output: {proc.stdout[:200]!r}")

        # ANY failure from here on must kill the job we just submitted. If
        # __init__ raises, the object never reaches the caller, so the hook
        # layer's cleanup sees `sandbox is None` and close() is NEVER called —
        # the job then runs to its max-run-time cap unreferenced. This was the
        # dominant leak path (574 abandoned jobs on 2026-08-25, all from
        # `_wait_running` timing out during control-plane flaps), much larger
        # than the close() path.
        try:
            self._wait_running(ready_timeout)
        except BaseException as exc:
            killed, kill_err = _kill_job(self.job_id)
            if killed:
                logger.warning(
                    "EAISandbox %s: startup failed (%s); job %s killed",
                    name, exc, self.job_id,
                )
            else:
                logger.error(
                    "EAISandbox %s: startup failed AND job %s could not be killed: %s",
                    name, self.job_id, kill_err[:200],
                )
                _record_leak(self.job_id, name, f"startup cleanup failed: {kill_err}")
            try:
                shutil.rmtree(self._transfer_dir, ignore_errors=True)
            except Exception:
                pass
            raise
        logger.info("EAISandbox %s created (job: %s, image: %s)", name, self.job_id, self.image)

    def _state(self) -> str:
        proc = _eai("job", "get", self.job_id, "--field", "state", timeout=60)
        return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"

    def _wait_running(self, timeout: float) -> None:
        """Wait for RUNNING, pausing the clock while the control plane is blind.

        `_state()` returns "UNKNOWN" when `eai job get` fails, which during an
        outage means we cannot SEE the job -- not that it failed to start.
        Charging that blind time against the deadline converts a control-plane
        outage into a rollout failure, and (with raise_on_error=true) a rollout
        failure into a dead training run. That is the same mistake as
        classifying network errors as permanent; this is the third place it
        appeared.

        So blind polls extend the deadline instead of consuming it, bounded by
        WAIT_BLIND_EXTENSION so a genuinely unreachable API cannot hang a
        rollout forever. Observable non-RUNNING states (QUEUING) still consume
        the deadline normally -- a cluster that is merely busy is real evidence.
        """
        deadline = time.time() + timeout
        blind_total = 0.0
        blind_run = 0
        poll = 5.0
        while time.time() < deadline:
            state = self._state()
            if state == "RUNNING":
                return
            if state in _TERMINAL_STATES:
                raise RuntimeError(f"EAISandbox {self.name}: job {self.job_id} reached {state} before RUNNING")
            if state == "UNKNOWN":
                blind_run += 1
                if blind_total < WAIT_BLIND_EXTENSION:
                    deadline += poll
                    blind_total += poll
            else:
                blind_run = 0
            time.sleep(poll)
        extra = f" (+{blind_total:.0f}s blind)" if blind_total else ""
        raise RuntimeError(
            f"EAISandbox {self.name}: job {self.job_id} not RUNNING after {timeout}s{extra}")

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command in the sandbox job. ``user`` is ignored (fixed uid).

        Every command runs with ``HOME=/tmp``: the image's real HOME (``/root``)
        is not writable by the job uid, and harness installs (`uv tool install`),
        agent config files, and git state all need a writable home.
        """
        if user is not None:
            logger.debug("EAISandbox %s: ignoring user=%r (jobs run as a fixed uid)", self.name, user)
        wrapped = "export HOME=/tmp; " + command
        remote = wrapped
        if timeout is not None:
            remote = f"timeout {int(timeout)} bash -c {_shquote(wrapped)}"
        proc = None
        for attempt in range(6):
            proc = _eai(
                "job", "exec", self.job_id, "--", "bash", "-c", remote,
                timeout=(timeout + 60) if timeout is not None else EXEC_TIMEOUT_S,
            )
            if proc.returncode == 0:
                break
            # EAI control-plane 5xx blips surface as CLI "internal error"
            # (http: 500) etc. Retrying is safe only when the remote command
            # never started; exit 7 = transport error before execution.
            # Ladder totals ~5 min for multi-minute control-plane flaps.
            err = proc.stderr or ""
            # _is_transient requires a CLI error marker, so a remote command
            # that merely PRINTS "502" (line numbers, hashes, test ids) and
            # exits 1 is not re-executed — re-running a side-effecting agent
            # or verifier command would corrupt the episode.
            # A network error is only safe to retry when the CLI failed in
            # transport (exit 7), i.e. the remote command provably never ran.
            # On exit 1 the command DID run and merely printed something that
            # looks like a connectivity error -- re-running it would corrupt
            # the episode.
            transient = _is_transient(err) or (_is_network_error(err) and proc.returncode == 7)
            if attempt < 5 and transient and proc.returncode in (1, 7):
                time.sleep(min(120, 8 * 2**attempt))
                continue
            break
        if proc.returncode != 0:
            short_tail = 600
            err = proc.stderr or ""
            err_tail = err[-short_tail:] if len(err) > short_tail else err
            logger.debug(
                "Command failed (exit %d) in sandbox %s: %s\nstdout (tail):\n%s\nstderr (tail):\n%s",
                proc.returncode, self.name, command, proc.stdout[-8000:], err[-8000:],
            )
            raise RuntimeError(
                f"Command failed (exit {proc.returncode}) in sandbox {self.name}: {command}\nstderr (tail):\n{err_tail}"
            )
        return proc.stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload one file via the shared VAST transfer dir + in-sandbox cp."""
        os.makedirs(self._transfer_dir, exist_ok=True)
        staged = os.path.join(self._transfer_dir, os.path.basename(remote_path))
        shutil.copy2(local_path, staged)
        rel = os.path.relpath(staged, TRANSFER_HOST_DIR)
        self.exec(
            f"mkdir -p {_shquote(os.path.dirname(remote_path))} && "
            f"cp {_shquote(os.path.join(TRANSFER_REMOTE_DIR, rel))} {_shquote(remote_path)}"
        )

    def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Upload a directory tree via the shared VAST transfer dir."""
        os.makedirs(self._transfer_dir, exist_ok=True)
        staged = os.path.join(self._transfer_dir, os.path.basename(remote_path.rstrip("/")))
        if os.path.exists(staged):
            shutil.rmtree(staged)
        shutil.copytree(local_path, staged)
        rel = os.path.relpath(staged, TRANSFER_HOST_DIR)
        # Content-copy (src/. -> dest/) so a pre-existing destination dir is
        # populated rather than nested into (plain `cp -a src dest` would
        # create dest/src when dest exists — e.g. the baked /tests stub).
        dest = remote_path.rstrip("/")
        self.exec(
            f"mkdir -p {_shquote(dest)} && "
            f"cp -a {_shquote(os.path.join(TRANSFER_REMOTE_DIR, rel) + '/.')} {_shquote(dest + '/')}"
        )

    def is_alive(self) -> bool:
        try:
            return self._state() == "RUNNING"
        except Exception:
            logger.debug("EAISandbox %s is_alive check failed — treating as dead", self.name, exc_info=True)
            return False

    def close(self) -> None:
        """Kill the sandbox job, then drop its transfer dir.

        Teardown must be as resilient as creation: `_eai` does NOT raise on a
        non-zero exit, so the old ``try/except: pass`` swallowed 5xx kills and
        still logged success — every control-plane flap silently leaked a
        4-CPU/16-GB job until its ``max_run_time`` cap. The ladder is kept
        SHORT (~35 s) so teardown latency stays bounded during an outage;
        whatever it cannot kill is written to the leak log for the reaper
        (`scripts/eai/reap_sandboxes.py`).
        """
        killed, last_err = _kill_job(self.job_id)

        try:
            if os.path.isdir(self._transfer_dir):
                shutil.rmtree(self._transfer_dir, ignore_errors=True)
        except Exception:
            pass

        if killed:
            logger.info("EAISandbox %s closed (job: %s)", self.name, self.job_id)
        else:
            logger.error(
                "EAISandbox %s LEAKED job %s — kill failed after retries: %s",
                self.name, self.job_id, last_err[:300],
            )
            _record_leak(self.job_id, self.name, last_err)


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def create_eai_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> EAISandbox:
    """Factory function for creating an EAISandbox."""
    return EAISandbox(name=name, image=image, **kwargs)
