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
        max_run_time: int = 21600,
        ready_timeout: float = 600.0,
        **kwargs,
    ):
        self.name = name
        self.image = map_image(image)
        self._sbx_id = uuid.uuid4().hex[:10]
        job_name = "sbx_" + re.sub(r"[^a-z0-9_]", "_", name.lower())[:40] + "_" + self._sbx_id
        self._transfer_dir = os.path.join(TRANSFER_HOST_DIR, self._sbx_id)

        args = [
            "job", "new", "--preemptable",
            "-i", self.image,
            "--cpu", str(cpu), "--mem", str(mem),
            "--max-run-time", str(max_run_time),
            "--data", TRANSFER_DATA_SPEC,
            "--name", job_name,
            "--field", "id", "--no-header", "--format", "csv",
            "--", "sleep", "infinity",
        ]
        proc = _eai(*args, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(f"EAISandbox {name}: job submission failed: {proc.stderr.strip()[:500]}")
        self.job_id = proc.stdout.strip().splitlines()[-1].strip()
        if not re.fullmatch(r"[0-9a-f-]{36}", self.job_id):
            raise RuntimeError(f"EAISandbox {name}: unexpected job id output: {proc.stdout[:200]!r}")

        self._wait_running(ready_timeout)
        logger.info("EAISandbox %s created (job: %s, image: %s)", name, self.job_id, self.image)

    def _state(self) -> str:
        proc = _eai("job", "get", self.job_id, "--field", "state", timeout=60)
        return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"

    def _wait_running(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self._state()
            if state == "RUNNING":
                return
            if state in _TERMINAL_STATES:
                raise RuntimeError(f"EAISandbox {self.name}: job {self.job_id} reached {state} before RUNNING")
            time.sleep(5)
        raise RuntimeError(f"EAISandbox {self.name}: job {self.job_id} not RUNNING after {timeout}s")

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        """Execute a command in the sandbox job. ``user`` is ignored (fixed uid).

        Every command runs with ``HOME=/tmp``: the image's real HOME (``/root``)
        is not writable by the job uid, and harness installs (`uv tool install`),
        agent config files, and git state all need a writable home.
        """
        if user is not None:
            logger.debug("EAISandbox %s: ignoring user=%r (jobs run as a fixed uid)", self.name, user)
        remote = "export HOME=/tmp; " + command
        if timeout is not None:
            remote = f"timeout {int(timeout)} bash -c {_shquote(command)}"
        proc = _eai(
            "job", "exec", self.job_id, "--", "bash", "-c", remote,
            timeout=(timeout + 60) if timeout is not None else 3600,
        )
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
        try:
            _eai("job", "kill", self.job_id, timeout=60)
        except Exception:
            pass
        try:
            if os.path.isdir(self._transfer_dir):
                shutil.rmtree(self._transfer_dir, ignore_errors=True)
        except Exception:
            pass
        logger.info("EAISandbox %s closed (job: %s)", self.name, self.job_id)


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def create_eai_sandbox(name: str, image: str = "python:3.11-slim", **kwargs) -> EAISandbox:
    """Factory function for creating an EAISandbox."""
    return EAISandbox(name=name, image=image, **kwargs)
