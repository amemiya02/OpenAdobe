"""Runs engine binaries (melt, ffmpeg, ffprobe…) inside the version-locked engine image.

Directories are bind-mounted at their identical absolute path, so paths written into compiled
engine files are valid both on the host and inside the container.

    OPENADOBE_ENGINE=docker|local   (default docker)
    OPENADOBE_IMAGE=openadobe/engines:dev
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_IMAGE = "openadobe/engines:dev"
DOCKERFILE = Path(__file__).resolve().parents[3] / "docker" / "engines.Dockerfile"


class EngineError(RuntimeError):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def _collapse(paths: Iterable[Path]) -> list[Path]:
    """Drop mounts already covered by a parent mount."""
    out: list[Path] = []
    for p in sorted({Path(p).resolve() for p in paths}, key=lambda x: len(x.parts)):
        if not any(p == q or q in p.parents for q in out):
            out.append(p)
    return out


class Runner:
    def __init__(self, mode: str | None = None, image: str | None = None):
        self.mode = mode or os.environ.get("OPENADOBE_ENGINE", "docker")
        self.image = image or os.environ.get("OPENADOBE_IMAGE", DEFAULT_IMAGE)
        if self.mode not in ("docker", "local"):
            raise EngineError(f"OPENADOBE_ENGINE must be docker or local, not {self.mode!r}")

    def command(self, args: Sequence[str], mounts: Iterable[Path] = (), ro: Iterable[Path] = ()) -> list[str]:
        if self.mode == "local":
            return list(args)
        # Qt-based MLT services (glaxnimate, qtblend, kdenlivetitle) only check that DISPLAY is set;
        # with the offscreen platform they render without any X server.
        cmd = ["docker", "run", "--rm", "-i", "-e", "HOME=/tmp", "-e", "LC_ALL=C.UTF-8",
               "-e", "DISPLAY=:99", "-e", "QT_QPA_PLATFORM=offscreen"]
        if sys.platform.startswith("linux"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        rw = _collapse(mounts)
        for p in rw:
            cmd += ["-v", f"{p}:{p}"]
        for p in _collapse(ro):
            if not any(p == q or q in p.parents for q in rw):
                cmd += ["-v", f"{p}:{p}:ro"]
        return cmd + [self.image, *args]

    def run(
        self,
        args: Sequence[str],
        mounts: Iterable[Path] = (),
        ro: Iterable[Path] = (),
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = self.command(args, mounts, ro)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace", check=False)
        except FileNotFoundError as exc:
            raise EngineError(f"cannot start {cmd[0]!r}: {exc}. Run `openadobe doctor`.") from exc
        if check and proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise EngineError(f"{args[0]} failed (exit {proc.returncode}):\n{tail}", proc.stderr)
        return proc

    # ---- health ----------------------------------------------------------------------
    def image_present(self) -> bool:
        if self.mode == "local":
            return True
        if not shutil.which("docker"):
            return False
        proc = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, check=False)
        return proc.returncode == 0

    def build_image(self) -> None:
        subprocess.run(
            ["docker", "build", "-f", str(DOCKERFILE), "-t", self.image, str(DOCKERFILE.parent)], check=True
        )

    def versions(self) -> dict[str, str]:
        if self.mode == "local":
            out = {}
            for tool in ("melt", "ffmpeg"):
                path = shutil.which(tool)
                out[tool] = path or "missing"
            return out
        proc = self.run(["cat", "/etc/openadobe-engines.lock"])
        return dict(line.split("=", 1) for line in proc.stdout.split() if "=" in line)
