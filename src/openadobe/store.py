"""On-disk project layout, atomic saves, operation log and undo snapshots.

    <project>/
      project.oa.json   the model (source of truth)
      ops.jsonl         append-only operation log
      history/          model snapshot taken before each revision (undo)
      media/            normalised mezzanine media
      build/            compiled engine files (regenerated, never edited)
      renders/          deliverables, named by revision
      perceive/         frames, contact sheets, waveforms for agent review
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from pathlib import Path

from .model import Project, Track, now_iso
from .timebase import Rate

PROJECT_FILE = "project.oa.json"
SUBDIRS = ("history", "media", "build", "renders", "perceive")


class ProjectError(Exception):
    def __init__(self, message: str, issues: list[dict] | None = None):
        super().__init__(message)
        self.issues = issues or []


def project_path(root: Path) -> Path:
    return Path(root) / PROJECT_FILE


def init(root: Path, name: str, rate: Rate, width: int, height: int) -> Project:
    root = Path(root).resolve()
    if project_path(root).exists():
        raise ProjectError(f"project already exists at {root}")
    root.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    project = Project(name=name, rate=rate, width=width, height=height)
    project.tracks = [
        Track(id="V1", kind="video", role="main", name="Video 1"),
        Track(id="A1", kind="audio", role="dia", name="Dialogue"),
        Track(id="A2", kind="audio", role="mus", name="Music"),
    ]
    project.counters = {"V": 1, "A": 2}
    save(root, project)
    return project


def load(root: Path) -> Project:
    path = project_path(root)
    if not path.exists():
        raise ProjectError(f"no OpenAdobe project at {Path(root).resolve()} (run `openadobe init`)")
    return Project.model_validate_json(path.read_text("utf-8"))


def save(root: Path, project: Project) -> None:
    project.updated = now_iso()
    path = project_path(root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(project.model_dump_json(indent=1), "utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(root: Path) -> Iterator[None]:
    """Serialise writers: two agents editing one project must not interleave."""
    with open(Path(root) / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def snapshot(root: Path, project: Project) -> None:
    (Path(root) / "history").mkdir(exist_ok=True)
    path = Path(root) / "history" / f"rev-{project.revision:06d}.json"
    path.write_text(project.model_dump_json(), "utf-8")


def append_log(root: Path, entry: dict) -> None:
    with open(Path(root) / "ops.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log(root: Path, last: int | None = None) -> list[dict]:
    path = Path(root) / "ops.jsonl"
    if not path.exists():
        return []
    lines = path.read_text("utf-8").splitlines()
    if last:
        lines = lines[-last:]
    return [json.loads(line) for line in lines if line.strip()]


def undo(root: Path) -> Project:
    """Revert the latest op that has not been undone yet; the undo itself is a new revision.

    ``history/rev-N.json`` always holds the model as it was at revision N, so undoing the op
    that produced revision r restores ``rev-(r-1)``.
    """
    with locked(root):
        current = load(root)
        log = read_log(root)
        undone = {e["undoes"] for e in log if e.get("op") == "history.undo"}
        target = next(
            (e["revision"] for e in reversed(log) if e.get("op") != "history.undo" and e["revision"] not in undone),
            None,
        )
        snap = Path(root) / "history" / f"rev-{(target or 0) - 1:06d}.json"
        if target is None or not snap.exists():
            raise ProjectError("nothing to undo")
        snapshot(root, current)
        restored = Project.model_validate_json(snap.read_text("utf-8"))
        restored.revision = current.revision + 1
        save(root, restored)
        append_log(root, {"revision": restored.revision, "op": "history.undo", "undoes": target, "ts": now_iso()})
        return restored
