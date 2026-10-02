"""Mixing-engine ops: choose who mixes the audio (MLT or Ardour) and export per-role stems."""

from __future__ import annotations

from typing import Literal

from ..engines import ardour
from ..model import Project
from ..ops import Ctx, Params, op


class EngineP(Params):
    engine: Literal["mlt", "ardour"]


@op("mix.engine", EngineP)
def mix_engine(ctx: Ctx, p: Project, a: EngineP):
    """Choose the engine that mixes the final audio: 'mlt' (default, fast, mixes inside the video render) or
    'ardour' (a real DAW session: one bus per role, ACE plugins for track effects, stems available).
    If the project uses something the Ardour adapter cannot reproduce (clip speed, clip-level audio effects,
    unmapped track effects) delivery mixes with MLT instead and says so; `would_fall_back` lists those items."""
    p.mix.engine = a.engine
    out: dict = {"engine": a.engine}
    if a.engine == "ardour":
        from ..engines import ardour

        out["would_fall_back"] = ardour.losses(p)
    return out


class ExportStemsP(Params):
    pass


@op("mix.export_stems", ExportStemsP)
def mix_export_stems(ctx: Ctx, p: Project, a: ExportStemsP):
    """Mix the project in Ardour and export the master mix plus one stem per role present (dia/mus/sfx/amb) as
    48 kHz 24-bit stereo WAVs into renders/stems-r<revision>/. Does not change the project. Returns the paths,
    the Ardour session directory, which track effects were mapped to Ardour plugins and which were skipped."""
    out_dir = ctx.root / "renders" / f"stems-r{p.revision}"
    res = ardour.mixdown(ctx.runner, ctx.root, p, out_dir, stems=True)
    return {
        "mix": str(res["mix"]), "stems": {role: str(path) for role, path in res["stems"].items()},
        "session": str(res["session"]), "seconds": round(res["seconds"], 1),
        "mapped": res["report"]["mapped"], "skipped": res["report"]["skipped"],
    }
