"""Delivery ops: deliverable presets and preview proxies."""

from __future__ import annotations

from pydantic import Field

from .. import deliver
from ..model import Project
from ..ops import Ctx, Params, op


class DeliverablePresetP(Params):
    name: str = Field(description="deliverable name, e.g. 'short' or 'main'")
    preset: str = Field(description="one of: " + ", ".join(sorted(deliver.PRESETS)))


@op("deliverable.preset", DeliverablePresetP)
def deliverable_preset(ctx: Ctx, p: Project, a: DeliverablePresetP):
    """Create/replace a deliverable from a platform preset (size, loudness, reframing). Fine-tune with
    deliverable.set. Vertical presets follow the tracked subject (run asset.analyze faces first)."""
    p.deliverables[a.name] = deliver.preset(a.name, a.preset)
    return p.deliverables[a.name].model_dump()


class DeliverableRemoveP(Params):
    name: str


@op("deliverable.remove", DeliverableRemoveP)
def deliverable_remove(ctx: Ctx, p: Project, a: DeliverableRemoveP):
    """Remove a deliverable."""
    if a.name not in p.deliverables:
        raise KeyError(f"no deliverable {a.name!r}")
    del p.deliverables[a.name]
    return {"removed": a.name}


class ProxyP(Params):
    assets: list[str] = Field(default_factory=list, description="asset ids; empty = every video asset")


@op("asset.make_proxy", ProxyP)
def asset_make_proxy(ctx: Ctx, p: Project, a: ProxyP):
    """Create 360p proxies so preview renders are fast (finals always use full-quality media)."""
    return {"proxies": deliver.make_proxies(ctx.runner, ctx.root, p, a.assets or None)}
