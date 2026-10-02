"""Feature op sets. Each module registers its ops with ``openadobe.ops.op`` on import.

One owner per module; shared helpers come from ``openadobe.ops`` (Params, Ctx, Time, op, resolve_time).
"""

from importlib import import_module

MODULES = ("editing", "effects", "comps", "audio", "mixing", "analysis", "delivery")

for _name in MODULES:
    try:
        import_module(f"{__name__}.{_name}")
    except ModuleNotFoundError as exc:  # a module not written yet
        if exc.name != f"{__name__}.{_name}":
            raise
