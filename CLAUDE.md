# OpenAdobe

Agent-operated post-production on real engines. Read `docs/ARCHITECTURE.md` first (decisions §1.1–1.2).

- Python 3.12 via uv: `uv sync --extra asr --extra mcp --extra otio --extra analysis`
- Engines run in Docker image `openadobe/engines:dev` (`docker/engines.Dockerfile`); `uv run openadobe doctor --build`
- Tests: `uv run pytest -m "not engine"` (fast) / `uv run pytest` (needs the image; ~15 min incl. Blender/Ardour)
- Acceptance runs: `examples/talking_head_demo.sh [workdir]` (P0) and `examples/full_showcase.sh [workdir]` (P1–P3, ~10 min);
  both use macOS `say` for the voice-over

Rules
- The model (`src/openadobe/model.py`) is the source of truth; engine files are compiled outputs. Never add
  code that parses engine files back into the model.
- Every project mutation is an op in `src/openadobe/ops.py` (typed params, docstring = agent-facing doc).
  Pure edit logic lives in `timeline.py` and stays I/O-free and unit-tested.
- The control layer orchestrates engines; it never renders pixels or samples itself.
- New engine behaviour needs an engine e2e test that renders and measures (luma, loudness, …), not just XML asserts.
