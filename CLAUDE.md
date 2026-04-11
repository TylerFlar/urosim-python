# UroSim (Python)

## Project overview

UroSim is a ureteroscopy simulator. This repository is the **Python side** of the system — Unity handles physics and rendering on the other side of a gRPC connection.

Python owns:
- **anatomy/** — anatomical modeling (meshes, SDFs, geometry)
- **nav/** — navigation and planning on anatomical structures
- **client/** — the only module that talks to Unity (gRPC)
- **eval/** — evaluation on saved trajectories/recordings (no live sim)
- **agents/** — scoped policies/LLM agents that drive the scope
- **datagen/**, **datasets/**, **recording/** — data pipelines
- **env/** — gym-style environment wrappers
- **sim2real/** — image translation / domain transfer

## Build commands

```bash
uv sync                                   # install deps (incl. dev group)
pytest tests/                             # run tests
pytest tests/ -m "not unity"              # skip tests that need a live Unity sim
ruff check .                              # lint
mypy urosim/ --ignore-missing-imports     # type-check
```

## Code style

- Type hints on all public functions and dataclass fields.
- Prefer `@dataclass` (or `@dataclass(frozen=True)`) for plain data containers.
- Use `numpy.ndarray` for numeric data; document shape and dtype in docstrings.
- **Google-style docstrings** on public functions, classes, and modules.
- Line length: 100.

## Module boundary rules

These are hard rules — violating them breaks the layering that keeps evaluation reproducible and anatomy testable in isolation.

- `anatomy/` **never** imports `grpc`, `protobuf`, or `torch`.
- `nav/` depends only on `anatomy/` (plus stdlib, numpy, scipy, networkx).
- `eval/` operates on **saved files only** — no live simulator calls, no `client/` imports.
- `client/` is the **only** module that talks to Unity.
- `sim2real/` **never** imports `client/`.

## Testing rules

- Every function gets a test.
- `anatomy/` tests must be **deterministic** — seed every RNG (`numpy.random.default_rng(seed)`, not global state).
- Tests that require a running Unity simulator must be marked `@pytest.mark.unity` and are skipped by default in CI via `-m "not unity"`.
- The `unity` mark is registered in `tests/conftest.py`.
