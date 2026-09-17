# Atlas

Measures whether an evolving knowledge graph retrieves better context than vector similarity search, on a synthetic corpus where facts go stale over time. Active, last touched 2026-08-23.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Python 3.14 with ruff, mypy and pytest is on PATH here; no `.venv` exists in the folder yet, and there is no lockfile.

## Checks

| Command | Proves |
|---|---|
| `.venv/Scripts/python.exe -m ruff check .` | style clean, verified 2026-09-17 |
| `.venv/Scripts/python.exe -m ruff format --check .` | formatting clean, documented in pyproject.toml, not run here |
| `.venv/Scripts/python.exe -m mypy .` | types clean, documented in pyproject.toml, not run here (needs the `dev` extras installed) |
| `.venv/Scripts/python.exe -m pytest` | tests pass, documented in pyproject.toml, not run here |

Run the whole table before saying the tree is green, and quote the fresh output.

## Run

```bash
atlas --seed 42
```

No services needed: the benchmark runs in memory. The `docker/docker-compose.yml` file starts Postgres, Qdrant and Neo4j that nothing in the code connects to, so skip it.

## Layout

- `src/atlas/` package: cli.py, config.py, types.py, embedding.py, logging.py, bench/, memory/, experiment/
- `tests/` flat, one `test_<module>.py` per source module
- `docker/docker-compose.yml` unused by the code, leave alone

## Rules

- The `graph`, `tracking` and `swebench` extras and the related `Settings` fields in `config.py` are consumed by no source file. Do not wire them up without being asked.
- Reported numbers depend on the knobs in `config.py`; `--seed 42` reproduces the README table.
- Every behaviour change carries a test in `tests/`. Run the smallest suite while iterating, the full table before handoff.
- No DESIGN.md: this project has no visual surface.
- Commit in the repo's existing style (conventional commits, from git log). Never push, tag or open a PR unless asked.

## Workspace

This repo lives in the Yashiru workspace. When `_second_brain/` sits above this folder, the workspace root `AGENTS.md` applies as well: ask `python ../../../_second_brain/recall.py "question"` before reading widely, and record a durable fact with `python ../../../_second_brain/remember.py Atlas "fact"`.
