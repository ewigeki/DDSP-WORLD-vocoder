# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python DDSP/WORLD vocoder project. Core code lives in `src/`: model definitions are in `src/models/`, Lightning modules and signal-processing pipelines are in `src/modules/`, stage orchestration lives in `src/stages/`, dataset and score readers are in `src/data/`, and shared training helpers are in `src/training.py`. Hydra configuration is in top-level `conf/`, with full-pipeline settings in `conf/run.yaml` and individual stage configs such as `conf/train_f0.yaml`, `conf/train_world.yaml`, and `conf/preprocess_wav.yaml`. The full pipeline entry point is `src/run.py`; individual stage entry points are `src/stages/preprocess_wav.py`, `src/stages/train_f0.py`, and `src/stages/train_world.py`. Local datasets should stay under `data/`, while generated artifacts such as `lightning_logs/`, `logs/`, `outputs/`, and checkpoints should not be treated as source code.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install pinned Python dependencies, including PyTorch, Lightning, librosa, Hydra, and pyworld.
- `python -m src.run`: run the configured pipeline stages from `conf/run.yaml`.
- `python -m src.stages.preprocess_wav`: cache WAV features according to `conf/preprocess_wav.yaml`.
- `python -m src.stages.train_f0`: run F0 training according to `conf/train_f0.yaml`.
- `python -m src.stages.train_world`: run WORLD training according to `conf/train_world.yaml`; set `model.f0_checkpoint_path` unless the checkpoint is passed from the full pipeline.
- `python -m src.run stages.from=2 stages.to=3`: run a subset of the full pipeline with Hydra overrides.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, snake_case for functions and variables, and PascalCase for classes such as `WORLDVocoder` and `F0Pipeline`. Keep modules focused by domain: readers under `src/data/readers/`, neural network components under `src/models/`, Lightning modules and signal-processing pipelines under `src/modules/`, and orchestration code under `src/stages/` or `src/run.py`. Prefer `pathlib.Path` for filesystem paths, type hints for public helpers, and concise comments only where the signal-processing or model logic is non-obvious.

## Testing Guidelines

No test suite or test runner is currently configured. When adding tests, use `pytest`, place files under `tests/`, and name them `test_<module>.py`. Prioritize unit tests for readers, tensor shape contracts, config composition, and utility functions before adding slow training checks. For manual verification, run `python -m src.run stages.from=1 stages.to=1` against a tiny WAV subset, then a short F0 or WORLD training smoke test with small split and trainer overrides.

## Commit & Pull Request Guidelines

The existing Git history uses Conventional Commit-style subjects, for example `feat: add WORLD training pipeline`, `chore: add visualization helper`, and `refactor: add gitignore`. Follow that pattern with concise imperative subjects. Pull requests should describe the change, list verification commands, note dataset/checkpoint assumptions, and link any relevant issue. Include screenshots or TensorBoard notes only for visualization or training-behavior changes.

## Security & Configuration Tips

Do not commit private datasets, generated checkpoints, TensorBoard event files, or machine-specific paths. Keep large artifacts in ignored local directories or external storage, and document required paths through config or setup notes instead of hard-coding personal locations.
