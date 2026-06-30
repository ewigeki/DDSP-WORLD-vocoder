# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python DDSP/WORLD vocoder project. Core code lives in `src/`: model definitions are in `src/models/`, Lightning modules and signal-processing pipelines are in `src/pipelines/`, dataset and score readers are in `src/data/`, and Hydra configuration is in `src/conf/config.yaml`. Training entry points are `src/train_f0.py` and `src/train_world.py`; `src/cli.py` is a minimal Hydra config runner. Local datasets should stay under `data/`, while generated artifacts such as `lightning_logs/`, `outputs/`, and checkpoints should not be treated as source code.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install pinned Python dependencies, including PyTorch, Lightning, librosa, Hydra, and pyworld.
- `python -m src.cli`: print the active Hydra configuration.
- `python -m src.train_f0`: run the F0 training pipeline after setting `DATASET_ROOT_DIR` and related constants in the script.
- `python -m src.train_world`: run WORLD training after configuring dataset and F0 checkpoint paths.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, snake_case for functions and variables, and PascalCase for classes such as `WORLDVocoder` and `F0Pipeline`. Keep modules focused by domain: readers under `src/data/readers/`, neural network components under `src/models/`, and orchestration code under `src/pipelines/` or training scripts. Prefer `pathlib.Path` for filesystem paths, type hints for public helpers, and concise comments only where the signal-processing or model logic is non-obvious.

## Testing Guidelines

No test suite or test runner is currently configured. When adding tests, use `pytest`, place files under `tests/`, and name them `test_<module>.py`. Prioritize unit tests for readers, tensor shape contracts, and utility functions before adding slow training checks. For manual verification, run `python -m src.cli` and a short training smoke test on a tiny WAV subset.

## Commit & Pull Request Guidelines

The existing Git history uses Conventional Commit-style subjects, for example `feat: add WORLD training pipeline`, `chore: add visualization helper`, and `refactor: add gitignore`. Follow that pattern with concise imperative subjects. Pull requests should describe the change, list verification commands, note dataset/checkpoint assumptions, and link any relevant issue. Include screenshots or TensorBoard notes only for visualization or training-behavior changes.

## Security & Configuration Tips

Do not commit private datasets, generated checkpoints, TensorBoard event files, or machine-specific paths. Keep large artifacts in ignored local directories or external storage, and document required paths through config or setup notes instead of hard-coding personal locations.
