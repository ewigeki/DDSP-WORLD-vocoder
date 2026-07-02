# DDSP WORLD Vocoder

Python project for training a DDSP/WORLD-style singing vocoder with PyTorch Lightning, Hydra configuration, WORLD feature extraction, and CREPE-based F0 prediction.

## Project Layout

- `src/run.py`: full pipeline runner with selectable stages.
- `src/stages/`: individual stage entry points for WAV preprocessing, F0 training, and WORLD training.
- `src/modules/`: Lightning modules and training pipelines.
- `src/models/`: neural network components and vocoder implementations.
- `src/data/`: datasets, readers, and WAV feature caching.
- `conf/`: Hydra configuration for data, models, modules, logging, trainers, and stage runs.
- `data/`: local datasets and generated feature caches.

Generated training outputs such as `logs/`, `lightning_logs/`, `outputs/`, and checkpoints are local artifacts and should not be treated as source code.

## Setup

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file lists the direct libraries used by the project without hard-pinning every transitive package. It includes PyTorch, torchaudio, Lightning, Hydra, librosa, pyworld, matplotlib, TensorBoard logging support, and WAV/audio helpers.

Install the PyTorch build that matches your machine if the default `pip` wheel is not appropriate. For CUDA-specific or CPU-only installs, follow the PyTorch install command for your platform first, then run:

```bash
pip install -r requirements.txt
```

On Linux, `pyworld` and audio libraries may need system build/audio packages. The provided Dockerfile installs `build-essential` and `libsndfile1`, which are the usual requirements for this project.

## Data

By default, `conf/data/wav.yaml` looks for WAV files under `data/` and writes cached samples to `data/processed/`.

Important defaults:

- Sample rate: `16000`
- Sample duration: `4.0` seconds
- WORLD frame period: `4.0` ms
- Batch size: `8`
- Data source: `wav`

To train from precomputed caches instead of raw WAV files, set:

```bash
python -m src.run data.source=precomputed data.processed_dir=data/processed
```

## Running

Run the full pipeline configured in `conf/run.yaml`:

```bash
python -m src.run
```

Run only selected stages:

```bash
python -m src.run stages.from=1 stages.to=1
python -m src.run stages.from=2 stages.to=3
```

Run stages directly:

```bash
python -m src.stages.preprocess_wav
python -m src.stages.train_f0
python -m src.stages.train_world
```

WORLD training requires a trained F0 checkpoint. When running `train_world` directly, override the checkpoint path for your machine:

```bash
python -m src.stages.train_world model.f0_checkpoint_path=/path/to/f0.ckpt
```

The full pipeline passes the F0 checkpoint from stage 2 into stage 3 automatically.
