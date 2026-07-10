# DDSP-WORLD Vocoder

DDSP-WORLD Vocoder is an experimental project for training neural singing vocoders with WORLD-style acoustic features and DDSP-style signal modeling.

The goal is to make a lightweight and interpretable vocoder training pipeline for singing voice synthesis experiments.

## What This Project Does

This repository provides a training pipeline for a WORLD/DDSP-style vocoder.

Given a folder of WAV files, the pipeline can:

1. preprocess audio files into cached training samples;
2. train an F0 prediction model;
3. train a WORLD/DDSP vocoder conditioned on the trained F0 model.

The project is mainly intended for research, experiments, and learning how neural vocoders for singing voice synthesis can be built.

## Current Status

The project is experimental.

Currently supported:

* WAV dataset preprocessing;
* cached feature generation;
* F0 model training;
* WORLD/DDSP vocoder training;
* Hydra-based configuration;
* staged training pipeline.

Not included yet:

* pretrained model checkpoints;
* public dataset;
* standalone inference script;
* model export utilities.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux, you may also need:

```bash
sudo apt install build-essential libsndfile1
```

Put WAV files into:

```text
data/
```

Run the full training pipeline:

```bash
python -m src.run
```

This will run the default stages:

1. preprocess WAV files;
2. train the F0 predictor;
3. train the WORLD/DDSP vocoder.

## Data Layout

Default input directory:

```text
data/
```

Default processed cache directory:

```text
data/processed/
```

Default data settings:

| Parameter          | Value         |
| ------------------ | ------------- |
| Sample rate        | `16000`       |
| Sample duration    | `4.0` seconds |
| WORLD frame period | `4.0` ms      |
| Batch size         | `8`           |
| Data source        | `wav`         |

To use already preprocessed samples:

```bash
python -m src.run data.source=precomputed data.processed_dir=data/processed
```

## Training Pipeline

The default pipeline has three stages:

| Stage | Name             | Description                                          |
| ----- | ---------------- | ---------------------------------------------------- |
| 1     | `preprocess_wav` | Load WAV files and build cached training samples     |
| 2     | `train_f0`       | Train the F0 prediction model                        |
| 3     | `train_world`    | Train the WORLD/DDSP vocoder using the F0 checkpoint |

Run the full pipeline:

```bash
python -m src.run
```

Run only preprocessing:

```bash
python -m src.run stages.from=1 stages.to=1
```

Run F0 and vocoder training:

```bash
python -m src.run stages.from=2 stages.to=3
```

Run stages directly:

```bash
python -m src.stages.preprocess_wav
python -m src.stages.train_f0
python -m src.stages.train_world
```

When running WORLD/DDSP training directly, provide an F0 checkpoint:

```bash
python -m src.stages.train_world model.f0_checkpoint_path=/path/to/f0.ckpt
```

When running the full pipeline, the F0 checkpoint from stage 2 is passed to stage 3 automatically.

## Configuration

The project uses Hydra configs.

Main configs:

```text
conf/run.yaml              # full pipeline
conf/preprocess_wav.yaml   # preprocessing stage
conf/train_f0.yaml         # F0 training
conf/train_world.yaml      # WORLD/DDSP vocoder training
conf/data/wav.yaml         # WAV dataset settings
```

Examples:

```bash
# Change input WAV directory
python -m src.run data.root_dir=/path/to/wavs

# Rebuild cached features
python -m src.run data.dataset.rebuild_cache=true

# Change batch size
python -m src.run data.loader.batch_size=4

# Run a development config
python -m src.stages.train_f0 --config-name train_f0.dev
```

## Roadmap

Planned work:

* **[v0.0.5] Inference ~ add standalone inference code for loading trained local checkpoints and saving generated audio.**
* **Training guide ~ document practical tips for stable model training, dataset quality, F0 issues, validation checks, and common failure modes.**
* **Parallel preprocessing ~ add multiprocessing support for faster WAV preprocessing on larger datasets.**
* **Documentation ~ add a minimal end-to-end example from WAV files to trained checkpoint and generated audio.**
* **Experiments ~ add notes from model ablations and training experiments.**

Remote model download or model registry support is not planned for the first inference release. The initial inference version will focus on local checkpoints.

## Repository Structure

```text
.
├── conf/                 # Hydra configs for data, models, trainers, loggers, and stages
├── src/
│   ├── data/             # datasets, WAV loading, feature caching
│   ├── models/           # neural network and vocoder components
│   ├── modules/          # PyTorch Lightning training modules
│   ├── stages/           # standalone pipeline stages
│   ├── run.py            # full pipeline runner
│   ├── training.py       # dataloader/training helpers
│   └── visualization.py  # plotting utilities
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated artifacts such as `logs/`, `lightning_logs/`, `outputs/`, checkpoints, and cached datasets are local experiment outputs and are not part of the source code.

## Notes and Limitations

* The current pipeline is aimed at experimentation and research iteration.
* Inference/export scripts are planned but not yet included.

## References

Relevant papers and related vocoder work:

* [Differentiable WORLD Synthesizer-based Neural Vocoder With Application To End-To-End Audio Style Transfer](https://arxiv.org/abs/2208.07282)
* [Ultra-lightweight Neural Differential DSP Vocoder For High Quality Speech Synthesis](https://arxiv.org/abs/2401.10460)
* [Source-Filter HiFi-GAN: Fast and Pitch Controllable High-Fidelity Neural Vocoder](https://arxiv.org/abs/2210.15533)
* [Unified Source-Filter GAN: Unified Source-filter Network Based On Factorization of Quasi-Periodic Parallel WaveGAN](https://arxiv.org/abs/2104.04668)
* [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
* [BigVGAN: A Universal Neural Vocoder with Large-Scale Training](https://arxiv.org/abs/2206.04658)

## License

MIT License.
