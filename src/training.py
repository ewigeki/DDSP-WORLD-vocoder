from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.source import PrecomputedWavDataset, WavDataset
from src.utils import find_wav_files


def _as_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def _split_paths(paths: list[Path], cfg: DictConfig) -> tuple[list[Path], list[Path], list[Path]]:
    split_cfg = cfg.get("split", {})
    val_size = int(split_cfg.get("val_size", 3))
    test_size = int(split_cfg.get("test_size", 3))
    train_size = split_cfg.get("train_size")

    test_paths = paths[:test_size]
    val_paths = paths[test_size:test_size + val_size]
    train_paths = paths[test_size + val_size:]
    if train_size is not None:
        train_paths = train_paths[:int(train_size)]

    if not train_paths:
        raise ValueError(
            "No training files remain after applying "
            f"test_size={test_size}, val_size={val_size}, and train_size={train_size}"
        )

    return train_paths, val_paths, test_paths


def build_wav_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    source = cfg.get("source", "wav")
    if source not in {"wav", "precomputed"}:
        raise ValueError(f"Unsupported data.source={source!r}. Expected 'wav' or 'precomputed'.")

    if source == "wav":
        paths = find_wav_files(cfg.root_dir)
        if not paths:
            raise ValueError(f"No WAV files found under data.root_dir={cfg.root_dir!r}")
        dataset_cls = WavDataset
        dataset_kwargs = _as_container(cfg.get("dataset", {}))
    else:
        paths = sorted(Path(cfg.processed_dir).rglob("*.pt"))
        if not paths:
            raise ValueError(f"No precomputed cache files found under data.processed_dir={cfg.processed_dir!r}")
        dataset_cls = PrecomputedWavDataset
        dataset_kwargs = {}

    train_paths, val_paths, test_paths = _split_paths(paths, cfg)
    loader_kwargs = _as_container(cfg.get("loader", {}))

    train_dataset = dataset_cls(train_paths, **dataset_kwargs)
    val_dataset = dataset_cls(val_paths, **dataset_kwargs)
    test_dataset = dataset_cls(test_paths, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
