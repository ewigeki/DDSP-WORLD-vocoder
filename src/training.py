from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.source import WavDataset
from src.utils import find_wav_files


def _as_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)


def build_wav_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    wav_paths = find_wav_files(cfg.root_dir)
    if not wav_paths:
        raise ValueError(f"No WAV files found under data.root_dir={cfg.root_dir!r}")

    split_cfg = cfg.get("split", {})
    val_size = int(split_cfg.get("val_size", 3))
    test_size = int(split_cfg.get("test_size", 3))

    test_paths = wav_paths[:test_size]
    val_paths = wav_paths[test_size:test_size + val_size]
    train_paths = wav_paths[test_size + val_size:]

    if not train_paths:
        raise ValueError(
            "No training WAV files remain after applying "
            f"test_size={test_size} and val_size={val_size}"
        )

    dataset_kwargs = _as_container(cfg.get("dataset", {}))
    loader_kwargs = _as_container(cfg.get("loader", {}))

    train_dataset = WavDataset(train_paths, **dataset_kwargs)
    val_dataset = WavDataset(val_paths, **dataset_kwargs)
    test_dataset = WavDataset(test_paths, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
