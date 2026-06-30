import math
import os
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Tuple, List

import librosa
import torch
from scipy.ndimage import binary_dilation, label, find_objects
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pyworld as pw


WAV_CACHE_VERSION = 1


class WavDataset(Dataset):
    def __init__(
            self,
            wav_paths: List[str] | List[Path],
            sample_rate: int = 16000,
            frame_period: float = 4.0, # in ms
            sample_duration: float = 4.0, # in seconds
            split_margin: float = 0.0, # in seconds
            overlap_ratio: float = 0.0,
            cache_dir: str | Path | None = None,
            load_cache: bool = True,
            save_cache: bool = True,
            rebuild_cache: bool = False,
    ):
        self.wav_paths = wav_paths
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.sample_duration = sample_duration
        self.split_margin = split_margin
        self.overlap_ratio = overlap_ratio
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.rebuild_cache = rebuild_cache

        self.cache = []

        for wav_path in tqdm(wav_paths):
            self.cache += self.load_or_collect_samples(wav_path)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache[idx]

    def load_or_collect_samples(self, wav_path: str | Path) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        cache_path = self.get_cache_path(wav_path)
        metadata = self.get_cache_metadata(wav_path)

        if cache_path is not None and self.load_cache and not self.rebuild_cache:
            samples = self.load_samples_from_cache(cache_path, metadata)
            if samples is not None:
                return samples

        features = self.collect_features(wav_path)
        samples = [
            (
                torch.as_tensor(y, dtype=torch.float32),
                torch.as_tensor(f0, dtype=torch.float32),
            )
            for y, f0 in self.split_features(*features)
        ]

        if cache_path is not None and self.save_cache:
            self.save_samples_to_cache(cache_path, metadata, samples)

        return samples

    def get_cache_metadata(self, wav_path: str | Path) -> dict[str, Any]:
        wav_path = Path(wav_path)
        stat = wav_path.stat()
        return {
            "version": WAV_CACHE_VERSION,
            "wav_path": str(wav_path.resolve()),
            "wav_size": stat.st_size,
            "wav_mtime_ns": stat.st_mtime_ns,
            "sample_rate": self.sample_rate,
            "frame_period": self.frame_period,
            "sample_duration": self.sample_duration,
            "split_margin": self.split_margin,
            "overlap_ratio": self.overlap_ratio,
        }

    def get_cache_path(self, wav_path: str | Path) -> Path | None:
        if self.cache_dir is None:
            return None

        metadata = self.get_cache_metadata(wav_path)
        cache_key = json.dumps(metadata, sort_keys=True).encode("utf-8")
        digest = sha256(cache_key).hexdigest()[:16]
        stem = Path(wav_path).stem
        return self.cache_dir / f"{stem}-{digest}.pt"

    def load_samples_from_cache(
            self,
            cache_path: Path,
            metadata: dict[str, Any],
    ) -> list[Tuple[torch.Tensor, torch.Tensor]] | None:
        if not cache_path.exists():
            return None

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError, ValueError):
            return None

        if not isinstance(payload, dict):
            return None

        if payload.get("metadata") != metadata:
            return None

        return payload["samples"]

    def save_samples_to_cache(
            self,
            cache_path: Path,
            metadata: dict[str, Any],
            samples: list[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"metadata": metadata, "samples": samples}, tmp_path)
        tmp_path.replace(cache_path)

    def collect_features(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sample_rate)
        f0, _, _ = pw.wav2world(y.astype(np.double), self.sample_rate, frame_period=self.frame_period)
        return y, f0

    def split_features(self, y, f0):
        data_points_per_sample = self.sample_duration * self.sample_rate
        frames_per_sample = 1000 * self.sample_duration / self.frame_period
        data_points_per_frame = int(self.frame_period * self.sample_rate / 1000)

        vuv = f0 > 0
        dilated_vuv = binary_dilation(vuv, iterations=int(1000 * self.split_margin / self.frame_period))
        labeled_vuv, num_regions = label(dilated_vuv)

        for (region_slice,) in find_objects(labeled_vuv):
            if region_slice is None:
                continue

            start_frame = region_slice.start
            end_frame = region_slice.stop

            step_size_frames = int(frames_per_sample * (1 - self.overlap_ratio))
            num_sub_regions = math.ceil((end_frame - start_frame) / step_size_frames)

            for sub_region_index in range(num_sub_regions):
                start_frame_sub = int(start_frame + sub_region_index * step_size_frames)
                end_frame_sub = int(start_frame_sub + frames_per_sample)
                start_sample_sub = int(start_frame_sub * data_points_per_frame)
                end_sample_sub = int(end_frame_sub * data_points_per_frame)

                y_sub = y[start_sample_sub:end_sample_sub]
                f0_sub = f0[start_frame_sub:end_frame_sub]
                vuv_sub = vuv[start_frame_sub:end_frame_sub]

                if vuv_sub.sum() < 0.3 * vuv_sub.size:
                    continue

                if len(y_sub) < data_points_per_sample:
                    d = int(data_points_per_sample - len(y_sub))
                    y_sub = np.pad(y_sub, (0, d), "constant")

                if len(f0_sub) < frames_per_sample:
                    d = int(frames_per_sample - len(f0_sub))
                    f0_sub = np.pad(f0_sub, (0, d), "constant")

                yield y_sub, f0_sub


class MusicXMLDataset(Dataset):
    def __init__(
        self,
        paths: List[str] | List[Path]
    ):
        self.paths = paths
        self.cache = []

        for path in self.paths:
            self.cache += self.collect_features(path)

    def collect_features(self, path: str | Path):
        pass


class USTDataset(Dataset):
    def __init__(
            self,
            paths: List[str] | List[Path],
    ):
        self.paths = paths

    def collect_features(self, path: str | Path):
        pass
