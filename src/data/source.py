import math
from pathlib import Path
from typing import Tuple, List

import librosa
import torch
from scipy.ndimage import binary_dilation, label, find_objects
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pyworld as pw


class WavDataset(Dataset):
    def __init__(
            self,
            wav_paths: List[str] | List[Path],
            sample_rate: int = 16000,
            frame_period: float = 4.0, # in ms
            sample_duration: float = 4.0, # in seconds
            split_margin: float = 0.0, # in seconds
            overlap_ratio: float = 0.0,
    ):
        self.wav_paths = wav_paths
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.sample_duration = sample_duration
        self.split_margin = split_margin
        self.overlap_ratio = overlap_ratio

        self.cache = []

        for wav_path in tqdm(wav_paths):
            features = self.collect_features(wav_path)
            self.cache += list(self.split_features(*features))

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache[idx]

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


if __name__ == '__main__':
    paths = [
        '/home/ewigeki/MyProjects/ewigeki/DDSP-WORLD-vocoder/data/「波音リツ」歌声データベースVer2.0.2/DATABASE/sacrifice/sacrifice.musicxml',
    ]

    dataset = MusicXMLDataset(paths)
