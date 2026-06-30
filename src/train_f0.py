from pathlib import Path

import librosa
import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch.functional as F

from src.data.source import WavDataset
from src.models.crepe import Crepe
from src.pipelines.f0 import F0Pipeline
from src.utils import find_wav_files


def read_paths_from_tsv(tsv_path: str | Path, base_dir: str | Path | None = None) -> list[Path]:
    tsv_path = Path(tsv_path)

    df = pd.read_csv(tsv_path, sep="\t")

    if "path" not in df.columns:
        raise ValueError(f"Column 'path' not found. Available columns: {list(df.columns)}")

    paths = df["path"].dropna().astype(str).tolist()

    if base_dir is not None:
        base_dir = Path(base_dir)
        paths = [base_dir / p for p in paths]
    else:
        paths = [Path(p) for p in paths]

    return paths


# TODO: create separate interface file

DATASET_ROOT_DIR = ""
CHECKPOINT_PATH = ""
SAMPLE_AUDIO_FILE = ""


def train_pipeline():
    cent_predictor = Crepe()
    model = F0Pipeline(cent_predictor)
    dataset = find_wav_files(DATASET_ROOT_DIR)
    train_dataset = WavDataset(dataset[6:])
    val_dataset = WavDataset(dataset[3:6])
    test_dataset = WavDataset(dataset[:3])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=12)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch:03d}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,
        min_delta=1e-4,
    )

    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def inference_pipeline():
    cent_predictor = Crepe()
    device = torch.device("cuda")
    model = F0Pipeline.load_from_checkpoint(CHECKPOINT_PATH, device, weights_only=True, cent_predictor=cent_predictor)
    y, sr = librosa.load(SAMPLE_AUDIO_FILE, sr=16000)
    y = torch.tensor(y).reshape(1, -1)
    print(y[:, 480000:480000+64000])
    f0 = model(y[:, 480000:480000+64000].to(device))
    print(f0[0, :30])
    f0 = model(y[:, 480000:480000 + 16384].to(device))
    print(f0[0, :30])
    f0 = model(F.pad(y[:, 480000:480000 + 16384], (0, 64000-16384)).to(device))
    print(f0[0, :30])
    f0 = model(y[:, 480000:480000 + 8192].to(device))
    print(f0[0, :30])
    f0 = model(y[:, 480000:480000 + 4096].to(device))
    print(f0[0, :30])


if __name__ == "__main__":
    train_pipeline()