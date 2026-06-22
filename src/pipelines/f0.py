from pathlib import Path
from typing import Any

import librosa
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

from src.data.source import WavDataset
from src.models.crepe import get_target, logits_to_cent, cent_to_hz, Crepe
from src.utils import find_wav_files


class F0Pipeline(L.LightningModule):
    def __init__(self, cent_predictor: nn.Module):
        super().__init__()

        self.cent_predictor = cent_predictor
        self.sample_rate = 16000

    def _shared_step(self, batch, batch_idx):
        y, f0 = batch
        cent_pred_logits = self.cent_predictor(y)
        loss = F.binary_cross_entropy_with_logits(cent_pred_logits, get_target(f0[:, ::4]))
        return loss, y, f0, cent_pred_logits

    def _log_f0_plot(self, f0, pred_f0, stage="val"):
        f0 = f0[0].detach().cpu()
        pred_f0 = pred_f0[0].detach().cpu()

        fig = plt.figure(figsize=(10, 4))
        plt.plot(f0.numpy(), label="target f0")
        plt.plot(pred_f0.numpy(), label="pred f0")
        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("F0 cents")
        plt.title(f"{stage} F0 prediction")

        logger = self.logger

        if hasattr(logger, "experiment"):
            logger.experiment.add_figure(
                f"{stage}/f0_prediction",
                fig,
                global_step=self.global_step,
            )

        plt.close(fig)

    def _log_stft_plot(self, audio, pred_audio=None, stage="val"):
        audio = audio[0].detach().cpu()

        if pred_audio is not None:
            pred_audio = pred_audio[0].detach().cpu()

        def compute_spectrogram(x):
            # x: [T]
            if x.ndim > 1:
                x = x.squeeze()

            n_fft = 1024
            hop_length = 256
            win_length = 1024

            window = torch.hann_window(win_length)

            stft = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
            )

            spec = torch.abs(stft)
            spec_db = 20 * torch.log10(spec + 1e-8)

            return spec_db

        target_spec = compute_spectrogram(audio)

        if pred_audio is not None:
            pred_spec = compute_spectrogram(pred_audio)

            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            axes[0].imshow(
                target_spec.numpy(),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            axes[0].set_title("Target STFT")
            axes[0].set_ylabel("Frequency bin")

            axes[1].imshow(
                pred_spec.numpy(),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            axes[1].set_title("Predicted STFT")
            axes[1].set_xlabel("Frame")
            axes[1].set_ylabel("Frequency bin")

            fig.suptitle(f"{stage} STFT comparison")

        else:
            fig = plt.figure(figsize=(10, 4))

            plt.imshow(
                target_spec.numpy(),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            plt.colorbar(label="Magnitude dB")
            plt.xlabel("Frame")
            plt.ylabel("Frequency bin")
            plt.title(f"{stage} STFT")

        logger = self.logger

        if hasattr(logger, "experiment"):
            logger.experiment.add_figure(
                f"{stage}/stft",
                fig,
                global_step=self.global_step,
            )

        plt.close(fig)

    def _log_cent_logits_plot(
            self,
            cent_pred_logits: torch.Tensor,
            target: torch.Tensor | None = None,
            stage: str = "val",
    ):
        """
        cent_pred_logits:
            shape (B, C, T) or (B, C)

        target:
            optional, shape (B, C, T) or (B, C)
        """

        if cent_pred_logits.ndim == 2:
            # (B, C) -> (B, C, 1)
            cent_pred_logits = cent_pred_logits.unsqueeze(-1)

        probs = torch.sigmoid(cent_pred_logits[0]).detach().cpu()
        # probs: (C, T)

        fig = plt.figure(figsize=(12, 5))

        plt.imshow(
            probs.numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )

        plt.colorbar(label="sigmoid(logit)")
        plt.xlabel("Frame")
        plt.ylabel("Cent bin")
        plt.title(f"{stage} cent prediction probabilities")

        if target is not None:
            if target.ndim == 2:
                target = target.unsqueeze(-1)

            target_0 = target[0].detach().cpu()  # (C, T)

            # Overlay target argmax per frame
            target_bins = target_0.argmax(dim=0)
            frames = torch.arange(target_bins.shape[0])

            plt.plot(
                frames.numpy(),
                target_bins.numpy(),
                linewidth=1.5,
                label="target bin",
            )
            plt.legend()

        logger = self.logger

        if hasattr(logger, "experiment"):
            logger.experiment.add_figure(
                f"{stage}/cent_pred_logits",
                fig,
                global_step=self.global_step,
            )

        plt.close(fig)

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, f0, cent_pred_logits = self._shared_step(batch, batch_idx)

        self.log("val/loss", loss)
        if batch_idx == 0:
            f0_pred = cent_to_hz(logits_to_cent(cent_pred_logits))
            probs = F.sigmoid(cent_pred_logits)
            confidence, indices = torch.max(probs, dim=2)
            voiced = confidence >= 0.3
            f0_pred = f0_pred * voiced
            self._log_f0_plot(f0[:, ::4], f0_pred)

            self._log_cent_logits_plot(
                cent_pred_logits=cent_pred_logits.permute(0, 2, 1),
                stage="val",
                batch_idx=batch_idx,
            )

    def forward(self, y, threshold = 0.2, output_logits=False) -> Any:
        cent_pred_logits = self.cent_predictor(y)
        probs = F.sigmoid(cent_pred_logits)
        confidence, indices = torch.max(probs, dim=2)
        voiced = confidence >= threshold
        f0 = cent_to_hz(logits_to_cent(cent_pred_logits))
        f0 = f0 * voiced
        if output_logits:
            return f0, cent_pred_logits
        return f0

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer


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
