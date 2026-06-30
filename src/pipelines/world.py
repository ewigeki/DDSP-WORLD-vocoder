from pathlib import Path

import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from src.data.source import WavDataset
from src.models.crepe import Crepe
from src.models.decoder import EmformerDecoder
from src.models.encoder import ZEncoder
from src.models.vocoder import WORLDVocoder
from src.pipelines.f0 import F0Pipeline
from src.utils import find_wav_files
from src.visualization import AudioVisualizationLogger


class WORLD(L.LightningModule):
    def __init__(self, encoder, decoder, f0_predictor, vocoder, sample_rate):
        super(WORLD, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.f0_predictor = f0_predictor.freeze()
        self.vocoder = vocoder
        self.sample_rate = sample_rate

        self.visualizer = AudioVisualizationLogger(
            sample_rate=sample_rate,
            max_seconds=10.0,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )

    def _m_sftf_loss(self, y_true, y_pred):
        """
        Multi-scale STFT loss:
        L = spectral_convergence + log_magnitude_loss

        y_true, y_pred: [B, T] or [B, 1, T]
        """

        if y_true.dim() == 3:
            y_true = y_true.squeeze(1)

        if y_pred.dim() == 3:
            y_pred = y_pred.squeeze(1)

        stft_configs = [
            (64, 16, 64),
            (128, 32, 128),
            (256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ]

        total_loss = 0.0
        eps = 1e-7

        for n_fft, hop_length, win_length in stft_configs:
            window = torch.hann_window(win_length, device=y_true.device)

            true_stft = torch.stft(
                y_true,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=True,
            )

            pred_stft = torch.stft(
                y_pred,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True,
                center=True,
            )

            true_mag = torch.abs(true_stft)
            pred_mag = torch.abs(pred_stft)

            # Spectral convergence
            sc_loss = (
                    torch.linalg.norm(true_mag - pred_mag, ord="fro")
                    / (torch.linalg.norm(true_mag, ord="fro") + eps)
            )

            # Log magnitude loss
            log_mag_loss = F.l1_loss(
                torch.log(true_mag + eps),
                torch.log(pred_mag + eps),
            )

            total_loss = total_loss + sc_loss + log_mag_loss

        return total_loss / len(stft_configs)

    def _adversarial_loss(self, y_true, y_pred):
        pass

    def _visualize(
        self,
        y, y_pred, sp, ap,
        stage: str = "val",
    ):
        self.visualizer.log_audio_and_stft(
            pl_module=self,
            audio=y,
            name="target",
            stage=stage,
        )

        self.visualizer.log_audio_and_stft(
            pl_module=self,
            audio=y_pred,
            name="predicted",
            stage=stage,
        )

        self.visualizer.log_image(
            pl_module=self,
            image=sp[0],
            name="sp",
            stage=f"{stage}/world",
            title=f"{stage} spectral envelope",
            colorbar_label="SP",
            xlabel="Frame",
            ylabel="Frequency bin",
        )

        self.visualizer.log_image(
            pl_module=self,
            image=ap[0],
            name="ap",
            stage=f"{stage}/world",
            title=f"{stage} aperiodicity",
            colorbar_label="AP",
            xlabel="Frame",
            ylabel="Frequency bin",
        )

    def _shared_step(self, batch, batch_idx):
        y, _ = batch

        z = self.encoder(y)
        f0_predicted = self.f0_predictor(y).detach()
        sp, ap = self.decoder(f0_predicted, z)
        y_p = self.vocoder(f0_predicted, sp, ap)

        loss = self._m_sftf_loss(y, y_p)
        return loss, (sp, ap, y_p)

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y, f0 = batch
        loss, prediction = self._shared_step(batch, batch_idx)
        sp, ap, y_p = prediction
        self.log("val/loss", loss)
        if batch_idx == 0:
            self._visualize(y, y_p, sp, ap)

    def test_step(self, batch, batch_idx):
        loss, prediction = self._shared_step(batch, batch_idx)
        self.log("test/loss", loss)


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

