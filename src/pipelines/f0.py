from typing import Any

import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F

from src.models.crepe import get_target, logits_to_cent, cent_to_hz
from src.visualization import AudioVisualizationLogger


class F0Pipeline(L.LightningModule):
    def __init__(self, cent_predictor: nn.Module):
        super().__init__()
        self.cent_predictor = cent_predictor

        self.visualizer = AudioVisualizationLogger(
            sample_rate=16000,
            max_seconds=10.0,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )

    def _shared_step(self, batch, batch_idx):
        y, f0 = batch
        cent_pred_logits = self.cent_predictor(y)
        loss = F.binary_cross_entropy_with_logits(cent_pred_logits, get_target(f0[:, ::4]))
        return loss, cent_pred_logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y, f0 = batch
        loss, cent_pred_logits = self._shared_step(batch, batch_idx)

        self.log("val/loss", loss)
        if batch_idx == 0:
            f0_target = f0[:, ::4]

            f0_pred = self._logits_to_voiced_f0(
                cent_pred_logits,
                threshold=0.3,
            )

            self.visualizer.log_f0_plot(
                pl_module=self,
                f0=f0_target,
                pred_f0=f0_pred,
                stage="val",
                name="f0_prediction",
            )

            self.visualizer.log_cent_logits_plot(
                pl_module=self,
                cent_pred_logits=cent_pred_logits,
                target=None,
                stage="val",
                name="cent_pred_logits",
            )

            self.visualizer.log_audio_and_stft(
                pl_module=self,
                audio=y,
                name="input_audio",
                stage="val",
            )

    def _logits_to_voiced_f0(
        self,
        cent_pred_logits: torch.Tensor,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        cent_pred_logits: [B, T, C]
        returns f0: [B, T]
        """
        probs = torch.sigmoid(cent_pred_logits)

        confidence, _ = torch.max(probs, dim=-1)  # [B, T]
        voiced = confidence >= threshold

        f0 = cent_to_hz(logits_to_cent(cent_pred_logits))  # [B, T]
        f0 = f0 * voiced

        return f0

    def forward(
        self,
        y,
        threshold: float = 0.2,
        output_logits: bool = False,
    ) -> Any:
        cent_pred_logits = self.cent_predictor(y)

        f0 = self._logits_to_voiced_f0(
            cent_pred_logits,
            threshold=threshold,
        )

        if output_logits:
            return f0, cent_pred_logits

        return f0

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer
