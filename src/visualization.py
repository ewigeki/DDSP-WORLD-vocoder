import torch
import lightning as L
from matplotlib import pyplot as plt


class AudioVisualizationLogger:
    def __init__(
            self,
            sample_rate: int = 16000,
            max_seconds: float = 10.0,
            n_fft: int = 1024,
            hop_length: int = 256,
            win_length: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    @staticmethod
    def _has_tensorboard_logger(pl_module: L.LightningModule) -> bool:
        return hasattr(pl_module.logger, "experiment")

    def _prepare_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to [T] CPU tensor.

        Supports:
            [B, T]
            [B, 1, T]
            [1, T]
            [T]
        """
        audio = audio.detach().cpu()

        if audio.ndim == 3:
            audio = audio[0]  # [B, C, T] -> [C, T]

        if audio.ndim == 2:
            audio = audio[0]  # [C, T] or [B, T] -> [T]

        audio = audio.squeeze()

        max_len = int(self.sample_rate * self.max_seconds)
        audio = audio[:max_len]

        max_abs = audio.abs().max()
        if max_abs > 1.0:
            audio = audio / (max_abs + 1e-8)

        return audio

    def _compute_stft_db(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: [T]
        returns: [freq_bins, frames]
        """
        window = torch.hann_window(self.win_length)

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )

        spec = torch.abs(stft)
        spec_db = 20 * torch.log10(spec + 1e-8)

        return spec_db

    def log_audio(
            self,
            pl_module: L.LightningModule,
            audio: torch.Tensor,
            name: str,
            stage: str = "val",
    ):
        if not self._has_tensorboard_logger(pl_module):
            return

        audio = self._prepare_audio(audio)

        pl_module.logger.experiment.add_audio(
            tag=f"{stage}/audio/{name}",
            snd_tensor=audio,
            global_step=pl_module.global_step,
            sample_rate=self.sample_rate,
        )

    def log_stft(
            self,
            pl_module: L.LightningModule,
            audio: torch.Tensor,
            name: str,
            stage: str = "val",
    ):
        if not self._has_tensorboard_logger(pl_module):
            return

        audio = self._prepare_audio(audio)
        spec_db = self._compute_stft_db(audio)

        self.log_image(
            pl_module=pl_module,
            image=spec_db,
            name=name,
            stage=f"{stage}/stft",
            title=f"{stage} STFT: {name}",
            colorbar_label="Magnitude dB",
            xlabel="Frame",
            ylabel="Frequency bin",
        )

    def log_audio_and_stft(
            self,
            pl_module: L.LightningModule,
            audio: torch.Tensor,
            name: str,
            stage: str = "val",
    ):
        self.log_audio(
            pl_module=pl_module,
            audio=audio,
            name=name,
            stage=stage,
        )

        self.log_stft(
            pl_module=pl_module,
            audio=audio,
            name=name,
            stage=stage,
        )

    def log_f0_plot(
            self,
            pl_module: L.LightningModule,
            f0: torch.Tensor,
            pred_f0: torch.Tensor,
            stage: str = "val",
            name: str = "f0_prediction",
    ):
        if not self._has_tensorboard_logger(pl_module):
            return

        f0 = f0[0].detach().cpu()
        pred_f0 = pred_f0[0].detach().cpu()

        fig = plt.figure(figsize=(10, 4))

        plt.plot(f0.numpy(), label="target f0")
        plt.plot(pred_f0.numpy(), label="pred f0")

        plt.legend()
        plt.xlabel("Frame")
        plt.ylabel("F0 Hz")
        plt.title(f"{stage} F0 prediction")
        plt.tight_layout()

        pl_module.logger.experiment.add_figure(
            tag=f"{stage}/{name}",
            figure=fig,
            global_step=pl_module.global_step,
        )

        plt.close(fig)

    def log_image(
            self,
            pl_module: L.LightningModule,
            image: torch.Tensor,
            name: str,
            stage: str,
            title: str | None = None,
            colorbar_label: str = "Value",
            xlabel: str = "Frame",
            ylabel: str = "Bin",
    ):
        if not self._has_logger(pl_module):
            return

        image = image.detach().cpu().squeeze()

        fig = plt.figure(figsize=(10, 4))

        plt.imshow(
            image.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        plt.colorbar(label=colorbar_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or f"{stage}: {name}")
        plt.tight_layout()

        pl_module.logger.experiment.add_figure(
            tag=f"{stage}/{name}",
            figure=fig,
            global_step=pl_module.global_step,
        )

        plt.close(fig)

    def log_cent_logits_plot(
            self,
            pl_module: L.LightningModule,
            cent_pred_logits: torch.Tensor,
            target: torch.Tensor | None = None,
            stage: str = "val",
            name: str = "cent_pred_logits",
    ):
        """
        Expected cent_pred_logits shape:
            [B, T, C]

        Expected target shape:
            [B, T, C] or None
        """
        if not self._has_tensorboard_logger(pl_module):
            return

        if cent_pred_logits.ndim != 3:
            raise ValueError(
                f"Expected cent_pred_logits with shape [B, T, C], "
                f"got {tuple(cent_pred_logits.shape)}"
            )

        # [B, T, C] -> [C, T] for image
        probs = torch.sigmoid(cent_pred_logits[0]).detach().cpu()
        probs = probs.transpose(0, 1)

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
            if target.ndim != 3:
                raise ValueError(
                    f"Expected target with shape [B, T, C], "
                    f"got {tuple(target.shape)}"
                )

            target_0 = target[0].detach().cpu()  # [T, C]
            target_bins = target_0.argmax(dim=-1)  # [T]
            frames = torch.arange(target_bins.shape[0])

            plt.plot(
                frames.numpy(),
                target_bins.numpy(),
                linewidth=1.5,
                label="target bin",
            )

            plt.legend()

        plt.tight_layout()

        pl_module.logger.experiment.add_figure(
            tag=f"{stage}/{name}",
            figure=fig,
            global_step=pl_module.global_step,
        )

        plt.close(fig)