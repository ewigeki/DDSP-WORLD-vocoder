import lightning as L
import torch
import torch.nn.functional as F
from torch import optim

from src.models.discriminators import (
    Discriminator,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from src.visualization import AudioVisualizationLogger


class WORLD(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        f0_predictor,
        vocoder,
        sample_rate,
        adversarial_loss_weight: float = 1.0,
        feature_matching_loss_weight: float = 10.0,
        discriminator_start_step: int = 0,
        stft_loss_configs=None,
        generator_lr: float = 1e-4,
        discriminator_lr: float = 1e-5,
    ):
        super(WORLD, self).__init__()

        self.automatic_optimization = False
        self.encoder = encoder
        self.decoder = decoder
        self.f0_predictor = f0_predictor.freeze()
        self.vocoder = vocoder
        self.discriminator = Discriminator(
            MultiPeriodDiscriminator(),
            MultiResolutionDiscriminator(),
        )
        self.sample_rate = sample_rate
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_matching_loss_weight = feature_matching_loss_weight
        self.discriminator_start_step = discriminator_start_step
        self.stft_loss_configs = self._normalize_stft_loss_configs(stft_loss_configs)
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.visualizer = AudioVisualizationLogger(
            sample_rate=sample_rate,
            max_seconds=10.0,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )

    @staticmethod
    def _normalize_stft_loss_configs(stft_loss_configs):
        if stft_loss_configs is None:
            stft_loss_configs = [
                (64, 16, 64, 1.0),
                (128, 32, 128, 1.0),
                (256, 64, 256, 1.0),
                (512, 128, 512, 1.0),
                (1024, 256, 1024, 1.0),
                (2048, 512, 2048, 1.0),
            ]

        normalized_configs = []
        for config in stft_loss_configs:
            if len(config) == 3:
                n_fft, hop_length, win_length = config
                weight = 1.0
            elif len(config) == 4:
                n_fft, hop_length, win_length, weight = config
            else:
                raise ValueError("STFT loss configs must be (n_fft, hop_length, win_length[, weight])")

            if weight < 0.0:
                raise ValueError("STFT loss config weights must be non-negative")

            normalized_configs.append((int(n_fft), int(hop_length), int(win_length), float(weight)))

        if sum(weight for *_, weight in normalized_configs) <= 0.0:
            raise ValueError("At least one STFT loss config must have a positive weight")

        return normalized_configs

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

        total_loss = 0.0
        eps = 1e-7

        for n_fft, hop_length, win_length, weight in self.stft_loss_configs:
            if weight == 0.0:
                continue

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
            diff = true_mag - pred_mag

            sc_loss = (
                    torch.linalg.norm(diff, ord="fro", dim=(-2, -1))
                    / (torch.linalg.norm(true_mag, ord="fro", dim=(-2, -1)) + eps)
            ).mean()

            # Log magnitude loss
            log_mag_loss = F.l1_loss(
                torch.log(true_mag + eps),
                torch.log(pred_mag + eps),
            )

            total_loss = total_loss + weight * (sc_loss + log_mag_loss)

        return total_loss

    def _adversarial_loss(self, y_true, y_pred):
        y_real_scores, y_fake_scores, f_map_real, f_map_fake = self.discriminator(y_true, y_pred)

        real_loss = sum(
            torch.mean((score - 1.0) ** 2)
            for score in y_real_scores
        ) / len(y_real_scores)
        fake_loss = sum(
            torch.mean(score ** 2)
            for score in y_fake_scores
        ) / len(y_fake_scores)
        generator_loss = sum(
            torch.mean((score - 1.0) ** 2)
            for score in y_fake_scores
        ) / len(y_fake_scores)
        discriminator_loss = 0.5 * (real_loss + fake_loss)
        feature_matching_loss = sum(
            F.l1_loss(fake_feature, real_feature.detach())
            for real_features, fake_features in zip(f_map_real, f_map_fake)
            for real_feature, fake_feature in zip(real_features, fake_features)
        ) / sum(len(features) for features in f_map_real)

        return generator_loss, discriminator_loss, feature_matching_loss

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
        f0_predicted = self.f0_predictor(y).detach().unsqueeze(1)
        f0_predicted = F.upsample(f0_predicted, size=1000, mode="linear", align_corners=False)
        sp, ap = self.decoder(f0_predicted.permute(0, 2, 1), z.permute(0, 2, 1))
        y_p = self.vocoder(f0_predicted, sp, ap)

        loss = self._m_sftf_loss(y, y_p)
        return loss, (sp, ap, y_p)

    def training_step(self, batch, batch_idx):
        generator_optimizer, discriminator_optimizer = self.optimizers()
        y, _ = batch

        reconstruction_loss, prediction = self._shared_step(batch, batch_idx)
        _, _, y_p = prediction

        use_discriminator = self.global_step >= self.discriminator_start_step

        if not use_discriminator:
            self.toggle_optimizer(generator_optimizer)
            generator_optimizer.zero_grad()
            self.manual_backward(reconstruction_loss)
            generator_optimizer.step()
            self.untoggle_optimizer(generator_optimizer)

            self.log("train/loss", reconstruction_loss)
            self.log("train/reconstruction_loss", reconstruction_loss)
            self.log("train/use_discriminator", 0.0)
            return

        self.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()
        _, discriminator_loss, _ = self._adversarial_loss(y, y_p.detach())
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        self.untoggle_optimizer(discriminator_optimizer)

        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()
        adversarial_loss, _, feature_matching_loss = self._adversarial_loss(y, y_p)
        generator_loss = (
            reconstruction_loss
            + self.adversarial_loss_weight * adversarial_loss
            + self.feature_matching_loss_weight * feature_matching_loss
        )
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)

        self.log("train/loss", generator_loss)
        self.log("train/reconstruction_loss", reconstruction_loss)
        self.log("train/adversarial_loss", adversarial_loss)
        self.log("train/feature_matching_loss", feature_matching_loss)
        self.log("train/discriminator_loss", discriminator_loss)
        self.log("train/use_discriminator", 1.0)

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

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def autocode(self, x):
        z = self.encoder(x)
        f0_predicted = self.f0_predictor(x).detach().unsqueeze(1)
        f0_predicted = F.upsample(f0_predicted, size=1000, mode="linear", align_corners=False)
        sp, ap = self.decoder(f0_predicted.permute(0, 2, 1), z.permute(0, 2, 1))
        y = self.vocoder(f0_predicted, sp, ap)
        return y

    def configure_optimizers(self):
        generator_optimizer = optim.AdamW(
            [
                *self.encoder.parameters(),
                *self.decoder.parameters(),
                *self.vocoder.parameters(),
            ],
            lr=self.generator_lr,
        )

        discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
        )

        return generator_optimizer, discriminator_optimizer
