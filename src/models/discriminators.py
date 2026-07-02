from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


DiscriminatorNorm = Literal["weight", "spectral", "none"]


def _apply_norm(module: nn.Module, norm: DiscriminatorNorm) -> nn.Module:
    if norm == "weight":
        return weight_norm(module)
    if norm == "spectral":
        return spectral_norm(module)
    if norm == "none":
        return module
    raise ValueError(f"Unsupported discriminator norm: {norm}")


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        norm: DiscriminatorNorm = "weight",
    ):
        super(PeriodDiscriminator, self).__init__()

        self.period: int = period

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(1, 64, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
                nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
                nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
                nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
                nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=((kernel_size - 1) // 2, 0)),
            ]
        )

        self.post_conv = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.apply(lambda module: _apply_norm(module, norm) if isinstance(module, nn.Conv2d) else None)

    def forward(self, x):
        f_map = []

        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, channel, time = x.shape

        if time % self.period != 0:
            pad_len = self.period - (time % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            time = time + pad_len

        x = x.view(batch, channel, time // self.period, self.period)

        for conv in self.conv:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            f_map.append(x)

        x = self.post_conv(x)
        f_map.append(x)

        score = torch.flatten(x, start_dim=1)

        return score, f_map


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), norm: DiscriminatorNorm = "weight"):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods

        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period, norm=norm) for period in self.periods]
        )

    def forward(self, y_true, y_pred):
        y_real_scores = []
        y_fake_scores = []
        f_map_real = []
        f_map_fake = []

        for discriminator in self.discriminators:
            real_score, f_map_real_ = discriminator(y_true)
            fake_score, f_map_fake_ = discriminator(y_pred)

            y_real_scores.append(real_score)
            y_fake_scores.append(fake_score)
            f_map_real.append(f_map_real_)
            f_map_fake.append(f_map_fake_)

        return y_real_scores, y_fake_scores, f_map_real, f_map_fake


class ScaleDiscriminator(nn.Module):
    def __init__(self, norm: DiscriminatorNorm = "weight"):
        super(ScaleDiscriminator, self).__init__()

        self.conv = nn.ModuleList(
            [
                nn.Conv1d(1, 16, 15, 1, padding=7),
                nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
                nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
                nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
                nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
                nn.Conv1d(1024, 1024, 5, 1, padding=2),
            ]
        )

        self.post_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)
        self.apply(lambda module: _apply_norm(module, norm) if isinstance(module, nn.Conv1d) else None)

    def forward(self, x):
        f_map = []

        if x.dim() == 2:
            x = x.unsqueeze(1)

        for conv in self.conv:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            f_map.append(x)

        x = self.post_conv(x)
        f_map.append(x)

        score = torch.flatten(x, start_dim=1)

        return score, f_map


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales: int = 3, norm: DiscriminatorNorm = "weight"):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(norm=norm) for _ in range(scales)]
        )
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, y_true, y_pred):
        y_real_scores = []
        y_fake_scores = []
        f_map_real = []
        f_map_fake = []

        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(1)

        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(1)

        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                y_true = self.pooling(y_true)
                y_pred = self.pooling(y_pred)

            real_score, f_map_real_ = discriminator(y_true)
            fake_score, f_map_fake_ = discriminator(y_pred)

            y_real_scores.append(real_score)
            y_fake_scores.append(fake_score)
            f_map_real.append(f_map_real_)
            f_map_fake.append(f_map_fake_)

        return y_real_scores, y_fake_scores, f_map_real, f_map_fake


class SpectralDiscriminator(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        norm: DiscriminatorNorm = "weight",
    ):
        super(SpectralDiscriminator, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(1, 32, (3, 9), padding=(1, 4)),
                nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.Conv2d(32, 32, (3, 3), padding=1),
            ]
        )

        self.post_conv = nn.Conv2d(32, 1, (3, 3), padding=1)
        self.apply(lambda module: _apply_norm(module, norm) if isinstance(module, nn.Conv2d) else None)

    def _magnitude_spectrogram(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        x = torch.abs(x)
        return x.unsqueeze(1)

    def forward(self, x):
        f_map = []
        x = self._magnitude_spectrogram(x)

        for conv in self.conv:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            f_map.append(x)

        x = self.post_conv(x)
        f_map.append(x)

        score = torch.flatten(x, start_dim=1)

        return score, f_map


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
        norm: DiscriminatorNorm = "weight",
    ):
        super(MultiResolutionDiscriminator, self).__init__()

        self.resolutions = resolutions
        self.discriminators = nn.ModuleList(
            [
                SpectralDiscriminator(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    norm=norm,
                )
                for n_fft, hop_length, win_length in self.resolutions
            ]
        )

    def forward(self, y_true, y_pred):
        y_real_scores = []
        y_fake_scores = []
        f_map_real = []
        f_map_fake = []

        for discriminator in self.discriminators:
            real_score, f_map_real_ = discriminator(y_true)
            fake_score, f_map_fake_ = discriminator(y_pred)

            y_real_scores.append(real_score)
            y_fake_scores.append(fake_score)
            f_map_real.append(f_map_real_)
            f_map_fake.append(f_map_fake_)

        return y_real_scores, y_fake_scores, f_map_real, f_map_fake


class Discriminator(nn.Module):
    def __init__(self, *discriminators: nn.Module):
        super(Discriminator, self).__init__()

        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y_true, y_pred):
        y_real_scores = []
        y_fake_scores = []
        f_map_real = []
        f_map_fake = []

        for discriminator in self.discriminators:
            real_scores, fake_scores, f_map_real_, f_map_fake_ = discriminator(y_true, y_pred)

            y_real_scores.extend(real_scores)
            y_fake_scores.extend(fake_scores)
            f_map_real.extend(f_map_real_)
            f_map_fake.extend(f_map_fake_)

        return y_real_scores, y_fake_scores, f_map_real, f_map_fake
