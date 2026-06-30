from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super(PeriodDiscriminator, self).__init__()

        self.period: int = period

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=2 ** 64, kernel_size=(kernel_size, 1), stride=(stride, 1)),
                *[nn.Conv2d(in_channels=2 ** (4 + l), out_channels=2 ** (5 + l), kernel_size=(kernel_size, 1), stride=(stride, 1)) for l in
                  range(2, 5)],
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(kernel_size, 1), stride=(3, 1), padding=((kernel_size - 1) // 2, 1)),
            ]
        )

        self.post_conv = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 1), stride=(1, 1))

    def forward(self, x):
        f_map = []

        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, channel, time = x.shape

        if time % self.period != 0:
            pad_len = self.period - (time % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            time = time + pad_len

        x = x.view(-1, time // self.period, self.period)

        for conv in self.conv:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            f_map.append(x)

        x = self.post_conv(x)
        f_map.append(x)

        score = torch.flatten(x, start_dim=1)

        return score, f_map


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods

        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(period) for period in self.periods
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


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super(MultiResolutionDiscriminator, self).__init__()