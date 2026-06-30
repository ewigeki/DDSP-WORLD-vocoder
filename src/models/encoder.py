import torch
from torch import nn
from torchaudio import transforms

from src.models.resnet import ResBlock


class ZEncoder(nn.Module):
    """
    https://arxiv.org/pdf/2001.04643
    """

    def __init__(self, sample_rate: int, n_fft: int, win_length: int, hop_length: int, z_dim: int = 16, n_time_samples: int = 125):
        super(ZEncoder, self).__init__()

        self.mel_transform = transforms.MelSpectrogram(sample_rate,
                                                       n_fft,
                                                       win_length,
                                                       hop_length,
                                                       n_mels=81,
                                                       center=True,
                                                       normalized=True)

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 1), padding=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            ResBlock(64, 128, 21, n_time_samples, 1),
            ResBlock(128, 128, 21, n_time_samples, 1),
            ResBlock(128, 256, 21, n_time_samples, 2),
            ResBlock(256, 256, 11, n_time_samples, 1),
            ResBlock(256, 256, 11, n_time_samples, 1),
            ResBlock(256, 512, 11, n_time_samples, 2),
            ResBlock(512, 512, 6, n_time_samples, 1),
            ResBlock(512, 512, 6, n_time_samples, 1),
            ResBlock(512, 512, 6, n_time_samples, 1),
            ResBlock(512, 1024, 6, n_time_samples, 2),
            ResBlock(1024, 1024, 3, n_time_samples, 1),
            ResBlock(1024, 1024, 3, n_time_samples, 1),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.dense = nn.Linear(3072, z_dim)

        self.upsampler = nn.Sequential(
            nn.Upsample(size=1000, mode="linear", align_corners=True),
            nn.Softplus()
        )

    def forward(self, x):
        """

        :param x: Shape(B, T)
        :return: Shape(B, C, 1000)
        """

        output = self.mel_transform(x)
        output = output[:, None, :, :]
        output = self.model(output)
        output = output.permute(0, 2, 1)
        output = self.dense(output)
        output = output.permute(0, 2, 1)
        output = self.upsampler(output)

        return output


if __name__ == '__main__':
    x = torch.randn(1, 63999)
    model = ZEncoder(16000, 2048, 2048, 512, n_time_samples=125)
    y = model(x)
    print(y.shape)
