import torch
import torch.nn.functional as F
from torch import nn


def get_cents_grid(c_min=2051.148762868, c_max=9151.288843308, n=360):
    return torch.linspace(c_min, c_max, n)


def get_cent(self, predicted_cent_logit):
    "predicted_cent_prob - (B, C, T)"
    predicted_cent_prob = F.sigmoid(predicted_cent_logit)
    cents = get_cents_grid()[None, :, None].to(predicted_cent_prob)
    pred_cent = torch.sum(predicted_cent_prob * cents, dim=1) / torch.sum(predicted_cent_prob, dim=1)
    return pred_cent


class Crepe(nn.Module):
    """
    https://arxiv.org/pdf/1802.06182
    """

    def __init__(self, n_cents=360):
        super(Crepe, self).__init__()

        c_outs = [1024, 128, 128, 128, 256, 512]
        c_ins = [1] + c_outs[:-1]
        kernels = [512, 65, 65, 65, 65, 65, 65]
        strides = [4, 1, 1, 1, 1, 1]
        paddings = [254, 32, 32, 32, 32, 32]

        layers = []
        for c_in, c_out, w, s, p in zip(c_ins, c_outs, kernels, strides, paddings):
            layers += [
                nn.Conv1d(c_in, c_out, w, stride=s, padding=p),
                nn.ReLU(),
                nn.BatchNorm1d(c_out),
                nn.MaxPool1d(2),
            ]

        self.model = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(2048, n_cents),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Shape (B, T)
        :return: Shape (B, n_cents)
        """

        output = x[:, None, :]

        output = self.model(output)

        return output


if __name__ == '__main__':
    x = torch.randn(1, 1024)
    model = Crepe()
    x = model(x)
    print(x.shape)