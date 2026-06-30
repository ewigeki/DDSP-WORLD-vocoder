import math

from torch import nn


class ResBlock(nn.Module):
  def __init__(self, c_in: int, c_out: int, h: int, w: int, freq_stride: int):
    super(ResBlock, self).__init__()

    self.model = nn.Sequential( # C, H, W
      nn.LayerNorm([c_in, h, w]), # C, H, W
      nn.ReLU(),
      nn.Conv2d(c_in, c_out // 4, kernel_size=1), # C / 4, H, W
      nn.LayerNorm([c_out // 4, h, w]), # C / 4, H, W
      nn.ReLU(),
      nn.Conv2d(c_out // 4, c_out // 4, kernel_size=3, padding=1, stride=(freq_stride, 1)), # C / 4, (H - 1) // 2 + 1, W
      nn.LayerNorm([c_out // 4, int(math.floor((h - 1) / freq_stride) + 1), w]),
      nn.ReLU(),
      nn.Conv2d(c_out // 4, c_out, kernel_size=1) # C, (H - 1) // 2 + 1, W
    )

    self.shortcut = nn.Sequential(
      nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=(freq_stride, 1)) # C, (H - 1) // 2 + 1, W
    )

  def forward(self, x):
    """

    :param x: Shape (B, C, H, W)
    :return:
    """
    return self.model(x) + self.shortcut(x)