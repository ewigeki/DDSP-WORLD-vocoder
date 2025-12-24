import math

import torch
import torch.nn.functional as F
from torch import nn


class WORLD(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, window_size: int, hop_length: int, min_f0_hz: int = 71, audio_length_seconds: float = 4):
        super(WORLD, self).__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_length = hop_length

        self.min_f0_hz = min_f0_hz
        self.audio_length_seconds = audio_length_seconds

        self.K = int(math.ceil(self.sample_rate / (2 * self.min_f0)))


    def saw_impulse(self, f0: torch.Tensor, sample_rate: int, duration: float, K=155, f_min=71):
        """
        Generate harmonic pulse train excitation via summation of sinusoids.
        f0: Fundamental frequency contour (interpolated to audio rate). Shape: (B, 1, T)
        sample_rate: Sampling rate (e.g., 22050 Hz).
        duration: Duration of the signal in seconds.
        K: Number of harmonics.
        f_min: Minimum floor frequency to estimate harmonic range.
        """
        nyquist = sample_rate / 2
        k = torch.arange(1, K + 1, device=f0.device).view(1, K, 1)  # Shape: (1, K, 1)
        harmonic_freq = k * f0  # Shape: (B, K, T)
        harmonic_mask = (harmonic_freq <= nyquist) & (f0 >= f_min)  # Shape: (B, K, T)
        c_k = (1.0 / k).view(1, K, 1)
        phase = torch.cumsum(2 * torch.pi * f0 / float(sample_rate), dim=-1) * k  # (B, K, T)
        sinusoids = torch.sin(phase) * c_k  # (B, K, T)
        sinusoids = sinusoids * harmonic_mask.float()
        e_h = sinusoids.sum(dim=1)  # (B, T)

        min_f0_mask = (f0 >= f_min).squeeze(1)
        e_h = e_h * min_f0_mask

        return e_h


    def noise_filtering(self, e_n, sp, ap, n_fft=1024, hop_length=259, center=False, lenght=65536):
        """
        Filter harmonic component using spectral envelope and aperiodicity.
        e_h: Pulse train excitation signal.
        sp: Spectral envelope (magnitude).
        ap: Aperiodicity factor.
        n_fft: FFT size.
        hop_length: Hop size for STFT.
        """
        window = torch.hann_window(n_fft).to(e_n.device)
        stft_n = torch.stft(e_n, n_fft=n_fft, hop_length=hop_length, window=window, center=center, return_complex=True)

        filtered_n = ap * sp * stft_n
        n = torch.istft(filtered_n, n_fft=n_fft, hop_length=hop_length, window=window, center=center, length=lenght)

        return n


    def harmonic_filtering(self, e_h, sp, ap, n_fft=1024, hop_length=259, center=False, lenght=65536):
        """
        Filter harmonic component using spectral envelope and aperiodicity.
        e_h: Pulse train excitation signal.
        sp: Spectral envelope (magnitude).
        ap: Aperiodicity factor.
        n_fft: FFT size.
        hop_length: Hop size for STFT.
        """
        window = torch.hann_window(n_fft).to(e_h.device)
        stft_h = torch.stft(e_h, n_fft=n_fft, hop_length=hop_length, window=window, center=center, return_complex=True)

        filtered_h = (1 - ap) * sp * stft_h
        h = torch.istft(filtered_h, n_fft=n_fft, hop_length=hop_length, window=window, center=center, length=lenght)

        return h


    def forward(self, f0_hz, sp, ap, c_h = 1.0, c_n = 1.0) -> torch.Tensor:
        B = f0_hz.shape[0]

        interpolated_f0 = F.interpolate(f0_hz.unsqueeze(1), int(self.sample_rate * self.audio_length_seconds), mode="linear", align_corners=True)

        e_h = 0.48 * self.saw_impulse(interpolated_f0, self.sample_rate, self.audio_length_seconds, self.K, self.min_f0_hz)
        e_n = 2 * torch.rand((B, int(self.sample_rate * self.audio_length_seconds))).to(sp.device) - 1

        h = self.harmonic_filtering(e_h, sp, ap, n_fft=self.n_fft, hop_length=self.hop_length, lenght=int(self.sample_rate * self.audio_length_seconds), center=True)
        n = self.noise_filtering(e_n, sp, ap, n_fft=self.n_fft, hop_length=self.hop_length, lenght=int(self.sample_rate * self.audio_length_seconds), center=True)

        return c_h * h + c_n * n