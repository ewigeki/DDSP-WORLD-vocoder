import librosa
import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.models import Emformer


class EmformerDecoder(nn.Module):
    def __init__(self,
                 z_dim: int = 16,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 ffn_dim: int = 512,
                 num_emformer_layers: int = 2,
                 emformer_segment_lenght: int = 32,
                 emformer_left_context_length: int = 12,
                 emformer_right_context_length: int = 12,
                 emformer_max_memory_size: int = 4
                 ):
        super(EmformerDecoder, self).__init__()

        self.prenet = nn.Sequential(
            nn.Linear(1 + z_dim, hidden_dim), # log f0 + z
            nn.Tanh(),
            nn.Dropout(0.25),
        )

        self.emformer = Emformer(
            hidden_dim,
            num_heads,
            ffn_dim,
            num_emformer_layers,
            emformer_segment_lenght,
            left_context_length=emformer_left_context_length,
            right_context_length=emformer_right_context_length,
            max_memory_size=emformer_max_memory_size
        )

        self.postnet = nn.Sequential(
            nn.Linear(hidden_dim, 200),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(200, 80 + 16),
        )

        self.ap_upsampler = nn.Upsample(size=129, mode='linear', align_corners=False)
        self.M = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=129 * 2 - 1, n_mels=80))  # (n_mels, n_fft/2+1)
        self.M_r = nn.Parameter(
            torch.clamp(
                torch.linalg.pinv(self.M),
                min=0
            ),
            requires_grad=False
        )  # (n_fft/2+1, n_mels)

    def forward(self, f0_hz, z):
        output = torch.concat([
            torch.log1p(f0_hz),  # 1
            z,  # z_dim
        ], dim=-1)

        output = self.prenet(output)

        B, T, _ = f0_hz.shape
        right_context_length = self.emformer.right_context_length

        lengths = torch.full((B,), T, dtype=torch.long)
        lengths = lengths.to(output.device)
        output = F.pad(output, (0, 0, 0, right_context_length))  # (B, T, F)
        output, lengths = self.emformer(output, lengths)

        output = self.postnet(output)

        sp = output[:, :, :80]  # (B, T, F)
        ap = output[:, :, 80:]  # (B, T, F)

        sp = torch.pow(10, sp)
        sp = sp @ self.M_r.T  # (B, T, F)
        ap = F.sigmoid(ap)
        ap = self.ap_upsampler(ap)  # (B, T, F)

        sp = sp.permute(0, 2, 1)
        ap = ap.permute(0, 2, 1)

        return sp, ap