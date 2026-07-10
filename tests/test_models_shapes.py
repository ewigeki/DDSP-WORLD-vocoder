import pytest
import torch

from src.models.crepe import Crepe
from src.models.decoder import EmformerDecoder
from src.models.discriminators import (
    PeriodDiscriminator,
    ScaleDiscriminator,
    SpectralDiscriminator,
)
from src.models.encoder import ZEncoder
from src.models.vocoder import WORLDVocoder


SAMPLE_RATE_CASES = (22500, 44100, 48000)


def _scaled_fft_config(sample_rate: int) -> tuple[int, int]:
    n_fft = int(round(sample_rate * 0.016))
    if n_fft % 2:
        n_fft += 1
    return n_fft, n_fft // 4


def test_crepe_outputs_time_major_cent_logits():
    model = Crepe(n_cents=360).eval()
    audio = torch.randn(2, 4096)

    with torch.no_grad():
        logits = model(audio)

    assert logits.shape == (2, 16, 360)
    assert torch.isfinite(logits).all()


def test_z_encoder_outputs_fixed_length_latent_sequence():
    model = ZEncoder(
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=128,
        z_dim=8,
        n_time_samples=125,
    ).eval()
    audio = torch.randn(2, 16000)

    with torch.no_grad():
        z = model(audio)

    assert z.shape == (2, 8, 1000)
    assert torch.isfinite(z).all()


@pytest.mark.parametrize("sample_rate", SAMPLE_RATE_CASES)
def test_z_encoder_supports_common_high_sample_rates(sample_rate):
    hop_length = sample_rate // 125
    n_fft = 4 * hop_length
    model = ZEncoder(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        z_dim=4,
        n_time_samples=125,
    ).eval()
    audio = torch.randn(1, sample_rate)

    with torch.no_grad():
        z = model(audio)

    assert z.shape == (1, 4, 1000)
    assert torch.isfinite(z).all()


def test_z_encoder_rejects_non_batched_waveform():
    model = ZEncoder(
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=128,
        z_dim=8,
        n_time_samples=125,
    )

    with pytest.raises(ValueError, match="expected 2D input"):
        model(torch.randn(1, 1, 16000))


def test_emformer_decoder_outputs_world_feature_shapes_and_ranges():
    model = EmformerDecoder(
        z_dim=4,
        sample_rate=16000,
        n_fft=64,
        n_mels=8,
        ap_bins=4,
        hidden_dim=16,
        num_heads=2,
        ffn_dim=32,
        num_emformer_layers=1,
        emformer_segment_lenght=4,
        emformer_left_context_length=2,
        emformer_right_context_length=2,
        emformer_max_memory_size=2,
    ).eval()
    f0_hz = torch.full((2, 10, 1), 220.0)
    z = torch.randn(2, 10, 4)

    with torch.no_grad():
        sp, ap = model(f0_hz, z)

    assert sp.shape == (2, 33, 10)
    assert ap.shape == (2, 33, 10)
    assert torch.isfinite(sp).all()
    assert torch.isfinite(ap).all()
    assert torch.all(sp >= 0.0)
    assert torch.all((ap >= 0.0) & (ap <= 1.0))


@pytest.mark.parametrize("sample_rate", SAMPLE_RATE_CASES)
def test_emformer_decoder_supports_common_high_sample_rates(sample_rate):
    n_fft, _ = _scaled_fft_config(sample_rate)
    model = EmformerDecoder(
        z_dim=4,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=16,
        ap_bins=4,
        hidden_dim=16,
        num_heads=2,
        ffn_dim=32,
        num_emformer_layers=1,
        emformer_segment_lenght=4,
        emformer_left_context_length=2,
        emformer_right_context_length=2,
        emformer_max_memory_size=2,
    ).eval()
    f0_hz = torch.full((1, 10, 1), 220.0)
    z = torch.randn(1, 10, 4)

    with torch.no_grad():
        sp, ap = model(f0_hz, z)

    assert sp.shape == (1, n_fft // 2 + 1, 10)
    assert ap.shape == (1, n_fft // 2 + 1, 10)
    assert torch.isfinite(sp).all()
    assert torch.isfinite(ap).all()
    assert torch.all(sp >= 0.0)
    assert torch.all((ap >= 0.0) & (ap <= 1.0))


def test_world_vocoder_returns_audio_length():
    model = WORLDVocoder(
        sample_rate=8000,
        n_fft=64,
        window_size=64,
        hop_length=16,
        min_f0_hz=71,
        audio_length_seconds=0.032,
    )
    batch_size = 2
    audio_length = int(model.sample_rate * model.audio_length_seconds)
    frames = torch.stft(
        torch.zeros(batch_size, audio_length),
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        window=torch.hann_window(model.n_fft),
        center=True,
        return_complex=True,
    ).shape[-1]
    freq_bins = model.n_fft // 2 + 1
    f0_hz = torch.full((batch_size, frames), 120.0)
    sp = torch.ones(batch_size, freq_bins, frames)
    ap = torch.full((batch_size, freq_bins, frames), 0.5)

    with torch.no_grad():
        audio = model(f0_hz, sp, ap)

    assert audio.shape == (batch_size, audio_length)
    assert torch.isfinite(audio).all()


@pytest.mark.parametrize("sample_rate", SAMPLE_RATE_CASES)
def test_world_vocoder_supports_common_high_sample_rates(sample_rate):
    n_fft, hop_length = _scaled_fft_config(sample_rate)
    model = WORLDVocoder(
        sample_rate=sample_rate,
        n_fft=n_fft,
        window_size=n_fft,
        hop_length=hop_length,
        min_f0_hz=71,
        audio_length_seconds=0.032,
    )
    batch_size = 1
    audio_length = int(model.sample_rate * model.audio_length_seconds)
    frames = torch.stft(
        torch.zeros(batch_size, audio_length),
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        window=torch.hann_window(model.n_fft),
        center=True,
        return_complex=True,
    ).shape[-1]
    freq_bins = model.n_fft // 2 + 1
    f0_hz = torch.full((batch_size, frames), 120.0)
    sp = torch.ones(batch_size, freq_bins, frames)
    ap = torch.full((batch_size, freq_bins, frames), 0.5)

    with torch.no_grad():
        audio = model(f0_hz, sp, ap)

    assert audio.shape == (batch_size, audio_length)
    assert torch.isfinite(audio).all()


@pytest.mark.parametrize(
    ("model", "expected_feature_maps"),
    [
        (PeriodDiscriminator(period=3, norm="none"), 6),
        (ScaleDiscriminator(norm="none"), 7),
        (
            SpectralDiscriminator(
                n_fft=64,
                hop_length=16,
                win_length=64,
                norm="none",
            ),
            6,
        ),
    ],
)
def test_discriminators_return_batch_scores_and_feature_maps(
    model,
    expected_feature_maps,
):
    audio = torch.randn(2, 512)

    with torch.no_grad():
        score, feature_maps = model(audio)

    assert score.shape[0] == audio.shape[0]
    assert score.ndim == 2
    assert len(feature_maps) == expected_feature_maps
    assert all(feature_map.shape[0] == audio.shape[0] for feature_map in feature_maps)
