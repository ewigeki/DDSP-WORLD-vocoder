import torch

from src.models.crepe import cent_to_hz, get_target, hz_to_cent, logits_to_cent


def test_hz_cent_conversion_round_trip():
    hz = torch.tensor([[80.0, 220.0, 440.0, 880.0]])

    reconstructed_hz = cent_to_hz(hz_to_cent(hz))

    assert torch.allclose(reconstructed_hz, hz, rtol=1e-5, atol=1e-5)


def test_get_target_matches_crepe_cent_bins():
    f0 = torch.tensor([[100.0, 220.0]])

    target = get_target(f0)

    assert target.shape == (1, 2, 360)
    assert torch.all((target >= 0.0) & (target <= 1.0))
    assert torch.argmax(target, dim=-1).shape == f0.shape


def test_logits_to_cent_preserves_batch_and_time_dimensions():
    logits = torch.zeros(2, 5, 360)

    cents = logits_to_cent(logits)

    assert cents.shape == (2, 5)
    assert torch.isfinite(cents).all()
