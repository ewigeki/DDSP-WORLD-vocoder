import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.modules.f0 import F0Pipeline
from src.training import build_wav_dataloaders

import matplotlib
matplotlib.use("Agg", force=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="train_world")
def main(cfg: DictConfig) -> None:
    train_world(cfg)


def _checkpoint_path(trainer) -> str | None:
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is None:
        return None

    return (
        getattr(checkpoint_callback, "last_model_path", None)
        or getattr(checkpoint_callback, "best_model_path", None)
        or None
    )


def _load_world_weights(model, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])


def train_world(cfg: DictConfig, f0_checkpoint_path: str | None = None) -> str | None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision('medium')

    train_dataloader, val_dataloader, _ = build_wav_dataloaders(cfg.data)

    checkpoint_path = f0_checkpoint_path or cfg.model.f0_checkpoint_path
    if not checkpoint_path:
        raise ValueError("Set model.f0_checkpoint_path to a trained F0 checkpoint before WORLD training")

    f0_cent_predictor = instantiate(cfg.model.f0_predictor.cent_predictor)
    f0_predictor = F0Pipeline.load_from_checkpoint(
        checkpoint_path,
        cent_predictor=f0_cent_predictor,
    )

    model = instantiate(
        cfg.pipeline,
        encoder=instantiate(cfg.model.encoder),
        decoder=instantiate(cfg.model.decoder),
        f0_predictor=f0_predictor,
        vocoder=instantiate(cfg.model.vocoder),
    )
    trainer = instantiate(cfg.trainer)
    world_checkpoint_path = cfg.get("world_checkpoint_path")
    if world_checkpoint_path:
        _load_world_weights(model, world_checkpoint_path)

    trainer.fit(model, train_dataloader, val_dataloader)
    return _checkpoint_path(trainer)


if __name__ == "__main__":
    main()
