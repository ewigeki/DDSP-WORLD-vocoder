import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.training import build_wav_dataloaders

import matplotlib
matplotlib.use("Agg", force=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="train_f0")
def main(cfg: DictConfig) -> None:
    train_f0(cfg)


def _checkpoint_path(trainer) -> str | None:
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    if checkpoint_callback is None:
        return None

    return (
        getattr(checkpoint_callback, "last_model_path", None)
        or getattr(checkpoint_callback, "best_model_path", None)
        or None
    )


def train_f0(cfg: DictConfig) -> str | None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    train_dataloader, val_dataloader, _ = build_wav_dataloaders(cfg.data)
    cent_predictor = instantiate(cfg.model.cent_predictor)
    model = instantiate(cfg.pipeline, cent_predictor=cent_predictor)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader)
    return _checkpoint_path(trainer)


if __name__ == "__main__":
    main()
