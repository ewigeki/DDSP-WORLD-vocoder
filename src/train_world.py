import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.pipelines.f0 import F0Pipeline
from src.training import build_wav_dataloaders


@hydra.main(version_base=None, config_path="../conf", config_name="train_world")
def train_pipeline(cfg: DictConfig):
    train_dataloader, val_dataloader, _ = build_wav_dataloaders(cfg.data)

    if not cfg.model.f0_checkpoint_path:
        raise ValueError("Set model.f0_checkpoint_path to a trained F0 checkpoint before WORLD training")

    f0_cent_predictor = instantiate(cfg.model.f0_predictor.cent_predictor)
    f0_predictor = F0Pipeline.load_from_checkpoint(
        cfg.model.f0_checkpoint_path,
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
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train_pipeline()
