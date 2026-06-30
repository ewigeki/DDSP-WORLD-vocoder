import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.training import build_wav_dataloaders


@hydra.main(version_base=None, config_path="../conf", config_name="train_f0")
def train_pipeline(cfg: DictConfig):
    train_dataloader, val_dataloader, _ = build_wav_dataloaders(cfg.data)
    cent_predictor = instantiate(cfg.model.cent_predictor)
    model = instantiate(cfg.pipeline, cent_predictor=cent_predictor)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    train_pipeline()
