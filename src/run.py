from collections.abc import Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from src.stages.preprocess_wav import preprocess_wav
from src.stages.train_f0 import train_f0
from src.stages.train_world import train_world


StageFn = Callable[[DictConfig], str | None]


def _stage_cfg(cfg: DictConfig, name: str) -> DictConfig:
    common = {
        "seed": cfg.get("seed"),
        "log_dir": cfg.get("log_dir"),
        "experiment_tag": cfg.get("experiment_tag"),
        "data": cfg.data,
        "logger": cfg.get("logger"),
        "trainer": cfg.get("trainer"),
    }
    return OmegaConf.merge(common, cfg.get(name, {}))


def _selected_stage_range(cfg: DictConfig) -> tuple[int, int]:
    stage_from = int(cfg.stages["from"])
    stage_to = int(cfg.stages["to"])
    if stage_from < 1 or stage_to > 3 or stage_from > stage_to:
        raise ValueError(
            f"Invalid stage range: stages.from={stage_from}, stages.to={stage_to}. "
            "Expected 1 <= from <= to <= 3."
        )
    return stage_from, stage_to


@hydra.main(version_base=None, config_path="../conf", config_name="run")
def main(cfg: DictConfig) -> None:
    stage_from, stage_to = _selected_stage_range(cfg)
    f0_checkpoint_path = None

    stages: list[tuple[int, str, StageFn]] = [
        (1, "preprocess_wav", preprocess_wav),
        (2, "train_f0", train_f0),
        (3, "train_world", train_world),
    ]

    for stage_number, stage_name, stage_fn in stages:
        if not stage_from <= stage_number <= stage_to:
            continue

        print(f"Running stage {stage_number}: {stage_name}")
        stage_cfg = _stage_cfg(cfg, stage_name)
        if stage_name == "train_world":
            f0_checkpoint_path = train_world(stage_cfg, f0_checkpoint_path)
        else:
            checkpoint_path = stage_fn(stage_cfg)
            if stage_name == "train_f0":
                f0_checkpoint_path = checkpoint_path


if __name__ == "__main__":
    main()
