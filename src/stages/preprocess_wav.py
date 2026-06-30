import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.source import WavDataset
from src.utils import find_wav_files


@hydra.main(version_base=None, config_path="../../conf", config_name="preprocess_wav")
def main(cfg: DictConfig) -> None:
    preprocess_wav(cfg)


def preprocess_wav(cfg: DictConfig) -> None:
    if cfg.data.get("source", "wav") == "precomputed":
        print(f"Using precomputed data from {cfg.data.processed_dir}; skipping WAV preprocessing")
        return

    wav_paths = find_wav_files(cfg.data.root_dir)
    if not wav_paths:
        raise ValueError(f"No WAV files found under data.root_dir={cfg.data.root_dir!r}")

    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset = WavDataset(wav_paths, **dataset_kwargs)
    print(f"Cached {len(dataset)} samples from {len(wav_paths)} WAV files in {cfg.data.processed_dir}")


if __name__ == "__main__":
    main()
