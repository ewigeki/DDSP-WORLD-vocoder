from pathlib import Path


def find_wav_files(folder: str | Path):
    return sorted(Path(folder).rglob("*.wav"))
