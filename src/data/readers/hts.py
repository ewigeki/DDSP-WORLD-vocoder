from dataclasses import dataclass

from pathlib import Path
from typing import List


@dataclass
class HTSPhoneme:
    start: int
    end: int
    phoneme: str


@dataclass
class HTSMono:
    hts_phonemes: List[HTSPhoneme]


def hts_time_to_seconds(value: int):
    return value / 10_000_000


def read_mono(path: Path) -> HTSMono:
    text = path.read_text(encoding="utf-8")

    hts_phonemes: List[HTSPhoneme] = []
    for line in text.split("\n"):
        start, end, phoneme = line.split("\t", 3)
        hts_phonemes.append(
            HTSPhoneme(
                start=int(start),
                end=int(end),
                phoneme=phoneme)
        )

    return HTSMono(hts_phonemes = hts_phonemes)