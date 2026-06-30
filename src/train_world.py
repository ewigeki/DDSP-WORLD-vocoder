# TODO: create separate interface file

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from src.data.source import WavDataset
from src.models.decoder import EmformerDecoder
from src.models.encoder import ZEncoder
from src.models.vocoder import WORLDVocoder
from src.pipelines.f0 import F0Pipeline
from src.pipelines.world import WORLD
from src.utils import find_wav_files

DATASET_ROOT_DIR = ""
F0_CHECKPOINT_PATH = ""
CHECKPOINT_PATH = ""
SAMPLE_AUDIO_FILE = ""


def train_pipeline():
    encoder = ZEncoder(16000, 1024, 1024, 256, 8)
    decoder = EmformerDecoder(z_dim=8)
    f0_predictor = F0Pipeline.load_from_checkpoint(F0_CHECKPOINT_PATH)
    vocoder = WORLDVocoder(16000, 1024, 1024, 256)

    model = WORLD(
        encoder,
        decoder,
        f0_predictor,
        vocoder,
        16000,
    )

    dataset = find_wav_files(DATASET_ROOT_DIR)
    train_dataset = WavDataset(dataset[6:])
    val_dataset = WavDataset(dataset[3:6])
    test_dataset = WavDataset(dataset[:3])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=12)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="{epoch:03d}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,
        min_delta=1e-4,
    )

    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)