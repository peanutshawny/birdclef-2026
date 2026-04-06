# Auto-generated from birdclef_2026_utilities (5).py for local Kaggle CLI workflow.

import numpy as np
import pandas as pd
import os
import random
import ast
from datetime import datetime

import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


class AudioToSpec:
    def __init__(
        self,
        mel_config,
    ):

        self.mel_config = mel_config

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.mel_config["sample_rate"],
            n_fft=self.mel_config["n_fft"],
            hop_length=self.mel_config["hop_length"],
            win_length=self.mel_config["win_length"],
            n_mels=self.mel_config["n_mels"],
            f_min=self.mel_config["f_min"],
            f_max=self.mel_config["f_max"],
            power=self.mel_config["power"],
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=self.mel_config["top_db"],
        )

    def __call__(self, wave):
        """
        wave: torch tensor of shape [num_samples]
        returns: torch tensor of shape [n_mels, time]
        """
        spec = self.mel_transform(wave)      # [n_mels, time]
        spec = self.db_transform(spec)       # roughly [-top_db, 0]
        spec = (spec + self.mel_config["top_db"]) / self.mel_config["top_db"]
        spec = torch.clamp(spec, 0.0, 1.0)
        return spec


class BirdClefDataset(Dataset):
    def __init__(
        self,
        df,
        mel_config,
        class_to_idx,
        sample_rate,
        duration,
        train=True,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.sample_rate = sample_rate
        self.duration = duration
        self.train = train
        self.num_samples = sample_rate * duration
        self.audio_to_spec = AudioToSpec(mel_config)

    def __len__(self):
        return len(self.df)

    def _fix_length(self, wave):
        if len(wave) < self.num_samples:
            wave = np.pad(wave, (0, self.num_samples - len(wave)))
        elif len(wave) > self.num_samples:
            wave = wave[:self.num_samples]
        return wave

    def _read_partial_audio(self, path, start_frame=None):
        with sf.SoundFile(path) as f:
            total_frames = len(f)
            sr = f.samplerate

            if start_frame is None:
                if total_frames <= self.num_samples:
                    wave = f.read(dtype="float32")
                else:
                    if self.train:
                        max_start = total_frames - self.num_samples
                        start_frame = random.randint(0, max_start)
                    else:
                        start_frame = 0

                    f.seek(start_frame)
                    wave = f.read(frames=self.num_samples, dtype="float32")
            else:
                f.seek(start_frame)
                wave = f.read(frames=self.num_samples, dtype="float32")

        return wave, sr

    def _load_audio_window(self, row):
        path = row["path"]
        start_sec = row["start_sec"]

        if pd.notna(start_sec):
            start_frame = int(float(start_sec) * self.sample_rate)
            wave, sr = self._read_partial_audio(path, start_frame=start_frame)
        else:
            wave, sr = self._read_partial_audio(path, start_frame=None)

        if wave.ndim > 1:
            wave = wave.mean(axis=1)

        wave = wave.astype(np.float32)

        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate}, got {sr}")

        wave = self._fix_length(wave)
        return wave

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wave = self._load_audio_window(row)
        spec = self.audio_to_spec(torch.tensor(wave, dtype=torch.float32)).unsqueeze(0)

        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in row["labels"]:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0

        return spec, target


class BirdClefEfficientNet(nn.Module):
    def __init__(self, num_classes: int, model_name: str, is_pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=is_pretrained,
            num_classes=num_classes,
            in_chans=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def macro_auc_multilabel(
    logits_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
) -> float:
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    probs = torch.sigmoid(logits).numpy()
    targets_np = targets.numpy()

    scored_columns = targets_np.sum(axis=0) > 0
    if not scored_columns.any():
        return float("nan")

    return float(
        roc_auc_score(
            targets_np[:, scored_columns],
            probs[:, scored_columns],
            average="macro",
        )
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0
    amp_device = "cuda" if device.startswith("cuda") else "cpu"

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device, enabled=device.startswith("cuda")):
            logits = model(specs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    logits_list = []
    targets_list = []

    for specs, targets in loader:
        specs = specs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=amp_device, enabled=device.startswith("cuda")):
            logits = model(specs)
            loss = criterion(logits, targets)

        running_loss += loss.item()
        logits_list.append(logits.cpu())
        targets_list.append(targets.cpu())

    val_auc = macro_auc_multilabel(
        logits_list=logits_list,
        targets_list=targets_list,
    )
    val_loss = running_loss / len(loader)
    return val_loss, val_auc


def _dedupe_labels(labels) -> list[str]:
    seen = set()
    deduped = []
    for label in labels:
        label = str(label).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped


def _parse_secondary_labels(value) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return _dedupe_labels(value)

    text = str(value).strip()
    if text == "" or text == "[]" or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return _dedupe_labels(parsed)
    except (ValueError, SyntaxError):
        pass

    if ";" in text:
        return _dedupe_labels(text.split(";"))

    return _dedupe_labels([text])


def _parse_soundscape_labels(value) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return _dedupe_labels(value)

    text = str(value).strip()
    if text == "" or text == "[]" or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return _dedupe_labels(parsed)
    except (ValueError, SyntaxError):
        pass

    if ";" in text:
        return _dedupe_labels(text.split(";"))

    return _dedupe_labels([text])


def _build_class_to_idx(audio_df: pd.DataFrame, soundscape_df: pd.DataFrame) -> dict[str, int]:
    class_names = set()

    for labels in audio_df["labels"]:
        class_names.update(labels)

    for labels in soundscape_df["labels"]:
        class_names.update(labels)

    classes = sorted(class_names)
    return {label: idx for idx, label in enumerate(classes)}


def _load_train_audio_df(config) -> pd.DataFrame:
    df = pd.read_csv(config["paths"]["train_csv"])

    unique_filenames = sorted(df["filename"].unique())
    filename_to_shard = {
        filename: idx % 4
        for idx, filename in enumerate(unique_filenames)
    }

    df["shard_id"] = df["filename"].map(filename_to_shard)

    df["path"] = df.apply(
        lambda row: os.path.join(
            config["paths"][f"train_audio_wav_root_{row['shard_id']}"],
            str(Path(row["filename"]).with_suffix(".wav")),
        ),
        axis=1,
    )

    print("Mapped wav paths:", df["path"].notna().sum())
    print(
        "Existing wav paths:",
        df["path"].map(lambda p: os.path.exists(p) if pd.notna(p) else False).sum(),
    )

    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)

    primary_col = config["train_columns"]["primary_label_col"]
    secondary_col = config["train_columns"]["secondary_label_col"]

    df["labels"] = df.apply(
        lambda row: _dedupe_labels(
            [row[primary_col]] + _parse_secondary_labels(row[secondary_col])
        ),
        axis=1,
    )
    df["audio_id"] = df["filename"].str.rsplit(".", n=1).str[0]
    df["start_sec"] = np.nan
    df["source"] = "train_audio"

    return df[["path", "labels", "audio_id", "start_sec", "source"]].copy()

    


def _load_soundscape_df(config) -> pd.DataFrame:
    df = pd.read_csv(config["paths"]["train_soundscapes_labels_csv"])
    df["path"] = df["filename"].apply(
        lambda x: os.path.join(config["paths"]["train_soundscapes"], x)
    )
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)

    df["labels"] = df["primary_label"].apply(_parse_soundscape_labels)
    df["audio_id"] = df["filename"].str.rsplit(".", n=1).str[0]
    df["start_sec"] = pd.to_timedelta(df["start"]).dt.total_seconds().astype(float)
    df["source"] = "train_soundscape"

    return df[["path", "labels", "audio_id", "start_sec", "source"]].copy()


def _build_combined_df(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audio_df = _load_train_audio_df(config)
    soundscape_df = _load_soundscape_df(config)
    combined_df = pd.concat([audio_df, soundscape_df], ignore_index=True)
    combined_df["group_id"] = combined_df["source"] + "::" + combined_df["audio_id"].astype(str)
    return audio_df, soundscape_df, combined_df


def _split_combined_df(config, combined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["train"]["val_size"],
        random_state=config["train"]["seed"],
    )

    train_idx, val_idx = next(
        splitter.split(combined_df, groups=combined_df["group_id"])
    )

    train_df = combined_df.iloc[train_idx].reset_index(drop=True)
    val_df = combined_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def _make_loader(dataset, config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["dataloader"]["num_workers"] > 0,
    )


def build_loaders(config) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    audio_df, soundscape_df, combined_df = _build_combined_df(config)
    class_to_idx = _build_class_to_idx(audio_df, soundscape_df)
    train_df, val_df = _split_combined_df(config, combined_df)

    train_ds = BirdClefDataset(
        df=train_df,
        mel_config=config["mel_spectrogram"],
        class_to_idx=class_to_idx,
        sample_rate=config["audio"]["sample_rate"],
        duration=config["audio"]["duration"],
        train=True,
    )

    val_ds = BirdClefDataset(
        df=val_df,
        mel_config=config["mel_spectrogram"],
        class_to_idx=class_to_idx,
        sample_rate=config["audio"]["sample_rate"],
        duration=config["audio"]["duration"],
        train=False,
    )

    train_loader = _make_loader(train_ds, config, shuffle=True)
    val_loader = _make_loader(val_ds, config, shuffle=False)

    return train_loader, val_loader, class_to_idx


### INFERENCE
#############

def build_segment_tensor(audio_path: Path, end_seconds: list[int], config) -> torch.Tensor:
    num_samples = config["sample_rate"] * config["duration"]
    audio_to_spec = AudioToSpec(config["mel_spectrogram"])
    wave, sr = sf.read(audio_path)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    wave = wave.astype(np.float32)

    if sr != config["sample_rate"]:
        raise ValueError(f"Expected {config['sample_rate']}, got {sr}")

    specs = []
    for end_sec in end_seconds:
        start_sample = max(0, (int(end_sec) - config["duration"]) * config["sample_rate"])
        chunk = wave[start_sample:start_sample + num_samples]

        if len(chunk) < num_samples:
            chunk = np.pad(chunk, (0, num_samples - len(chunk)))
        elif len(chunk) > num_samples:
            chunk = chunk[:num_samples]

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
        specs.append(audio_to_spec(chunk_tensor).unsqueeze(0))

    return torch.stack(specs, dim=0)


@torch.no_grad()
def predict_file(audio_path: Path, end_seconds: list[int], model, config) -> np.ndarray:
    specs = build_segment_tensor(audio_path, end_seconds, config)
    preds = []

    for start in range(0, len(specs), config["infer_batch_size"]):
        batch = specs[start:start + config["infer_batch_size"]]
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    return np.concatenate(preds, axis=0)