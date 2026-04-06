import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    train_audio: str = "/kaggle/input/competitions/birdclef-2026/train_audio"
    train_csv: str = "/kaggle/input/competitions/birdclef-2026/train.csv"
    sample_rate: int = 32000
    duration: int = 5
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 1024
    f_min: int = 40
    f_max: int = 15000
    top_db: int = 80
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-2
    model_name: str = "tf_efficientnet_b0_ns"
    valid_size: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioToSpec:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=cfg.top_db,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)
        spec = (spec + self.cfg.top_db) / self.cfg.top_db
        return torch.clamp(spec, 0.0, 1.0)


class SpecAugment:
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 32):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


class BirdClefDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        class_to_idx: dict[str, int],
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.class_to_idx = class_to_idx
        self.train = train
        self.num_samples = cfg.sample_rate * cfg.duration
        self.audio_to_spec = AudioToSpec(cfg)
        self.spec_augment = SpecAugment() if train else None

    def __len__(self) -> int:
        return len(self.df)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() < self.num_samples:
            pad_width = self.num_samples - waveform.numel()
            waveform = nn.functional.pad(waveform, (0, pad_width))
        elif waveform.numel() > self.num_samples:
            max_start = waveform.numel() - self.num_samples
            start = random.randint(0, max_start) if self.train else 0
            waveform = waveform[start : start + self.num_samples]
        return waveform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = row["path"]
        label = self.class_to_idx[row["primary_label"]]

        waveform, sample_rate = sf.read(path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

        if sample_rate != self.cfg.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.cfg.sample_rate,
            )

        waveform = self._fix_length(waveform)
        spec = self.audio_to_spec(waveform)

        if self.spec_augment is not None:
            spec = self.spec_augment(spec)

        spec = spec.unsqueeze(0)
        return spec, torch.tensor(label, dtype=torch.long)


class BirdClefEfficientNet(nn.Module):
    def __init__(self, num_classes: int, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def macro_auc_ovr(
    logits_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    num_classes: int,
) -> float:
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    one_hot = F.one_hot(targets, num_classes=num_classes).numpy()

    aucs = []
    for class_idx in range(num_classes):
        y_true = one_hot[:, class_idx]
        positive_count = int(y_true.sum())

        if positive_count == 0 or positive_count == len(y_true):
            continue

        y_score = probs[:, class_idx]
        aucs.append(roc_auc_score(y_true, y_score))

    return float(np.mean(aucs)) if aucs else float("nan")


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

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=amp_device, enabled=device.startswith("cuda")):
            logits = model(specs)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        logits_list.append(logits.cpu())
        targets_list.append(labels.cpu())

    val_auc = macro_auc_ovr(
        logits_list=logits_list,
        targets_list=targets_list,
        num_classes=len(loader.dataset.class_to_idx),
    )
    return running_loss / len(loader), val_auc


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    df = pd.read_csv(cfg.train_csv)
    df["path"] = df["filename"].apply(lambda x: os.path.join(cfg.train_audio, x))
    df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
    min_class_count = df["primary_label"].value_counts().min()
    stratify_labels = df["primary_label"] if min_class_count >= 2 else None

    train_df, valid_df = train_test_split(
        df,
        test_size=cfg.valid_size,
        stratify=stratify_labels,
        random_state=cfg.seed,
    )

    classes = sorted(df["primary_label"].unique())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    train_ds = BirdClefDataset(train_df, cfg, class_to_idx, train=True)
    valid_ds = BirdClefDataset(valid_df, cfg, class_to_idx, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, valid_loader, class_to_idx


def main() -> None:
    set_seed(CFG.seed)
    train_loader, valid_loader, class_to_idx = build_loaders(CFG)

    model = BirdClefEfficientNet(
        num_classes=len(class_to_idx),
        model_name=CFG.model_name,
    ).to(CFG.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = GradScaler(enabled=CFG.device.startswith("cuda"))

    best_val_auc = float("-inf")

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            CFG.device,
        )
        val_loss, val_auc = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            CFG.device,
        )
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{CFG.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "config": CFG.__dict__,
                },
                "/kaggle/working/efficientnet_b0_baseline.pth",
            )
            print("Saved new best checkpoint.")


if __name__ == "__main__":
    main()
