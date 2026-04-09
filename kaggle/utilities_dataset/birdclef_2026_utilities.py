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
from sklearn.model_selection import GroupKFold
from pathlib import Path

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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

        if self.mel_config["resize_shape"] is not None:
            spec = F.interpolate(
                spec.unsqueeze(0).unsqueeze(0),
                size=self.mel_config["resize_shape"],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        if self.mel_config["normalization"] == "top_db":
            spec = (spec + self.mel_config["top_db"]) / self.mel_config["top_db"]
        elif self.mel_config["normalization"] == "sample_minmax":
            spec_min = spec.min()
            spec_max = spec.max()
            spec = (spec - spec_min) / (spec_max - spec_min + 1e-7)
        else:
            raise ValueError(
                f"Unsupported normalization mode: {self.mel_config['normalization']}"
            )

        spec = torch.clamp(spec, 0.0, 1.0)
        return spec

class SpecAugment:
    def __init__(self, config: dict | None = None):
        self.config = config

        self.enabled = self.config["enabled"]
        self.probability = self.config["probability"]
        self.num_time_masks = self.config["num_time_masks"]
        self.time_mask_param = self.config["time_mask_param"]
        self.num_freq_masks = self.config["num_freq_masks"]
        self.freq_mask_param = self.config["freq_mask_param"]

        self.time_mask = None
        self.freq_mask = None
        if self.time_mask_param > 0:
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=self.time_mask_param
            )
        if self.freq_mask_param > 0:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=self.freq_mask_param
            )

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.enabled or random.random() >= self.probability:
            return spec

        augmented = spec.unsqueeze(0)

        if self.freq_mask is not None:
            for _ in range(self.num_freq_masks):
                augmented = self.freq_mask(augmented)

        if self.time_mask is not None:
            for _ in range(self.num_time_masks):
                augmented = self.time_mask(augmented)

        return augmented.squeeze(0)


class BirdClefDataset(Dataset):
    def __init__(
        self,
        df,
        mel_config,
        mixup_config,
        specaugment_config,
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
        self.wave_mixup_config = mixup_config
        self.spec_augment = SpecAugment(specaugment_config)

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

    def _build_target(self, row) -> torch.Tensor:
        target = torch.zeros(len(self.class_to_idx), dtype=torch.float32)
        for label in row["labels"]:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        return target

    def _sample_mixup_lambda(self) -> float:
        alpha = self.wave_mixup_config["alpha"]
        if alpha <= 0.0:
            return 1.0
        return float(np.random.beta(alpha, alpha))

    def _apply_wave_mixup(
        self,
        wave: np.ndarray,
        target: torch.Tensor,
        idx: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        if not self.train or not self.wave_mixup_config["enabled"]:
            return wave, target

        probability = self.wave_mixup_config["probability"]
        if probability <= 0.0 or random.random() >= probability or len(self.df) < 2:
            return wave, target

        mix_idx = random.randrange(len(self.df) - 1)
        if mix_idx >= idx:
            mix_idx += 1

        mix_row = self.df.iloc[mix_idx]
        mix_wave = self._load_audio_window(mix_row)
        mix_target = self._build_target(mix_row)
        lam = self._sample_mixup_lambda()

        mixed_wave = lam * wave + (1.0 - lam) * mix_wave
        mixed_target = lam * target + (1.0 - lam) * mix_target
        return mixed_wave.astype(np.float32), mixed_target

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wave = self._load_audio_window(row)
        target = self._build_target(row)

        if self.train and self.wave_mixup_config["enabled"]:
            wave, target = self._apply_wave_mixup(wave, target, idx)
        
        spec = self.audio_to_spec(torch.tensor(wave, dtype=torch.float32))

        if self.train and self.spec_augment.enabled:
            spec = self.spec_augment(spec)

        return spec.unsqueeze(0), target


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


def _split_combined_df(
    config,
    combined_df: pd.DataFrame,
    fold_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_splits = int(config["train"]["n_splits"])
    num_groups = combined_df["group_id"].nunique()

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    if num_groups < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds number of groups={num_groups}"
        )
    if not 0 <= fold_idx < n_splits:
        raise ValueError(
            f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}"
        )

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(combined_df, groups=combined_df["group_id"]))
    train_idx, val_idx = splits[fold_idx]

    train_df = combined_df.iloc[train_idx].reset_index(drop=True)
    val_df = combined_df.iloc[val_idx].reset_index(drop=True)

    print(f"Using GroupKFold fold {fold_idx + 1}/{n_splits}")
    print("train groups:", train_df["group_id"].nunique())
    print("validation groups:", val_df["group_id"].nunique())
    return train_df, val_df


def _make_loader(dataset, config, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    loader_seed = config["train"]["seed"]
    if not shuffle:
        loader_seed += 1
    generator.manual_seed(loader_seed)

    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["dataloader"]["num_workers"] > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def build_loaders(config, fold_idx: int) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    audio_df, soundscape_df, combined_df = _build_combined_df(config)
    class_to_idx = _build_class_to_idx(audio_df, soundscape_df)
    train_df, val_df = _split_combined_df(config, combined_df, fold_idx)

    train_ds = BirdClefDataset(
        df=train_df,
        mel_config=config["mel_spectrogram"],
        mixup_config=config["augmentations"]["wave_mixup"],
        specaugment_config=config["augmentations"]["specaugment"],
        class_to_idx=class_to_idx,
        sample_rate=config["audio"]["sample_rate"],
        duration=config["audio"]["duration"],
        train=True,
    )

    val_ds = BirdClefDataset(
        df=val_df,
        mel_config=config["mel_spectrogram"],
        mixup_config=config["augmentations"]["wave_mixup"],
        specaugment_config=config["augmentations"]["specaugment"],
        class_to_idx=class_to_idx,
        sample_rate=config["audio"]["sample_rate"],
        duration=config["audio"]["duration"],
        train=False,
    )

    train_loader = _make_loader(train_ds, config, shuffle=True)
    val_loader = _make_loader(val_ds, config, shuffle=False)

    return train_loader, val_loader, class_to_idx


def train_one_fold(
    config,
    fold_idx: int,
    run_stamp: str | None = None,
) -> dict[str, object]:
    n_splits = int(config["train"]["n_splits"])
    if not 0 <= fold_idx < n_splits:
        raise ValueError(
            f"fold_idx must be in [0, {n_splits - 1}], got {fold_idx}"
        )

    if run_stamp is None:
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    seed_everything(int(config["train"]["seed"]) + fold_idx)
    train_loader, val_loader, class_to_idx = build_loaders(config, fold_idx)
    fold_num = fold_idx + 1

    print(f"fold: {fold_num}/{n_splits}")
    print("train rows:", len(train_loader.dataset))
    print("validation rows:", len(val_loader.dataset))
    print("num classes:", len(class_to_idx))

    model = BirdClefEfficientNet(
        num_classes=len(class_to_idx),
        model_name=config["train"]["efficentnet_name"],
        is_pretrained=True,
    ).to(config["device"])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    scaler = GradScaler(enabled=config["device"].startswith("cuda"))

    best_val_auc = float("nan")
    best_val_auc_for_compare = float("-inf")
    best_ckpt_path = None
    history = []

    for epoch in range(config["train"]["epochs"]):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            config["device"],
        )

        val_loss, val_auc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            config["device"],
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{config['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "run_stamp": run_stamp,
            "fold_idx": fold_idx,
            "n_splits": n_splits,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "class_to_idx": class_to_idx,
            "config": config,
        }

        epoch_ckpt_path = (
            f"/kaggle/working/{run_stamp}_fold_{fold_num:02d}_epoch_{epoch + 1:02d}.pth"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "epoch_ckpt_path": epoch_ckpt_path,
            }
        )

        if not np.isnan(val_auc) and val_auc > best_val_auc_for_compare:
            best_val_auc_for_compare = val_auc
            best_val_auc = val_auc
            best_ckpt_path = (
                f"/kaggle/working/{run_stamp}_fold_{fold_num:02d}_best.pth"
            )
            torch.save(checkpoint, best_ckpt_path)
            print(f"Saved new best checkpoint: {best_ckpt_path}")

    print(f"Completed fold {fold_num}/{n_splits} | best_val_auc={best_val_auc:.4f}")
    return {
        "fold_idx": fold_idx,
        "fold_num": fold_num,
        "best_val_auc": best_val_auc,
        "best_ckpt_path": best_ckpt_path,
        "history": history,
    }


### INFERENCE
#############


def _parse_fold_best_checkpoint_path(checkpoint_path: Path) -> tuple[str, int]:
    name = checkpoint_path.name
    if not name.endswith("_best.pth") or "_fold_" not in name:
        raise ValueError(
            f"Checkpoint does not match fold-best naming convention: {checkpoint_path}"
        )

    stem = name[: -len("_best.pth")]
    run_stamp, fold_num_text = stem.rsplit("_fold_", maxsplit=1)
    if not run_stamp or not fold_num_text.isdigit():
        raise ValueError(
            f"Checkpoint does not match fold-best naming convention: {checkpoint_path}"
        )

    return run_stamp, int(fold_num_text)


def discover_fold_best_checkpoints(search_root: Path) -> list[Path]:
    checkpoint_candidates = sorted(search_root.rglob("*_fold_*_best.pth"))
    if not checkpoint_candidates:
        raise FileNotFoundError(
            f"No *_fold_*_best.pth checkpoints found under {search_root}"
        )

    grouped_candidates: dict[str, list[tuple[int, Path]]] = {}
    for checkpoint_path in checkpoint_candidates:
        run_stamp, fold_num = _parse_fold_best_checkpoint_path(checkpoint_path)
        grouped_candidates.setdefault(run_stamp, []).append((fold_num, checkpoint_path))

    if len(grouped_candidates) > 1:
        grouped_lines = []
        for run_stamp, items in sorted(grouped_candidates.items()):
            grouped_lines.append(f"{run_stamp}:")
            grouped_lines.extend(
                f"- {path}" for _, path in sorted(items, key=lambda item: item[0])
            )
        raise RuntimeError(
            "Multiple fold-checkpoint runs found under "
            f"{search_root}. Attach exactly one run:\n"
            + "\n".join(grouped_lines)
        )

    _, run_items = next(iter(grouped_candidates.items()))
    sorted_items = sorted(run_items, key=lambda item: item[0])
    fold_numbers = [fold_num for fold_num, _ in sorted_items]
    if len(fold_numbers) != len(set(fold_numbers)):
        raise RuntimeError(
            "Duplicate fold numbers found:\n"
            + "\n".join(f"- {path}" for _, path in sorted_items)
        )

    return [path for _, path in sorted_items]


def _extract_infer_config(checkpoint, infer_batch_size: int) -> tuple[dict, dict[str, int]]:
    checkpoint_config = checkpoint.get("config", {})
    if not checkpoint_config:
        raise ValueError("Checkpoint config empty.")

    class_to_idx = checkpoint["class_to_idx"]
    infer_config = {
        "model_name": checkpoint_config["train"]["efficentnet_name"],
        "sample_rate": checkpoint_config["audio"]["sample_rate"],
        "duration": checkpoint_config["audio"]["duration"],
        "infer_batch_size": infer_batch_size,
        "mel_spectrogram": checkpoint_config["mel_spectrogram"],
    }
    return infer_config, class_to_idx


def load_checkpoint_ensemble(
    checkpoint_paths: list[Path],
    device: str = "cpu",
    infer_batch_size: int = 24,
) -> tuple[list[nn.Module], dict[str, int], dict]:
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths cannot be empty")

    checkpoints = [torch.load(path, map_location=device) for path in checkpoint_paths]
    infer_config, class_to_idx = _extract_infer_config(
        checkpoints[0],
        infer_batch_size=infer_batch_size,
    )

    expected_n_splits = checkpoints[0].get("n_splits")
    fold_indices = []
    models = []

    for checkpoint_path, checkpoint in zip(checkpoint_paths, checkpoints):
        current_infer_config, current_class_to_idx = _extract_infer_config(
            checkpoint,
            infer_batch_size=infer_batch_size,
        )

        if current_class_to_idx != class_to_idx:
            raise RuntimeError(
                "class_to_idx mismatch across ensemble checkpoints:\n"
                f"- {checkpoint_paths[0]}\n"
                f"- {checkpoint_path}"
            )

        if current_infer_config != infer_config:
            raise RuntimeError(
                "Inference config mismatch across ensemble checkpoints:\n"
                f"- {checkpoint_paths[0]}\n"
                f"- {checkpoint_path}"
            )

        fold_idx = checkpoint.get("fold_idx")
        if fold_idx is None:
            raise ValueError(f"Checkpoint missing fold_idx: {checkpoint_path}")
        fold_indices.append(int(fold_idx))

        current_n_splits = checkpoint.get("n_splits")
        if expected_n_splits is not None and current_n_splits != expected_n_splits:
            raise RuntimeError(
                "n_splits mismatch across ensemble checkpoints:\n"
                f"- {checkpoint_paths[0]}\n"
                f"- {checkpoint_path}"
            )

        model = BirdClefEfficientNet(
            num_classes=len(class_to_idx),
            model_name=infer_config["model_name"],
            is_pretrained=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    if len(fold_indices) != len(set(fold_indices)):
        raise RuntimeError(
            "Duplicate fold_idx values in ensemble checkpoints:\n"
            + "\n".join(f"- {path}" for path in checkpoint_paths)
        )

    if expected_n_splits is not None:
        expected_fold_indices = set(range(int(expected_n_splits)))
        observed_fold_indices = set(fold_indices)
        if observed_fold_indices != expected_fold_indices:
            missing = sorted(expected_fold_indices - observed_fold_indices)
            extra = sorted(observed_fold_indices - expected_fold_indices)
            details = []
            if missing:
                details.append(f"missing fold_idx values: {missing}")
            if extra:
                details.append(f"unexpected fold_idx values: {extra}")
            raise RuntimeError(
                "Incomplete fold ensemble:\n"
                + "\n".join(details)
                + "\ncheckpoints:\n"
                + "\n".join(f"- {path}" for path in checkpoint_paths)
            )

    return models, class_to_idx, infer_config

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
def predict_specs(specs: torch.Tensor, model: nn.Module, config) -> np.ndarray:
    preds = []
    model_device = next(model.parameters()).device

    for start in range(0, len(specs), config["infer_batch_size"]):
        batch = specs[start:start + config["infer_batch_size"]].to(model_device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_file(audio_path: Path, end_seconds: list[int], model, config) -> np.ndarray:
    specs = build_segment_tensor(audio_path, end_seconds, config)
    return predict_specs(specs, model, config)


@torch.no_grad()
def predict_file_ensemble(
    audio_path: Path,
    end_seconds: list[int],
    models: list[nn.Module],
    config,
) -> np.ndarray:
    if not models:
        raise ValueError("models cannot be empty")

    specs = build_segment_tensor(audio_path, end_seconds, config)
    ensemble_probs = np.zeros((len(end_seconds), 0), dtype=np.float32)

    for model_idx, model in enumerate(models):
        model_probs = predict_specs(specs, model, config)
        if model_idx == 0:
            ensemble_probs = np.zeros_like(model_probs, dtype=np.float32)
        ensemble_probs += model_probs.astype(np.float32)

    ensemble_probs /= len(models)
    return ensemble_probs
