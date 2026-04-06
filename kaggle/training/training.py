# Auto-generated from birdclef-2026-training (8).ipynb for local Kaggle CLI workflow.

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime

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

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


from birdclef_2026_utilities import (
    BirdClefEfficientNet,
    train_one_epoch,
    validate_one_epoch,
    build_loaders,
)


CONFIG = {
    "experiment_name": "efficientnet_baseline_audio_soundscape_joint_training",
    "paths": {
        "train_audio": "/kaggle/input/competitions/birdclef-2026/train_audio",
    
        "train_audio_wav_root_0": "/kaggle/input/datasets/shawnliu30/birdclef-2026-wav-00/train_audio_wav_0",
        "train_audio_wav_root_1": "/kaggle/input/datasets/shawnliu30/birdclef-2026-wav-01/train_audio_wav_1",
        "train_audio_wav_root_2": "/kaggle/input/datasets/shawnliu30/birdclef-2026-wav-02/train_audio_wav_2",
        "train_audio_wav_root_3": "/kaggle/input/datasets/shawnliu30/birdclef-2026-wav-03/train_audio_wav_3",
    
        "train_csv": "/kaggle/input/competitions/birdclef-2026/train.csv",
        "train_soundscapes": "/kaggle/input/competitions/birdclef-2026/train_soundscapes",
        "train_soundscapes_labels_csv": "/kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv",
    },

    "mel_spectrogram":{
        "sample_rate": 32000,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 320,
        "win_length": 1024,
        "f_min": 40,
        "f_max": 15000,
        "top_db": 80,
        "power": 2.0
    },
    "train_columns": {
        "audio_col": "path",
        "primary_label_col": "primary_label",
        "secondary_label_col": "secondary_labels",
    },
    "audio": {
        "sample_rate": 32000,
        "duration": 5,
    },
    "dataloader": {
        "num_workers": 4,
        "shuffle": True,
    },
    "train": {
        "efficentnet_name": "tf_efficientnet_b0.ns_jft_in1k",
        "epochs": 8,
        "learning_rate": 4e-4,
        "weight_decay": 1e-2,
        "val_size": 0.2,
        "batch_size": 64,
        "seed": 42,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


train_loader, val_loader, class_to_idx = build_loaders(CONFIG)


print("train rows:", len(train_loader.dataset))
print("validation rows:", len(val_loader.dataset))
print("num classes:", len(class_to_idx))
print(train_loader.dataset.df["path"].head())


model = BirdClefEfficientNet(
    num_classes=len(class_to_idx),
    model_name=CONFIG["train"]["efficentnet_name"],
    is_pretrained=True
).to(CONFIG["device"])

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["train"]["learning_rate"],
    weight_decay=CONFIG["train"]["weight_decay"],
)
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["train"]["epochs"])
scaler = GradScaler(enabled=CONFIG["device"].startswith("cuda"))


best_val_auc = float("-inf")
best_ckpt_path = None
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = CONFIG["experiment_name"]

for epoch in range(CONFIG["train"]["epochs"]):
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
        CONFIG["device"],
    )

    val_loss, val_auc = validate_one_epoch(
        model,
        val_loader,
        criterion,
        CONFIG["device"],
    )

    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{CONFIG['train']['epochs']} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_auc={val_auc:.4f}"
    )

    checkpoint = {
        "experiment_name": exp_name,
        "epoch": epoch + 1,
        "stage": "joint_train",
        "run_stamp": run_stamp,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "class_to_idx": class_to_idx,
        "config": CONFIG,
    }

    epoch_ckpt_path = f"/kaggle/working/{exp_name}_{run_stamp}_epoch_{epoch + 1:02d}.pth"
    torch.save(checkpoint, epoch_ckpt_path)
    print(f"Saved epoch checkpoint: {epoch_ckpt_path}")

    if not np.isnan(val_auc) and val_auc > best_val_auc:
        best_val_auc = val_auc
        best_ckpt_path = f"/kaggle/working/{exp_name}_{run_stamp}_best.pth"
        torch.save(checkpoint, best_ckpt_path)
        print(f"Saved new best checkpoint: {best_ckpt_path}")