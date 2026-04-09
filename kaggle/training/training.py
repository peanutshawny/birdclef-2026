# Auto-generated from birdclef-2026-training (8).ipynb for local Kaggle CLI workflow.

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

import torch

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


UTILS_MODULE = "birdclef_2026_utilities.py"
UTILS_SEARCH_DIRS = [
    Path("/kaggle/input/datasets/shawnliu30/birdclef-2026-utilities-dataset"),
    Path("/kaggle/input/birdclef-2026-utilities-dataset"),
    Path(__file__).resolve().parents[1] / "utilities_dataset",
]

for utils_dir in UTILS_SEARCH_DIRS:
    if (utils_dir / UTILS_MODULE).exists():
        print(f"Found {UTILS_MODULE} in {utils_dir}, adding to sys.path")
        sys.path.insert(0, str(utils_dir))
        break
else:
    raise FileNotFoundError(
        f"Could not find {UTILS_MODULE} in any of: {UTILS_SEARCH_DIRS}"
    )

from birdclef_2026_utilities import (
    train_one_fold,
)


BASELINE_MEL_SPECTROGRAM = {
    "sample_rate": 32000,
    "n_mels": 128,
    "n_fft": 1024,
    "hop_length": 320,
    "win_length": 1024,
    "f_min": 40,
    "f_max": 15000,
    "top_db": 80,
    "power": 2.0,
    "normalization": "top_db",
    "resize_shape": None,
}

HGNET_MEL_SPECTROGRAM = {
    "sample_rate": 32000,
    "n_mels": 256,
    "n_fft": 2048,
    "hop_length": 313,
    "win_length": 626,
    "f_min": 20,
    "f_max": None,
    "top_db": 80,
    "power": 2.0,
    "center": True,
    "pad_mode": "reflect",
    "norm": "slaney",
    "mel_scale": "htk",
    "resize_shape": (256, 256),
    "normalization": "sample_minmax",
}


CONFIG = {
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

    "mel_spectrogram": BASELINE_MEL_SPECTROGRAM,
    "train_columns": {
        "audio_col": "path",
        "primary_label_col": "primary_label",
        "secondary_label_col": "secondary_labels",
    },
    "audio": {
        "sample_rate": 32000,
        "duration": 5,
    },
    "augmentations": {
        "wave_mixup": {
            "enabled": False,
            "probability": 0.3,
            "alpha": 0.2,
        },
        "specaugment": {
            "enabled": False,
            "probability": 0.5,
            "num_time_masks": 1,
            "time_mask_param": 16,
            "num_freq_masks": 1,
            "freq_mask_param": 8,
        },
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
        "n_splits": 4,
        "batch_size": 64,
        "seed": 42,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fold_results = []

for fold_idx in range(CONFIG["train"]["n_splits"]):
    fold_results.append(train_one_fold(CONFIG, fold_idx, run_stamp=run_stamp))

results_df = pd.DataFrame(fold_results)
print(results_df[["fold_num", "best_val_auc", "best_ckpt_path"]])

valid_auc = results_df["best_val_auc"].dropna()
if not valid_auc.empty:
    print(f"cv mean val_auc: {valid_auc.mean():.4f}")
    print(f"cv std val_auc: {valid_auc.std(ddof=0):.4f}")
