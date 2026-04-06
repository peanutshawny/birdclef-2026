# Auto-generated from birdclef-2026-inference.ipynb for local Kaggle CLI workflow.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torchaudio
from tqdm.auto import tqdm

from birdclef_2026_utilities import (
    AudioToSpec,
    BirdClefEfficientNet,
    build_segment_tensor,
    predict_file
)


COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUBMISSION = COMP_DIR / "sample_submission.csv"

CHECKPOINT_PATH = Path("/kaggle/input/notebooks/shawnliu30/birdclef-2026-training/efficientnet_baseline_audio_soundscape_joint_training_20260406_194229_best.pth") # edit this

INFER_CONFIG = {
    "model_name": "tf_efficientnet_b0.ns_jft_in1k",
    "sample_rate": 32000,
    "duration": 5,
    "infer_batch_size": 24,
    "mel_spectrogram": {
        "sample_rate": 32000,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 320,
        "win_length": 1024,
        "f_min": 40,
        "f_max": 15000,
        "top_db": 80,
        "power": 2.0,
    },
}


torch.set_num_threads(max(1, os.cpu_count() or 1))
device = "cpu"

submission = pd.read_csv(SAMPLE_SUBMISSION)
target_columns = [col for col in submission.columns if col != "row_id"]

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {idx: label for label, idx in class_to_idx.items()}

model = BirdClefEfficientNet(
    num_classes=len(class_to_idx),
    model_name=checkpoint.get("config", {}).get("train", {}).get(
        "efficentnet_name",
        INFER_CONFIG["model_name"],
    ),
    is_pretrained=False
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("model loaded")
print("sample submission rows:", len(submission))
print("submission columns:", len(target_columns))
print("checkpoint classes:", len(class_to_idx))


submission["file_id"] = submission["row_id"].str.rsplit("_", n=1).str[0]
submission["end_sec"] = submission["row_id"].str.rsplit("_", n=1).str[1].astype(int)
submission[target_columns] = 0.0

submission_col_to_idx = {col: idx for idx, col in enumerate(target_columns)}
model_to_submission = [
    (model_idx, submission_col_to_idx[label])
    for label, model_idx in class_to_idx.items()
    if label in submission_col_to_idx
]

for file_id, group_df in tqdm(submission.groupby("file_id", sort=False), total=submission["file_id"].nunique()):
    audio_path = TEST_DIR / f"{file_id}.ogg"
    if not audio_path.exists():
        continue

    end_seconds = group_df["end_sec"].tolist()
    probs = predict_file(audio_path, end_seconds, model, INFER_CONFIG)

    output = np.zeros((len(group_df), len(target_columns)), dtype=np.float32)
    for model_idx, submission_idx in model_to_submission:
        output[:, submission_idx] = probs[:, model_idx]

    submission.loc[group_df.index, target_columns] = output


submission = submission.drop(columns=["file_id", "end_sec"])
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("saved submission.csv")