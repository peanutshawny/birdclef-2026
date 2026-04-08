# Auto-generated from birdclef-2026-inference.ipynb for local Kaggle CLI workflow.

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

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
    BirdClefEfficientNet,
    predict_file
)


COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUBMISSION = COMP_DIR / "sample_submission.csv"

CHECKPOINT_SEARCH_ROOT = Path("/kaggle/input")
CHECKPOINT_CANDIDATES = sorted(CHECKPOINT_SEARCH_ROOT.rglob("*_best.pth"))
if not CHECKPOINT_CANDIDATES:
    raise FileNotFoundError(
        f"No *_best.pth checkpoint found under {CHECKPOINT_SEARCH_ROOT}"
    )
if len(CHECKPOINT_CANDIDATES) > 1:
    raise RuntimeError(
        "Multiple *_best.pth checkpoints found under "
        f"{CHECKPOINT_SEARCH_ROOT}:\n"
        + "\n".join(f"- {path}" for path in CHECKPOINT_CANDIDATES)
    )
CHECKPOINT_PATH = CHECKPOINT_CANDIDATES[0]

print(f"Using checkpoint: {CHECKPOINT_PATH}")
torch.set_num_threads(max(1, os.cpu_count() or 1))
device = "cpu"

submission = pd.read_csv(SAMPLE_SUBMISSION)
target_columns = [col for col in submission.columns if col != "row_id"]

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
checkpoint_config = checkpoint.get("config", {})

if not checkpoint_config:
    raise ValueError("Checkpoint config empty.")

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {idx: label for label, idx in class_to_idx.items()}

INFER_CONFIG = {
    "model_name": checkpoint_config["train"]["efficentnet_name"],
    "sample_rate": checkpoint_config["audio"]["sample_rate"],
    "duration": checkpoint_config["audio"]["duration"],
    "infer_batch_size": 24,
    "mel_spectrogram": checkpoint_config["mel_spectrogram"],
}

model = BirdClefEfficientNet(
    num_classes=len(class_to_idx),
    model_name=INFER_CONFIG["model_name"],
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
