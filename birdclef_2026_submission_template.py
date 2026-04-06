#!/usr/bin/env python3
# %%
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


# %%
class AudioToSpec:
    def __init__(self, mel_config):
        self.mel_config = mel_config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel_config["sample_rate"],
            n_fft=mel_config["n_fft"],
            hop_length=mel_config["hop_length"],
            win_length=mel_config["win_length"],
            n_mels=mel_config["n_mels"],
            f_min=mel_config["f_min"],
            f_max=mel_config["f_max"],
            power=mel_config["power"],
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=mel_config["top_db"],
        )

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        spec = self.mel_transform(wave)
        spec = self.db_transform(spec)
        spec = (spec + self.mel_config["top_db"]) / self.mel_config["top_db"]
        spec = torch.clamp(spec, 0.0, 1.0)
        return spec


class BirdClefEfficientNet(nn.Module):
    def __init__(self, num_classes: int, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


# %%
COMP_DIR = Path("/kaggle/input/birdclef-2026")
TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUBMISSION = COMP_DIR / "sample_submission.csv"

# Replace this with your attached model dataset path.
CHECKPOINT_PATH = Path("/kaggle/input/YOUR_MODEL_DATASET/efficientnet_b0_best.pth")

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


# %%
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
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

audio_to_spec = AudioToSpec(INFER_CONFIG["mel_spectrogram"])
num_samples = INFER_CONFIG["sample_rate"] * INFER_CONFIG["duration"]

print("sample submission rows:", len(submission))
print("submission columns:", len(target_columns))
print("checkpoint classes:", len(class_to_idx))


# %%
def build_segment_tensor(audio_path: Path, end_seconds: list[int]) -> torch.Tensor:
    wave, sr = sf.read(audio_path)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    wave = wave.astype(np.float32)

    if sr != INFER_CONFIG["sample_rate"]:
        raise ValueError(f"Expected {INFER_CONFIG['sample_rate']}, got {sr}")

    specs = []
    for end_sec in end_seconds:
        start_sample = max(0, (int(end_sec) - INFER_CONFIG["duration"]) * INFER_CONFIG["sample_rate"])
        chunk = wave[start_sample:start_sample + num_samples]

        if len(chunk) < num_samples:
            chunk = np.pad(chunk, (0, num_samples - len(chunk)))
        elif len(chunk) > num_samples:
            chunk = chunk[:num_samples]

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
        specs.append(audio_to_spec(chunk_tensor).unsqueeze(0))

    return torch.stack(specs, dim=0)


@torch.no_grad()
def predict_file(audio_path: Path, end_seconds: list[int]) -> np.ndarray:
    specs = build_segment_tensor(audio_path, end_seconds)
    preds = []

    for start in range(0, len(specs), INFER_CONFIG["infer_batch_size"]):
        batch = specs[start:start + INFER_CONFIG["infer_batch_size"]]
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)

    return np.concatenate(preds, axis=0)


# %%
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
    probs = predict_file(audio_path, end_seconds)

    output = np.zeros((len(group_df), len(target_columns)), dtype=np.float32)
    for model_idx, submission_idx in model_to_submission:
        output[:, submission_idx] = probs[:, model_idx]

    submission.loc[group_df.index, target_columns] = output


# %%
submission = submission.drop(columns=["file_id", "end_sec"])
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("saved submission.csv")
