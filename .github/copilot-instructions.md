### Quick orientation — what this repo is

- This repo contains a Kaggle competition workflow for BirdCLEF 2026. Primary artifacts:
  - `kaggle/training` and `kaggle/inference` — kernel sources (pushed with `kaggle kernels push`).
  - `src/birdclef_2026_utilities.py` — canonical shared utilities (vendored into each kernel folder).
  - top-level training/inference examples: `birdclef_efficientnet_baseline.py`, `birdclef_2026_submission_template.py`.

### Big-picture architecture

- Data flow: CSVs (train.csv, train_soundscapes_labels.csv) -> loader builders (`build_loaders`) -> `BirdClefDataset` -> `AudioToSpec` (mel spectrogram) -> model (`BirdClefEfficientNet`) -> training/inference loops.
- Two training styles exist in the repo:
  - Multi-label joint audio + soundscape training (uses BCEWithLogitsLoss). See `kaggle/training/training.py` which uses `config['train']['efficentnet_name']`, `build_loaders`, and `BirdClefEfficientNet`.
  - Single-label baseline (CrossEntropy) shown in `birdclef_efficientnet_baseline.py` and `src/birdclef_2026_utilities.py` alternate implementations.
- Inference is performed by slicing each soundscape into 5s segments via `build_segment_tensor` and `predict_file`, then mapping model class indices to submission columns using `class_to_idx` saved in checkpoints.

### Key functions & files to reference (search these names)

- `src/birdclef_2026_utilities.py`
  - build_loaders(config) — creates train/val DataLoader and class_to_idx
  - BirdClefDataset, AudioToSpec — audio-to-spectrogram and dataset behavior
  - BirdClefEfficientNet — wrapper around timm EfficientNet (handles 1->3 channel repeat)
  - train_one_epoch, validate_one_epoch — training/validation loops using AMP/GradScaler
  - build_segment_tensor, predict_file — inference utilities
- `kaggle/training/training.py` — example joint training kernel; saves checkpoints to `/kaggle/working/` and keeps best by val AUC
- `kaggle/inference/inference.py` — example kernel that loads a checkpoint and writes `submission.csv`
- `scripts/sync_kaggle_utilities.py` — keep vendored kernel utilities in sync (run before pushing kernels)

### Conventions and gotchas (project-specific)

- Vendoring: Edit `src/birdclef_2026_utilities.py` locally, then run `python scripts/sync_kaggle_utilities.py` to propagate into `kaggle/*` before `kaggle kernels push`.
- Sharded WAV layout: training code expects four WAV roots keyed as `train_audio_wav_root_0`..`_3` in the `CONFIG['paths']`. Filenames are mapped to a shard by filename % 4.
- Label parsing: functions `_parse_secondary_labels` and `_parse_soundscape_labels` normalize many formats (stringified lists, semicolon-separated text, NaN). Use these when adding new label sources.
- Splitting: group-aware split uses `GroupShuffleSplit` on `group_id = source::audio_id` to avoid leaking audio between train/val.
- Multi-label vs single-label: joint training creates multi-hot targets (BCEWithLogitsLoss). Some scripts expect single-label (CrossEntropy). Check which loss/target shape the script uses.
- Sample rate & duration: default 32 kHz and 5s segments (mel config in files). If you change them, update `mel_spectrogram` and `audio`/`INFER_CONFIG` consistently.

### Developer workflows (concrete commands)

- Sync shared utilities into kernels (after editing `src/`):
  - `python scripts/sync_kaggle_utilities.py`
- Push training kernel:
  - `kaggle kernels push -p kaggle/training`
  - Check status: `kaggle kernels status <username>/birdclef-2026-training`
  - Download output: `kaggle kernels output <username>/birdclef-2026-training -p out/training`
- Push inference kernel similarly: `kaggle kernels push -p kaggle/inference`
- Submit (after inference produces `submission.csv`):
  - `kaggle competitions submit -c birdclef-2026 -f out/inference/submission.csv -m "message"`

### Dependencies (discoverable from code)

- Core Python libs used: numpy, pandas, soundfile (pysoundfile), timm, torch, torchaudio, scikit-learn.
- If running locally, ensure torch + torchaudio versions are CUDA-compatible with your machine.

### Quick examples to look at when implementing changes

- To add a new augmentation or spec variant: update `AudioToSpec` in `src/birdclef_2026_utilities.py` and run `scripts/sync_kaggle_utilities.py`.
- To change class ordering/labels: check `_build_class_to_idx` (it sorts class names), and remember `checkpoint['class_to_idx']` is saved in training checkpoints and relied on by inference.
- To change how audio windows are sampled: update `BirdClefDataset._read_partial_audio` and `BirdClefDataset._fix_length`.

If anything here is unclear or you'd like additional focus (e.g., tests, CI, or a CONTRIBUTING section), tell me which area to expand and I will iterate.  