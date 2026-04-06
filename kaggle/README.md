# Kaggle CLI Workflow

## First-time setup
1. Create a Kaggle API token from your Kaggle account settings.
2. Save it to `~/.kaggle/kaggle.json` and set permissions to `600`.

## Training kernel
- Sync shared utilities first if you edited `src/birdclef_2026_utilities.py`: `python scripts/sync_kaggle_utilities.py`
- Push: `kaggle kernels push -p kaggle/training`
- Status: `kaggle kernels status shawnliu30/birdclef-2026-training`
- Download outputs: `kaggle kernels output shawnliu30/birdclef-2026-training -p out/training`

## Inference kernel
- Sync shared utilities first if you edited `src/birdclef_2026_utilities.py`: `python scripts/sync_kaggle_utilities.py`
- Push: `kaggle kernels push -p kaggle/inference`
- Status: `kaggle kernels status shawnliu30/birdclef-2026-inference`
- Download outputs: `kaggle kernels output shawnliu30/birdclef-2026-inference -p out/inference`

## Submit
- `kaggle competitions submit -c birdclef-2026 -f out/inference/submission.csv -m "your message"`

## Notes
- Training includes the four WAV shard datasets directly in `kernel-metadata.json`.
- Inference currently depends on the latest output of `shawnliu30/birdclef-2026-training` via `kernel_sources`.
- The shared utility module is vendored into both kernel folders so it no longer needs to be attached as a separate notebook source.
