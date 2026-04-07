# Kaggle CLI Workflow

## Shared utilities dataset
- First publish: `kaggle datasets create -p kaggle/utilities_dataset`
- Later updates: `kaggle datasets version -p kaggle/utilities_dataset -m "update utilities"`

## Training kernel
- Push the utilities dataset first so the kernel can import it
- Wave mixup and SpecAugment are controlled in `kaggle/training/training.py` under `CONFIG["augmentations"]`
- Push: `kaggle kernels push -p kaggle/training`
- Status: `kaggle kernels status shawnliu30/birdclef-2026-training-script`
- Download outputs: `kaggle kernels output shawnliu30/birdclef-2026-training-script -p out/training`

## Inference kernel
- Push the utilities dataset first so the kernel can import it
- Push: `kaggle kernels push -p kaggle/inference`
- Status: `kaggle kernels status shawnliu30/birdclef-2026-inference-script`
- Download outputs: `kaggle kernels output shawnliu30/birdclef-2026-inference-script -p out/inference`

## Submit
- `kaggle competitions submit -c birdclef-2026 -f out/inference/submission.csv -m "your message"`

## Notes
- Training includes the shared utilities dataset plus the four WAV shard datasets directly in `kernel-metadata.json`.
- Inference currently depends on the latest output of `shawnliu30/birdclef-2026-training-script` via `kernel_sources`.
- Both kernels import `birdclef_2026_utilities.py` from the attached `shawnliu30/birdclef-2026-utilities` dataset on Kaggle and fall back to `kaggle/utilities_dataset` locally.
