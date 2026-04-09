# BirdCLEF Ideas Tracker

## Completed Foundation Work

### Joint Audio + Soundscape Training
- Status: completed
- Both `train_audio` and labeled `train_soundscapes` are present in both training and validation splits
- This replaced the earlier base-train plus soundscape-finetune setup

### Fast WAV Loading
- Status: completed
- `train_audio` is now read from preconverted WAV shards
- Partial reads avoid loading full files when only a fixed window is needed
- This made experiments dramatically faster and more practical

### Reproducible Baseline Selection
- Status: completed
- Per-epoch `train_loss`, `val_loss`, and `val_auc` are logged
- Best checkpoint is selected by `val_auc`
- The current baseline is now reliable enough to compare follow-up experiments against

## High Priority Next

### Wave-Domain MixUp
- Status: high priority
- Mix raw waveforms before mel conversion instead of mixing spectrograms
- Why: repeatedly called out in the BirdCLEF 2026 single-model thread as one of the highest-ROI changes; several `0.90+` reports use wave mixup specifically

### Longer Context Windows
- Status: high priority
- Move beyond plain 5-second training windows
- First candidate: `10s`
- Why: the thread strongly suggests long context is one of the main unlocks for high-LB single models, especially for temporal texture and sparse events

### Temporal / SED Modeling
- Status: high priority
- Add a real temporal head rather than staying a pure clip classifier
- Candidate directions:
- weakly supervised SED head
- clipwise + framewise max supervision
- multi-context head over several adjacent chunks
- Why: the strongest single-model results in the thread repeatedly point to temporal modeling as the main gap between plain timm baselines and `0.90+`

### Time-Shift TTA
- Status: high priority
- Run inference with no shift and `+/- 2.5s` shift, then average
- Why: repeatedly mentioned in the thread and the pasted SED plan as a strong low-risk inference improvement

### Priors
- Status: high priority
- Test species priors based on site and hour at inference time
- Why: at least one `0.92+` single-model report in the thread uses priors as a meaningful extra gain on top of the base model

### Grouped Multi-Label Stratified Validation
- Status: planned
- Goal: reduce leakage while keeping species distribution balanced across train/val
- Why: should make `val_auc` more stable and more representative than a plain grouped split

### SpecAugment
- Status: planned
- Add mild time masking + frequency masking on train only
- Why: low-complexity robustness gain; likely higher ROI than jumping architectures immediately

### Class Imbalance Sampling
- Status: planned
- Try class-balanced or partially balanced sampling rather than plain shuffle
- Why: competition metric weights classes evenly, while raw data is long-tailed
- Variants:
- `gamma = -0.5` style soft balancing
- fully balanced sampling
- duplicate/oversample rare classes to a floor

### Label Smoothing For Weak Labels
- Status: planned
- Apply small smoothing, especially on `train_audio`
- Why: weak labels over full recordings are noisy; smoothing can make training less brittle

### Top-N File-Level Postprocessing
- Status: planned
- Multiply each clip/class probability by the file-level top-1 probability for that class
- Why: explicitly confirmed in the pasted SED plan and still looks like one of the cleanest low-cost inference wins once the base model is stronger

### Temporal Smoothing Across Adjacent Clips
- Status: planned
- Smooth neighboring clip predictions with a small kernel
- Why: low-risk inference improvement for soundscape continuity

## Medium Priority Model Ideas

### BCE + Focal Blend
- Status: planned
- Replace plain `BCEWithLogitsLoss` with either focal BCE or BCE/focal mixture
- Why: often helps multi-label imbalance, especially many easy negatives

### CrossEntropy / SoftAUCLoss Progression
- Status: backlog
- Candidate SED path from the pasted plan:
- first train with `CrossEntropy + label_smoothing`
- later pseudo-label rounds with `SoftAUCLoss`
- Why: reported as strong in top 2025 SED systems
- Caution: this does not cleanly drop into the current "all labels, multi-label BCE" baseline; likely only appropriate if/when we move to a more SED-like dominant-label setup

### CE-Style Clip + Frame Supervision
- Status: high priority
- If/when the model moves to a SED head, test `CrossEntropy`-style supervision on clipwise logits plus framewise max logits
- Why: multiple strong BirdCLEF 2026 single-model reports in the thread use this recipe and report it outperforming simpler BCE setups

### Background Mixing
- Status: planned
- Mix in background-only or low-bird audio from soundscapes and/or ESC-50 style noise
- Why: directly targets PAM/domain robustness

### Random Filtering / EQ Augmentation
- Status: planned
- Apply simple random channel filtering / equalizer distortion
- Why: simulates microphone/device differences mentioned in the 2025 domain-shift paper

### Raw-Wave Augmentation Beyond MixUp
- Status: planned
- Try lightweight waveform-domain augmentation such as pink noise or time stretch
- Why: the thread explicitly called out raw-wave augmentation as effective, while noting that many public notebooks stop at mixup

### StochasticDepth / `drop_path_rate`
- Status: backlog
- Add `drop_path_rate` regularization, especially in noisier pseudo-label rounds
- Why: explicitly highlighted in the pasted plan as useful during pseudo-label training

### Better Crop Strategy For `train_audio`
- Status: planned
- Move beyond plain random 5-second crops
- Ideas:
- energy-biased crops
- multiple random crops per file across epochs

### SED Head On Top Of Current Backbone
- Status: planned
- Keep current mel/data pipeline, replace classifier head with a time-aware weakly supervised SED head
- Why: may help sparse bird events inside each window and is strongly reinforced by the BirdCLEF 2026 thread
- Risk: moderate architectural shift and more tuning complexity

### Multi-Context Head
- Status: backlog
- Instead of one fixed clip, train over several adjacent contexts and aggregate with a temporal head
- Example ideas from the thread:
- `3 x 5s`
- `4 x 5s`
- Why: several high-LB comments argue that temporal context is the key even when staying within CNN-style architectures

### Backbone Upgrade Within EfficientNet Family
- Status: backlog
- Try `tf_efficientnet_b2.ns_jft_in1k` as the first step-up from B0
- Why: the pasted plan suggests B2 is a manageable jump before larger architectures

### Alternative Backbone Families
- Status: backlog
- Explore `tf_efficientnetv2_s` or `eca_nfnet_l0` later for diversity / stronger single-model performance
- Why: repeatedly present in top 2025 solutions

## Semi-Supervised / Domain Adaptation

### Pseudo-Label Unlabeled Soundscapes
- Status: backlog
- Use the ~10k unlabeled soundscapes after the supervised baseline stabilizes
- Why: likely better ROI than supervised FT on only 66 labeled soundscape files

### Confidence-Filtered Pseudo Selection
- Status: backlog
- Candidate rule from 2025 paper:
- keep chunks whose max class probability exceeds `0.5`
- zero out class probabilities below `0.1`
- Why: preserves soft targets while filtering obvious noise

### OOF Pseudo-Labeling
- Status: backlog
- Generate pseudo-labels using models that did not train on that fold/subset, then refresh every round
- Why: the pasted plan strongly reinforces this as safer and stronger than static full-dataset pseudo labels

### PowerTransform Pseudo Labels
- Status: backlog
- Apply `p ** gamma` to teacher probabilities before the next student round
- Why: emphasized in the pasted plan as a key anti-collapse trick in iterative noisy-student training
- Note: most relevant once we have multi-round pseudo-labeling in place

### Multi-Model Agreement Filtering
- Status: backlog
- Accept pseudo labels only when SED and Perch-style teachers broadly agree on the top species
- Why: this was proposed in your pasted plan and fits your current hybrid setup well

### Mixed Original + Pseudo Training
- Status: backlog
- Do not simply append pseudo data 1:1
- Sample original and pseudo data with a dedicated strategy
- Candidate idea from 2025 paper:
- if a class exists in pseudo data, sample pseudo roughly 40% of the time
- Why: adapts to soundscapes without letting noisy pseudo labels dominate

### Iterative Pseudo-Labeling
- Status: backlog
- Train -> pseudo-label unlabeled soundscapes -> retrain -> refresh pseudo labels
- Why: standard domain adaptation path once single-round pseudo labels are decent

## Transfer Learning / Pretraining

### Stronger Audio Pretraining
- Status: backlog
- Explore in-domain pretraining or stronger pretrained checkpoints if available
- Why: 2025 paper reports strong gains from in-domain transfer learning

### Perch Probe Baseline
- Status: optional benchmark
- Train a light probe on Perch embeddings using the current split/pipeline
- Why: not mandatory, but still useful as a calibration point if we need to separate "model family limit" from "pipeline issue"

### Perch Distillation
- Status: backlog
- Use Perch logits or embeddings as a teacher for the timm/SED model
- Why: the thread makes clear that Perch-derived temporal/embedding systems are already very strong, so distillation may be higher ROI than blind architecture search

### Additional External Bird Data
- Status: backlog
- Consider Xeno-Canto / iNaturalist / prior BirdCLEF data later, with careful leak controls
- Why: cited as material to the strongest 2025 systems

## Efficiency / Infrastructure

### Faster Audio Access
- Status: completed
- `train_audio` now uses preconverted WAV shards plus fast partial reads
- Why it mattered: reduced wall-clock training time and removed a major data-loading bottleneck

### Shared Spectrogram Reuse At Inference
- Status: backlog
- Compute spectrograms once and reuse across models/folds/ensemble members
- Why: simple inference speedup if/when ensembling comes later

### Model Soup
- Status: planned
- Average weights from the last few good checkpoints to reduce checkpoint sensitivity
- Why: explicitly reinforced by the pasted SED plan and low-risk if multiple checkpoints are saved

## Validation / Analysis

### Error Analysis By Source
- Status: planned
- Break metrics out by:
- `train_audio` validation rows
- `train_soundscape` validation rows
- Why: helps detect whether changes only improve the easier source domain

### Alien Speech Filtering
- Status: backlog
- Identify and remove only clear recordist/computer speech segments from training audio
- Why: the 2025 paper suggests aggressive curation often hurt, but targeted removal of obvious alien speech may still help

## Inference / Postprocessing

### Top-N Temporal Smoothing
- Status: planned
- For each species within a 1-minute soundscape, multiply each chunk score by the average of that species' top-N chunk scores in the same audio
- Candidate starting point: `N = 1`
- Why: in the 2025 paper, Top-N smoothing was the most consistently helpful postprocessing method

### Mean Temporal Smoothing
- Status: backlog
- Multiply each chunk score by the species' mean score across the whole audio
- Why: weaker than Top-N in the 2025 paper, but easy to test

### Convolution Over Neighbor Chunks
- Status: planned
- Smooth per-species chunk predictions with a small temporal kernel
- Why: tries to capture repeating vocalization patterns; lower priority because it underperformed Top-N in the paper

### Species Frequency Gap Analysis
- Status: backlog
- Compare species prevalence in train vs labeled/unlabeled soundscapes
- Why: the 2025 paper found major train/soundscape mismatch; useful for prioritizing pseudo-labeling and rare-class work

### Non-Bird Sub-Pipeline
- Status: backlog
- Consider a dedicated amphibian/insect pathway if 2026 evaluation mix or error analysis suggests very different behavior
- Why: the pasted plan highlights this as a niche but real gain when non-bird taxa matter

## Ensemble / Search

### Optuna Ensemble Weight Search
- Status: backlog

## Thread Takeaways
- Multiple competitors report `0.90+` single-model public LB without unlabeled data
- The common pattern is not "one magic backbone"; it is:
- `train_audio + labeled soundscapes`
- temporal / SED-style modeling
- longer context than plain 5-second clip classification
- wave-domain mixup
- light but targeted inference tricks such as time-shift TTA and priors
- Main implication for this project:
- the biggest remaining gap is probably temporal/context modeling, not generic loss tinkering or another small backbone swap
- Learn ensemble weights on the holdout rather than hand-tuning
- Why: clearly useful once you go beyond a few models, but not needed for the current single-model phase

### `min()` vs `mean()` Ensemble Reduction
- Status: backlog
- For CE-trained model families, compare `min()` vs `mean()` aggregation
- Why: explicitly called out in the pasted plan as a historical trick, but only relevant later once CE-trained ensembles exist

## Things To Avoid Repeating

### Separate Supervised Soundscape Fine-Tuning
- Status: deprioritized
- Why: with only 66 labeled soundscape files, it repeatedly hurt leaderboard performance and likely overfit correlated environments
