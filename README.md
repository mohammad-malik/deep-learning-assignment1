# Facial Affect Recognition Pipeline

This repository implements a complete, modular PyTorch pipeline for facial expression classification and valence/arousal regression using the provided Affect dataset (cropped RGB faces with landmark annotations).

## Features
- Stratified dataset preparation with cached metadata for rapid reuse.
- Configurable data augmentation (horizontal flip, colour jitter, rotation) and landmark normalisation.
- Multi-task CNN architecture with a shared backbone and dedicated heads for categorical (8 expressions) and continuous (valence, arousal) outputs.
- Landmark encoder branch that can be toggled per experiment.
- Support for multiple ImageNet-pretrained backbones (ResNet-18, EfficientNet-B0, MobileNet-V3) with optional freezing.
- Comprehensive evaluation suite covering Accuracy, Macro F1, Cohen's Kappa, Krippendorff's Alpha, AUROC, AUPR, RMSE, Pearson correlation, SAGR, and CCC.
- Training loop with AMP, gradient clipping, cosine/step/plateau schedulers, checkpointing, and history export.
- CLI scripts for training, evaluation, and qualitative visualisation of predictions.

## Project Layout
```
src/
  config.py              # Experiment configuration dataclasses
  data/                  # Dataset parsing, transforms, splits, loaders
  metrics/               # Classification & regression metrics
  models/                # Backbone factory and multi-task heads
  training/              # Trainer implementation
  utils/                 # Logging, config helpers, IO utilities
scripts/
  train.py               # Main training entry point
  evaluate.py            # Standalone evaluation on any split
  visualize_predictions.py  # Grid of correct/incorrect samples
Dataset/
  images/                # 224x224 RGB crops
  annotations/           # *.npy labels (expression, valence, arousal, landmarks)
```

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: The scripts expect the dataset to reside in Dataset/ (images + annotations). Override the path with --dataset-root if it lives elsewhere.

## Training
Run two baseline models (ResNet-18 and EfficientNet-B0) to satisfy the assignment requirement of comparing at least two CNNs:

```bash
# ResNet-18 baseline with landmarks
python scripts/train.py \
  --dataset-root Dataset \
  --backbone resnet18 \
  --output-dir experiments/resnet18

# EfficientNet-B0 baseline without freezing
python scripts/train.py \
  --dataset-root Dataset \
  --backbone efficientnet_b0 \
  --output-dir experiments/efficientnet_b0
```

Key flags:
- --no-landmarks disables the landmark branch.
- --freeze-backbone keeps pretrained features fixed for the first phase.
- --mixed-precision enables automatic mixed precision.
- --config path/to/experiment.yml loads a YAML/JSON configuration (same keys as ExperimentConfig).

All runs create an output folder containing:
- checkpoints/best.pt – best validation checkpoint.
- history.json – epoch-wise train/val loss and accuracy.
- metrics.json – final validation/test metrics.
- config_used.json – fully resolved configuration snapshot.

## Evaluation
Evaluate any checkpoint on the train/val/test split:
```bash
python scripts/evaluate.py \
  --config experiments/resnet18/config_used.json \
  --checkpoint experiments/resnet18/checkpoints/best.pt \
  --split test
```
The script prints summary metrics and drops a *.evaluation.json file alongside the checkpoint.

## Qualitative Results
Generate a qualitative grid with correctly and incorrectly classified faces:
```bash
python scripts/visualize_predictions.py \
  --config experiments/resnet18/config_used.json \
  --checkpoint experiments/resnet18/checkpoints/best.pt \
  --split val \
  --num-images 12 \
  --output experiments/resnet18/qualitative.png
```
Each tile shows the ground-truth / predicted expression, plus valence & arousal targets vs. predictions.

## Extending the Pipeline
- **Additional backbones**: add new branches in src/models/factory.py and reference them via --backbone.
- **Custom schedulers/optimisers**: adjust uild_scheduler in scripts/train.py or wire new components in src/training/trainer.py.
- **Metrics**: extend src/metrics/ with new computations and have the trainer log them.
- **Hyperparameter sweeps**: the CLI accepts YAML configs, so you can create multiple config files for automated experiments.

## Reproducibility
All scripts call seed_everything to fix RNG seeds (Python, NumPy, PyTorch). Set a custom seed through --seed.
l `seed_everything` to fix RNG seeds (Python, NumPy, PyTorch). Set a custom seed through `--seed`.
l `seed_everything` to fix RNG seeds (Python, NumPy, PyTorch). Set a custom seed through `--seed`.

## License
This codebase is provided for educational use within the deep learning assignment context.
