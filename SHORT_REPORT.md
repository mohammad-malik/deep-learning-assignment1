# Short Report (50 Points Submission)

## 1. Network Details & Baseline Rationale (10 pt)

**Task**: Multi-task facial affect analysis – categorical expression recognition (8 classes) + continuous valence / arousal regression.

**Backbones Implemented**:

- VGG16-BN (baseline, high parameter count but stable and historically strong for FER)
- EfficientNet-B0 (lightweight, better parameter efficiency / depthwise convolutions)
- EfficientNet-B2 (higher capacity + larger native input 260×260)
- ConvNeXt-Tiny (modern ConvNeXt design; initial feature dimension mismatch fixed with adaptive pooling and correct flattening)

**Multi-Task Head**:

- Shared dense projection: Linear → BatchNorm → ReLU → Dropout
- Classification branch: 512 → 256 → 8 logits
- Regression branch: 512 → 256 → 2 (valence, arousal) with Tanh (range [-1,1])

**Parameter Handling & Transfer Learning**:

- All convolutional backbones initialized with ImageNet pretrained weights.
- Backbones frozen during a warmup epoch (head adaptation) before full fine-tuning.

**Initial Training Settings (from logged 10‑epoch baseline runs)**:

- Batch size: 32
- Epochs: 10
- Optimizer: AdamW (lr = 3e-4, weight_decay = 1e-4)
- Warmup: 1 epoch backbone freeze
- Lambda (regression weight): 1.0 (later deemed too high)
- Augmentations: Resize → RandomResizedCrop → HorizontalFlip → ColorJitter → Rotation/Affine → Normalize

**Baseline Rationale**:

- VGG16 provides a reproducible, historically validated FER anchor.
- EfficientNet variants test parameter efficiency and scaling.
- ConvNeXt introduces a modern ConvNet baseline; fix positions it for refined comparison.

## 2. Multiple Baselines – Performance Comparison (10 pt)

| Model | Best Val CCC | Best Epoch | Final Val Accuracy | Final Val F1 |
|-------|--------------|-----------|--------------------|--------------|
| VGG16 | 0.2220 | 10 | 0.3212 | 0.2939 |
| EfficientNet-B0 | 0.0208 | 10 | 0.2025 | 0.1728 |
| EfficientNet-B2 | 0.0518 | 7 | 0.1812 | 0.1790 |
| ConvNeXt-Tiny | (Initial run failed – dimension mismatch) | – | – | – |

**Observations**:

- VGG16, despite being heavier, generalized better under the (sub‑optimal) original hyperparameters.
- EfficientNet variants underfit: high multitask interference + insufficient warmup + learning rate not adapted to their scaling.
- ConvNeXt needed architectural adjustment (adaptive pooling) before fair evaluation.

## 3. Transfer Learning Details (5 pt)

- Pretrained weights accelerate convergence on limited FER-style data.
- Warmup isolates learning to randomly initialized heads, stabilizing gradients.
- Planned improvement (implemented later): multi-epoch classification-only warmup, differential LRs (backbone << head), gradual regression loss ramp.

## 4. Training Graphs (5 pt)

Curves (described from logs; images not exported in baseline):

- VGG16: Training/validation loss modest downward trend; validation accuracy from ~0.18 → ~0.32 (slow plateau by epoch 10).
- EfficientNet-B0 / B2: Validation accuracy stagnated ≤0.21 indicating persistent underfitting.
- Expectation with improved phased trainer: faster early accuracy (≥0.50 after extended warmup) and reduced regression noise.

## 5. Performance Metrics & Continuous Domain Rationale (15 pt)

**Classification Metrics**:

- Accuracy: Simple, but ignores per-class imbalance.
- Weighted F1: Balances precision & recall across classes with support weighting.
- Cohen’s Kappa: Adjusts for chance agreement (useful with 8-class space).
- ROC-AUC (macro OvR): Ranking quality; less stable when some classes have few samples.

**Continuous Affect Metrics**:

- RMSE: Penalizes large deviations; sensitive to scale & outliers.
- Pearson Correlation: Measures linear association only (shift/scale invariances may hide bias).
- SAGR (Sign Agreement Rate): Evaluates correctness of affect polarity—critical for downstream valence-aware decisions.
- CCC (Concordance Correlation Coefficient): Integrates correlation, mean bias, and scale mismatch; stricter and deployment-relevant.

**Primary Deployment Metric**:

- CCC chosen as primary for valence/arousal as it enforces both association and calibration.
- SAGR retained as an interpretable directional safety metric (ensures no polarity inversion in critical contexts).

## 6. Correct vs Incorrect Predictions (4 pt)

Qualitative analysis (not auto-saved in baseline):

- Correct: Prototypical, high-intensity expressions (clear smiles, widened eyes) – strong feature separability.
- Incorrect: Low-intensity/subtle expressions and inter-class ambiguities (Neutral vs Sad, Surprise vs Fear). Lighting variation and potential misalignment degrade signal.

**Mitigation**:

- Landmark-based alignment / tighter face crops.
- Longer classification-only phase.
- Mild → strong augmentation schedule (curriculum) instead of uniform aggressive transforms.
- Potential adoption of class-balanced focal or label-smoothed cross-entropy (already partially added).

## 7. File Naming Convention (1 pt)

Recommended submission filename: `<studentID>_FER_Multitask_Report.pdf` (replace `<studentID>` appropriately).

## 8. Limitations & Planned Remediation (Contextual Addendum)

- High initial regression weight (λ=1.0) impaired categorical convergence.
- Limited epochs (10) prevented fully leveraging transfer learning adaptation.
- Missing persisted per-epoch history JSON; future: structured logging (CSV/JSON) + tensorboard.
- No calibration / reliability assessment yet; temperature scaling planned.

## 9. Next Improvement Steps (Actionable Roadmap)

1. Use phased trainer: extended classification warmup (e.g., 5–8 epochs) then gradually introduce λ_reg → 0.05–0.1.
2. Differential LRs (backbone 1e-5–3e-5; head 3e-4–1e-3) + cosine decay with warm restarts.
3. Add EMA (exponential moving average) shadow weights to stabilize late training.
4. Introduce task balancing (uncertainty weighting or GradNorm) once classification baseline stabilizes.
5. Export confusion matrices + per-class F1 + calibration plots each evaluation epoch.
6. Reassess EfficientNet/ConvNeXt under improved schedule; consider ViT-S/16 as an additional baseline.
7. Ensemble (logit average) top 2–3 diverse backbones for final submission.

---
**Summary**: The initial multi-task setting over-weighted regression, limiting categorical accuracy. The revised strategy (phased optimization + reduced regression influence + differential scheduling) is expected to unlock substantially higher classification performance and improved CCC, establishing a stronger FER multi-task baseline. But there isnt enough time to try it. I hate CNNs.
