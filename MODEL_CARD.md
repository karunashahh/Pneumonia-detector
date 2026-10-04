# Model Card — Pneumonia Chest X-Ray Classifier

## Model description

- **Architecture:** MobileNetV2 (ImageNet pretrained) + custom classification head
  (GlobalAveragePooling → Dense(256, relu) → BatchNorm → Dropout(0.5) → Dense(1, sigmoid))
- **Input:** 160×160 RGB chest X-ray image
- **Output:** single probability, 0 (NORMAL) to 1 (PNEUMONIA)
- **Training:** two-phase — frozen backbone (5 epochs) then fine-tuned last two
  MobileNetV2 blocks at a lower learning rate (5 epochs), with class weighting for
  imbalance and augmentation (rotation, zoom, flip, brightness)
- **Decision threshold:** 0.3 (not the default 0.5) — chosen to favor recall over
  precision, on the basis that missing a real case is worse than a false alarm in a
  screening context

## Intended use

- **Intended:** educational/portfolio demonstration of a chest X-ray screening classifier,
  including a worked example of finding and correcting patient-level data leakage
- **Not intended:** clinical diagnosis, triage, or any use affecting real patient care.
  Not validated on any data outside the public chest_xray dataset, not reviewed by a
  radiologist, and not cleared by any regulatory body.

## Training data

- Kaggle chest_xray dataset — pediatric patients, single hospital source
  (Guangzhou Women and Children's Medical Center)
- Original folder-based split had patient-level leakage; re-split by patient ID for
  the results reported as "clean split"

## Performance

Clean patient-level split (final, stable result over 3 random seeds):
accuracy 89.2% ± 0.1%, ROC-AUC 0.950 ± 0.002.
Original leaky official split, for comparison: accuracy 76%, ROC-AUC 0.832.

## Known limitations

- **NORMAL recall is notably lower than PNEUMONIA recall** — the model is tuned to
  minimize missed pneumonia cases at the cost of more false alarms on healthy X-rays.
  This is a deliberate trade-off, not an oversight.
- **Single dataset source** — all training/test images come from one hospital's
  equipment and imaging protocol; performance on X-rays from other sources is untested.
- **Coarse explainability** — Grad-CAM heatmaps are blocky due to MobileNetV2's small
  final feature map at this input resolution.
- **Pediatric-only training data** — performance on adult X-rays is untested.

## Ethical considerations

This model should never be used as a sole basis for a clinical decision. It is
presented as a demonstration of an ML workflow, not as a validated medical device.
