# Vision Transformer from Scratch — CIFAR-10

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

A Vision Transformer (ViT) implemented from scratch in TensorFlow/Keras for CIFAR-10 image classification, evaluated honestly on a held-out test set and **benchmarked against a CNN baseline**.

**Headline result:** the from-scratch ViT reaches **80.1% test accuracy** — but a small CNN with **7.6× fewer parameters** reaches **84.5%**. That gap is the point of this project: it's a hands-on demonstration of why ViTs are data-hungry and underperform CNNs at small scale without large-scale pretraining.

| Model | Parameters | Test accuracy (10k held-out) |
|---|---|---|
| ViT (from scratch, augmented) | 1,206,282 | **80.09%** (top-5: 98.96%) |
| CNN baseline | 158,122 | **84.53%** |

---

## Why this project is worth a look

Most from-scratch ViT demos report a single accuracy number with no baseline and — often — no clean test set. This one does three things differently:

1. **Honest evaluation.** Uses the official CIFAR-10 **50,000 / 10,000** train/test split. A 5,000-image validation set is carved out of *train only*; the 10,000-image test set is untouched during training and scored exactly once. (An earlier version of this notebook accidentally trained and evaluated on the same data — that bug was found and fixed, which is the reason evaluation is called out so explicitly here.)
2. **A baseline for context.** A single accuracy number means little in isolation. The ViT is compared against a CNN trained on the identical split, so the result can actually be interpreted.
3. **An explanation, not just a number.** The CNN wins with a fraction of the parameters — and the README explains *why* (below), which is the actual learning outcome.

---

## Architecture

Standard ViT, operating on native-resolution CIFAR images (no wasteful upscaling):

| Component | Value |
|---|---|
| Input | 32×32×3 (native resolution) |
| Patches | **4×4 → 64 patches** |
| Patch embedding | linear projection to `projection_dim = 64` |
| Positional encoding | learnable (`Embedding`) |
| Transformer blocks | **8**, each: LayerNorm → Multi-Head Self-Attention (4 heads) → residual → LayerNorm → MLP (GELU) → residual |
| MLP hidden units | [128, 128] |
| Dropout | 0.1 |
| Classifier head | LayerNorm → Flatten → Dropout → MLP → Dense(10) |
| Total parameters | ~1.2M |

---

## Training setup

| Item | Value |
|---|---|
| Augmentation | pad 32→40 + random-crop to 32, random horizontal flip (train only) |
| Optimizer | AdamW (weight decay 1e-4) |
| LR schedule | cosine decay with 5-epoch linear warmup (1e-4 → 1e-3 → decay) |
| Loss | categorical cross-entropy (from logits) |
| Batch size | 256 |
| Epochs | up to 100, early stopping (patience 15) on validation accuracy, best weights restored |
| Hardware | single GPU (Colab T4), ~16–17 s/epoch |

Training reached ~87% train / ~81.5% validation accuracy; the ~7-point train/val gap indicates mild overfitting, expected for a from-scratch transformer on a small dataset even with augmentation.

---

## Why the CNN wins (the takeaway)

The CNN beats the ViT (84.5% vs 80.1%) using **7.6× fewer parameters**. This is the well-documented data-efficiency gap:

- **CNNs have built-in inductive biases** — locality and translation equivariance — that match how images work. They get these "for free" by construction.
- **ViTs have almost no built-in image prior.** They must *learn* spatial relationships from data via self-attention, which needs far more data (or large-scale pretraining) to pay off.
- On a small dataset like CIFAR-10 (50k images), that trade-off favors the CNN. The famous ViT results that *beat* CNNs came from pretraining on datasets 1,000× larger (ImageNet-21k, JFT-300M), then fine-tuning.

So the ViT underperforming here isn't a bug — it's the expected behavior, and reproducing it from scratch is the educational value.

---

## Results & figures

<img width="1136" height="470" alt="vit_graph" src="https://github.com/user-attachments/assets/b422fc41-c710-4d15-8128-1d7c85b27479" />
train/val accuracy and loss per epoch
<img width="717" height="633" alt="vit_confusion_metrix" src="https://github.com/user-attachments/assets/a16a7364-e03f-40ab-b94c-105007557bfd" />
per-class performance on the test set

---

## Reproduce

Open the notebook in Google Colab (**Runtime → Change runtime type → GPU**) and run all cells. It will:

1. Load CIFAR-10 with the official train/test split (via `keras.datasets.cifar10`).
2. Train the ViT with augmentation and a cosine schedule.
3. Evaluate once on the held-out test set; plot curves + confusion matrix.
4. Train the CNN baseline on the same split and print the head-to-head table.

> Set `epochs = 30` for a faster run — the conclusion (CNN > from-scratch ViT) holds well before 100 epochs.

---

## Limitations & next steps

- **Small-data regime.** No pretraining, so the ViT is capped well below its potential. Fine-tuning a pretrained ViT would be the honest way to push past the CNN — and would directly demonstrate the "ViTs need scale" point with numbers.
- **Classifier head uses `Flatten`** over all patch tokens rather than a `[CLS]` token or global average pooling; GAP would be lighter and is a natural next experiment.
- **Single run.** Metrics are from one seed; averaging over a few seeds would tighten the comparison.

## Tech stack

TensorFlow / Keras · NumPy · scikit-learn (confusion matrix) · Matplotlib · Google Colab

---

## Repository structure

```
.
├── Vision_Transformer.ipynb   # main notebook: data, ViT, training, eval, CNN baseline
├── README.md
├── requirements.txt
├── LICENSE
└── figures/                   # generated: training curves, confusion matrix
    ├── vit_training_curves.png
    └── vit_confusion_matrix.png
```

## Installation

```bash
pip install -r requirements.txt
```

Or just open the notebook in Google Colab — all dependencies are preinstalled there.

---

## References

- Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (ICLR 2021) — the original ViT paper. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009) — the CIFAR-10 dataset.

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

## Author

**Priyanshi Kochar** — [GitHub](https://github.com/Priyanshi965)
