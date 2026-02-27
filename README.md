# Vision Transformer (ViT) for Image Classification
Implemented a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset, utilizing image patching and self-attention mechanisms.

An implementation of the groundbreaking Vision Transformer (ViT) architecture, which applies transformer designs—originally for NLP—to computer vision tasks by treating images as sequences of patches.

## Architecture Highlights
- **Patching:** Resizes input images to $224\times224$ and splits them into $16\times16$ non-overlapping patches, resulting in a sequence of 196 patches.
- **Embeddings:** Includes a Patch Embedding layer for linear projection and learnable Positional Encodings to retain spatial information.
- **Transformer Encoder:** Features a stack of layers with Multi-Head Self-Attention and Multi-Layer Perceptrons (MLP) using GELU activation.

## Dataset & Training
- **Dataset:** CIFAR-10 (60,000 images across 10 classes).
- **Optimizer:** AdamW with weight decay.
- **Hyperparameters:** 10 epochs, 256 batch size, and a dropout rate of 0.1.

## Results
- **Model Parameters:** ~1.2 Million trainable parameters.
- **Performance:** Achieved competitive results for image classification by capturing long-range dependencies and global context.
