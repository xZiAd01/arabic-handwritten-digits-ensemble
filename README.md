# Arabic Handwritten Digits Recognition — Ensemble Voting Model

## Overview
This project focuses on recognizing Arabic handwritten digits (0–9) using an **ensemble learning approach**.
Each team member independently designed and trained a neural network model, and the final prediction is obtained by combining all models using **hard voting** and **soft (probability-based) voting**.

The goal is to demonstrate how model diversity improves robustness and prediction reliability.

---

## Dataset
- **Type:** Arabic handwritten digit images (0–9)
- **Image size:** 28 × 28 grayscale
- **Structure:** Each digit stored in a separate folder (`0`–`9`)
- **Preprocessing:**
  - Grayscale conversion
  - Resizing to 28×28
  - Flattening to 784-dimensional vectors
  - Normalization to [0, 1]

> Dataset is expected to be placed inside `data/SmallDataset/`.

---

## Methodology

### Individual Models
Each team member trained a **custom Multi-Layer Perceptron (MLP)** from scratch using NumPy only (no deep-learning frameworks).

- Different architectures (depth & hidden units)
- Independent training processes
- Saved weights in `.npz` format

This design intentionally introduces **model diversity**, which is critical for effective ensembles.

---

### Ensemble Strategy

Two ensemble techniques are implemented:

#### 1. Hard Voting
- Each model predicts a class label
- Final prediction = majority vote

#### 2. Soft Voting
- Each model outputs class probabilities
- Probabilities are averaged
- Final prediction = class with highest average probability

Both methods are supported and compared.

---

## Results
- Individual models achieve competitive accuracy on the test set
- **Ensemble predictions outperform or stabilize individual model outputs**
- Soft voting generally provides smoother confidence estimates

*(Exact accuracy depends on dataset split and training seed.)*

---

## Project Structure
- individual_models/ # Training code for each team member
- ensemble/ # Voting logic and evaluation
- saved_models/ # Trained .npz model weights
- data/ # Dataset directory (not included if large)

---

## Installation
```bash
pip install -r requirements.txt
