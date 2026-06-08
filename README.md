# Continual Hate Speech Detection

> Sequential learning on evolving social media streams with automatic drift detection and anti-forgetting strategies.

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.7.0+cu128-orange)
![Transformers](https://img.shields.io/badge/transformers-4.51.3-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Hate speech detection systems trained on static datasets degrade quickly when deployed in real-world environments where language evolves continuously. This project addresses the problem through **Continual Learning (CL)**: training a DistilRoBERTa-based classifier sequentially on two hate speech datasets that differ significantly in class distribution, simulating an abrupt distribution shift without any task boundary information available to the model.

The core contributions are:

- A fully online CL pipeline with **ADWIN-based drift detection** that identifies distribution shifts automatically without requiring explicit task boundaries
- A **Continual Hyperparameter Selection** module that searches for optimal strategy parameters at drift time using only data seen so far.
- Implementation and comparison of three anti-forgetting strategies: **Class-Balanced Reservoir Replay**, **Elastic Weight Consolidation (EWC)**, and **Dark Experience Replay++ (DER++)**
- An interactive **Gradio demo** for qualitative comparison of model predictions across strategies

---

## Problem Setup

The model observes a continuous stream of hate speech samples from two datasets in sequence:

```
Davidson (19k samples, 78% offensive) ──► HateXplain (15k samples, balanced)
```

No shuffling occurs between datasets, the shift is abrupt and the model has no prior knowledge of when or how it will occur. The training objective is to maintain performance on Davidson (stability) while adapting to HateXplain (plasticity).

This setup is motivated by real-world content moderation scenarios where a model trained on historical data must adapt to new content trends without forgetting how to detect previously seen patterns.

---

## Model Architecture

```
Input text
    │
    ▼
DistilRoBERTa backbone
    │  6 transformer layers
    │  layers 0-2: frozen (general language representations)
    │  layers 3-5: trainable (task-specific adaptation)
    │
    ▼
[CLS] token representation  (768-dim)
    │
    ▼
Custom classification head
    Linear(768 → 384) → ReLU → Dropout(0.1) → Linear(384 → 3)
    │
    ▼
Output: hatespeech / offensive / normal
```

The backbone is partially frozen to preserve general language representations while allowing the upper layers to adapt to the hate speech domain. A differential learning rate is applied: `lr_backbone = 2e-5`, `lr_head = 1e-4`. Label smoothing (`ε = 0.1`) is applied to the cross-entropy loss to reduce overconfidence on the dominant class.

---

## Drift Detection

Distribution shift is detected online using **ADWIN** (Adaptive Windowing), which maintains an adaptive window over the error rate stream and triggers when two sub-windows show statistically different means:

$$|\mu\_W - \mu\_{W'}| \geq \varepsilon\_{\text{cut}}$$

Key design choices:
- Error rate is computed per batch (mean of binary correct/incorrect signals)
- Single trigger per stream, ADWIN fires once and then deactivates
- At drift detection: LR is decayed by a factor of 0.5, strategy hyperparameters are tuned, and strategy hooks are activated

---

## Anti-Forgetting Strategies

### Baseline
No anti-forgetting mechanism. Provides the lower bound on stability, all strategy improvements are measured relative to this.

### Replay (Class-Balanced Reservoir)
Maintains one reservoir per class to counteract Davidson's class imbalance. Each class gets equal buffer capacity:

$$slot\_size = \left\lfloor \frac{buffer\_size}{n\_classes} \right\rfloor$$

The buffer fills silently during Davidson and activates only at drift detection, avoiding redundant replay within the same task.

### Replay + EWC
Combines replay with **Elastic Weight Consolidation** (Kirkpatrick et al., 2017). EWC computes the Practical Fisher Information on the recent data buffer at drift detection and penalizes deviations from reference parameters:

$$L\_{EWC} = L\_{CE} + \lambda \sum\_i F\_i \left(\theta_i - \theta^*\_i\right)^2$$

EWC is used in its offline variant here, online Fisher accumulation would be contaminated by the replay gradients mixing the two task distributions.

### DER++
**Dark Experience Replay++** (Buzzega et al., 2020) extends replay by storing the model's output logits at insertion time. At replay, an MSE term penalizes changes to the model's past output distributions:

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \cdot \text{MSE}(\hat{z}, z^{*}) + \beta \cdot \mathcal{L}_{CE}(\hat{z}, y^{*})$$

This provides functional regularization, preserving what the model *used to predict*, not just which parameters it used.

---

## Continual Hyperparameter Selection

At drift detection, a grid search is performed on the recent data buffer following the algorithm from De Lange et al.:

1. **Plasticity ceiling** — fine-tune on new-task samples with no strategy → get accuracy $A$
2. **For each HP config** — mini-train with strategy on recent buffer → measure $A^*$ and forgetting
3. **Accept** if $A^* \geq A \cdot (1 - p)$ with $p = 0.05$ (5% plasticity tolerance)
4. **Select** the accepted config that minimizes forgetting

This is fully online, only data already seen is used. No look-ahead.

| Strategy | Tunable parameter | Search values |
|---|---|---|
| EWC | `lambda_` | 0.01, 0.1, 1.0 |
| DER++ | `alpha` | 0.1, 0.3, 0.5 |

---

## CL Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **BWT** | $R_{T,i} - R_{i,i}$ | Forgetting — negative means performance dropped on old task |
| **FWT** | $R_{i-1,i} - b_i$ | Zero-shot transfer to new task before training on it |
| **AAA** | $\frac{1}{T \cdot \|\mathcal{E}\|} \sum\_{e,i} R\_{e,i}$ | Average accuracy across all tasks at every checkpoint |

---

## Results

Results on the Davidson → HateXplain stream (batch_size=32, ADWIN δ=0.002, lr_decay=0.5):

| Strategy | BWT ↑ | FWT | AAA ↑ | DV F1 ↑ | HX F1 ↑ | Drift batch |
|---|---|---|---|---|---|---|
| Baseline | -0.33 | 0.43 | 0.63 | 0.48 | 0.60 | 671 |
| Replay | -0.13 | 0.47 | 0.67 | 0.59 | 0.66 | 671 |
| Replay+EWC | -0.15 | 0.43 | 0.65 | 0.59 | 0.63 | 671 |
| **DER++** | **-0.12** | 0.44 | **0.68** | **0.61** | **0.65** | 671 |

DER++ achieves the best BWT and AAA, reducing forgetting by ~64% relative to the Baseline while maintaining competitive HateXplain F1.

---

## Project Structure

```
continual-hate-speech-detection/
│
├── dataset/
│   ├── df_loader.py          # Davidson and HateXplain loaders, label map
│   └── stream_generator.py   # Continual stream construction (no inter-dataset shuffle)
│
├── model_utils/
│   ├── model_builder.py      # DistilRoBERTa + classification head
│   ├── trainer.py            # ContinualTrainer with ADWIN integration
│   └── tuner.py              # Continual HP selection via grid search
│
├── models/                   # Saved checkpoints (generated after training)
│
├── strategy/
│   ├── base.py               # BaseStrategy interface
│   ├── replay.py             # Class-balanced reservoir replay
│   ├── ewc.py                # EWC (offline, triggered at drift)
│   └── derpp.py              # DER++ (boundary-free, logits in buffer)
│
├── utils/
│   ├── metrics.py            # BWT, FWT, AAA, accuracy, F1
│   └── plot_utils.py         # Training curves, confusion matrices, comparison plots
│
│
├── main.ipynb                # Full training pipeline and experiment comparison
├── demo.ipynb                # Interactive Gradio demo
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/continual-hate-speech-detection
cd continual-hate-speech-detection
python -m venv .venv
```

Install dependencies (PyTorch must be installed first with the correct CUDA index):

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

> **Note**: the `cu128` build requires an RTX 40/50 series GPU (CUDA 12.8+). For older GPUs, replace `cu128` with the appropriate CUDA version.

---

## Usage

### Training

Open `main.ipynb` and run cells sequentially. Each experiment cell trains one strategy and saves the model to `models/`.

### Demo

After training, open `demo.ipynb` and run all cells. The Gradio interface will launch at `http://localhost:7860`.
---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.7.0+cu128 | Training, tensors |
| `transformers` | 4.51.3 | DistilRoBERTa backbone |
| `datasets` | 3.6.0 | Davidson dataset loading |
| `river` | 0.21.2 | ADWIN drift detector |
| `scikit-learn` | 1.6.1 | Metrics, class weights |
| `pandas` | 2.2.3 | DataFrame operations |
| `matplotlib` | 3.10.3 | Training plots |
| `seaborn` | 0.13.2 | Confusion matrices |
| `gradio` | latest | Interactive demo |
| `plotly` | latest | Demo visualizations |

---

## License

MIT
