# plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.metrics import compute_accuracy, compute_f1

# ─── helpers ────────────────────────────────────────────────────────────────

def _to_numpy(t):
    """Converte tensore o array in numpy array 1D."""
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.array(t)

def _split_batches(flat_tensor, batch_size):
    """Splitta tensore flat in lista di chunk di dimensione batch_size."""
    return list(flat_tensor.split(batch_size))

# ─── plot esistenti (fixati) ─────────────────────────────────────────────────

def plot_loss(losses, smooth_window=5):
    """Loss per batch con smoothing."""
    losses = np.array(losses)
    smoothed = np.convolve(losses, np.ones(smooth_window) / smooth_window, mode='valid')

    plt.figure(figsize=(9, 4))
    plt.plot(losses, alpha=0.3, color="steelblue", label="Raw loss")
    plt.plot(range(smooth_window - 1, len(losses)), smoothed,
             color="steelblue", linewidth=2, label=f"Smoothed (w={smooth_window})")
    plt.title("Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_batch_metrics(flat_preds, flat_labels, batch_size=10):
    """Accuracy e F1 per batch — accetta tensori flat."""
    preds_batches  = _split_batches(flat_preds, batch_size)
    labels_batches = _split_batches(flat_labels, batch_size)

    accs = [compute_accuracy(p, l) for p, l in zip(preds_batches, labels_batches)]
    f1s  = [compute_f1(p, l)       for p, l in zip(preds_batches, labels_batches)]

    plt.figure(figsize=(9, 4))
    plt.plot(accs, marker='o', label="Accuracy")
    plt.plot(f1s,  marker='x', label="F1")
    plt.xlabel("Batch")
    plt.ylabel("Score")
    plt.title("Accuracy e F1 per batch")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_conf_matrix(flat_preds, flat_labels, label_map):
    """Confusion matrix — accetta tensori flat."""
    preds  = _to_numpy(flat_preds)
    labels = _to_numpy(flat_labels)

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalizzata per riga

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    class_names = list(label_map.keys())

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (conteggi)", "Confusion Matrix (normalizzata)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_classwise_accuracy(flat_preds, flat_labels, label_map, batch_size=10):
    """Accuracy per classe nel tempo — accetta tensori flat."""
    preds_batches  = _split_batches(flat_preds, batch_size)
    labels_batches = _split_batches(flat_labels, batch_size)

    num_batches = len(preds_batches)
    num_classes = len(label_map)
    acc_per_class = np.zeros((num_batches, num_classes))

    for i, (p, l) in enumerate(zip(preds_batches, labels_batches)):
        for cls in range(num_classes):
            mask = (l == cls)
            if mask.sum() > 0:
                acc_per_class[i, cls] = (p[mask] == cls).float().mean().item()

    plt.figure(figsize=(9, 4))
    for cls_idx, cls_name in enumerate(label_map.keys()):
        plt.plot(acc_per_class[:, cls_idx], marker='o', label=cls_name)

    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per classe durante il training continuo")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─── plot nuovi per continual learning ───────────────────────────────────────

def plot_forgetting(eval_results: dict, label_map: dict):
    plt.figure(figsize=(9, 4))
    for task_name, accs in eval_results.items():
        # filtra None per il plot
        x_vals = [i for i, a in enumerate(accs) if a is not None]
        y_vals = [a for a in accs if a is not None]

        plt.plot(x_vals, y_vals, marker='o', label=task_name)

        # evidenzia il punto di massima accuracy
        best_idx = int(np.argmax(y_vals))
        plt.annotate(f"peak: {y_vals[best_idx]:.2f}",
                     xy=(x_vals[best_idx], y_vals[best_idx]),
                     xytext=(x_vals[best_idx] + 0.1, y_vals[best_idx] - 0.05),
                     fontsize=8, color="gray")

    plt.xlabel("Task visto")
    plt.ylabel("Accuracy")
    plt.title("Forgetting per task — accuracy nel tempo")
    plt.xticks([0, 1], ["Dopo Task 1\n(Davidson)", "Dopo Task 2\n(HateXplain)"])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_strategy_comparison(results: dict, metric="f1"):
    """
    Confronto tra strategie — per l'ablation study.

    results: dict con struttura
        {
          "Baseline":     {"f1": 0.61, "acc": 0.63},
          "Replay":       {"f1": 0.74, "acc": 0.76},
          "EWC":          {"f1": 0.70, "acc": 0.72},
          "Distillation": {"f1": 0.68, "acc": 0.70},
        }
    """
    strategies = list(results.keys())
    scores     = [results[s][metric] for s in strategies]
    colors     = ["#d9534f" if s == "Baseline" else "#5b9bd5" for s in strategies]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(strategies, scores, color=colors, edgecolor="white", linewidth=0.8)
    plt.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.ylabel(metric.upper())
    plt.title(f"Confronto strategie — {metric.upper()} globale")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_label_distribution(df_dv, df_hx, label_map):
    """
    Distribuzione delle label nei due dataset — utile per evidenziare lo shift.
    """
    class_names = list(label_map.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, df, title in zip(axes, [df_dv, df_hx], ["Davidson", "HateXplain"]):
        counts = df["label"].value_counts().reindex(class_names, fill_value=0)
        bars = ax.bar(class_names, counts.values, color="#5b9bd5", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title(f"Distribuzione label — {title}")
        ax.set_ylabel("Campioni")
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Confronto distribuzione label (possibile fonte di drift)", y=1.02)
    plt.tight_layout()
    plt.show()