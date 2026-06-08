import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_label_distribution(*dfs, names=None):
    n      = len(dfs)
    names  = names or [f"Dataset {i}" for i in range(n)]
    colors = ["#e74c3c", "#e67e22", "#2ecc71"]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, df, name in zip(axes, dfs, names):
        counts = df["label"].value_counts()
        bars   = ax.bar(counts.index, counts.values, color=colors[:len(counts)])
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.01,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{name}  (n={counts.sum()})")
        ax.set_ylim(0, max(counts.values) * 1.15)

    plt.suptitle("Label Distribution", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_text_length(*dfs, names=None, bins=40):
    n     = len(dfs)
    names = names or [f"Dataset {i}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, df, name in zip(axes, dfs, names):
        lengths = df["text"].apply(lambda x: len(x.split()))
        ax.hist(lengths, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(lengths.mean(), color="red", linestyle="--",
                   label=f"mean={lengths.mean():.0f}")
        ax.axvline(lengths.quantile(0.95), color="orange", linestyle="--",
                   label=f"p95={lengths.quantile(0.95):.0f}")
        ax.set_title(name)
        ax.set_xlabel("Words")
        ax.legend(fontsize=8)

    plt.suptitle("Text Length Distribution", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_stream_shift(stream):
    sources = [b["source"].iloc[0] for b in stream]

    fig, ax = plt.subplots(figsize=(12, 3))

    class_colors = {"hatespeech": "#e74c3c", "offensive": "#e67e22", "normal": "#2ecc71"}
    for cls, col in class_colors.items():
        ratios = [b["label"].value_counts(normalize=True).get(cls, 0) for b in stream]
        ax.plot(ratios, color=col, linewidth=1.5, label=cls)

    shift = next((i for i in range(1, len(sources)) if sources[i] != sources[i-1]), None)
    if shift:
        ax.axvline(shift - 0.5, color="black", linestyle="--", linewidth=1.5, label="shift")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Class ratio")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Stream Distribution Shift", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ─── TRAINING ────────────────────────────────────────────────────────────────

def plot_loss(losses, boundary=None, smooth=5):
    losses = np.array(losses)

    plt.figure(figsize=(10, 4))
    plt.plot(losses, alpha=0.3, color="steelblue", label="raw")

    if len(losses) >= smooth:
        smoothed = np.convolve(losses, np.ones(smooth) / smooth, mode="valid")
        plt.plot(range(smooth - 1, len(losses)), smoothed,
                 color="steelblue", linewidth=2, label=f"smoothed (w={smooth})")

    if boundary is not None:
        plt.axvline(boundary, color="red", linestyle="--", linewidth=1.5, label="task shift")

    plt.title("Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_eval_log(eval_log, boundary=None):
    # Accuracy of each dataset over training batches. Shows forgetting dynamics.
    import pandas as pd
    df      = pd.DataFrame(eval_log).set_index("batch")
    colors  = {"davidson": "#3498db", "hatexplain": "#e74c3c"}

    plt.figure(figsize=(10, 4))
    for col in df.columns:
        plt.plot(df.index, df[col], marker="o", markersize=4,
                 color=colors.get(col, "gray"), label=col)

    if boundary is not None:
        plt.axvline(boundary, color="black", linestyle="--", linewidth=1.5, label="task shift")

    plt.title("Accuracy over training — both val sets")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_conf_matrix(preds, labels, label_map, title=""):
    import torch
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    if isinstance(preds, torch.Tensor):
        preds  = preds.cpu().numpy()
        labels = labels.cpu().numpy()

    cm    = confusion_matrix(labels, preds)
    norm  = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    names = list(label_map.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, t in zip(axes, [cm, norm], ["d", ".2f"], ["counts", "normalized"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(f"{title} ({t})" if title else t)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.show()


# ─── FINAL COMPARISON ────────────────────────────────────────────────────────

def plot_comparison(cl_results, label_map):
    # Side-by-side comparison of all experiments.
    # Shows BWT, FWT, AAA, DV F1, HX F1 for each strategy.
    
    import pandas as pd
    from utils.metrics import compute_f1

    names = list(cl_results.keys())
    rows  = []
    for name, res in cl_results.items():
        m  = res["metrics"]
        dv = compute_f1(res["preds_dv"], res["labels_dv"])
        hx = compute_f1(res["preds_hx"], res["labels_hx"])
        rows.append({
            "Experiment": name,
            "BWT":   round(m["bwt"], 4),
            "FWT":   round(m["fwt"], 4),
            "AAA":   round(m["aaa"], 4),
            "DV F1": round(dv, 4),
            "HX F1": round(hx, 4),
        })

    df = pd.DataFrame(rows).set_index("Experiment")

    # ── bar chart ─────────────────────────────────────────────────────────────
    metrics  = ["BWT", "FWT", "AAA", "DV F1", "HX F1"]
    colors   = plt.cm.Set2(np.linspace(0, 1, len(names)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, metric in zip(axes, metrics):
        vals  = df[metric].values
        bars  = ax.bar(names, vals, color=colors)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(min(0, min(vals)) - 0.05, max(vals) + 0.05)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:+.3f}" if metric == "BWT" else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)

    plt.suptitle("Continual Learning — Strategy Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()

    return df


def plot_eval_logs_comparison(cl_results):
    # Overlay accuracy curves of all experiments on a single plot.
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    colors    = plt.cm.Set1(np.linspace(0, 1, len(cl_results)))
    tasks     = ["davidson", "hatexplain"]

    for ax, task in zip(axes, tasks):
        for (name, res), color in zip(cl_results.items(), colors):
            log = pd.DataFrame(res["eval_log"])
            if task not in log.columns:
                continue
            ax.plot(log["batch"], log[task], label=name, color=color, linewidth=1.5)
            if res.get("drift_at"):
                ax.axvline(res["drift_at"], color=color, linestyle="--",
                           alpha=0.4, linewidth=1)
        ax.set_title(task.capitalize(), fontweight="bold")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Accuracy over Training — All Strategies", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_stability_plasticity(cl_results):
    from utils.metrics import compute_f1

    colors = plt.cm.Set1(np.linspace(0, 1, len(cl_results)))
    fig, ax = plt.subplots(figsize=(7, 5))

    for (name, res), color in zip(cl_results.items(), colors):
        dv = compute_f1(res["preds_dv"], res["labels_dv"])
        hx = compute_f1(res["preds_hx"], res["labels_hx"])
        ax.scatter(hx, dv, s=180, color=color, zorder=3, label=name)
        ax.annotate(name, (hx, dv), textcoords="offset points",
                    xytext=(8, 4), fontsize=10, color=color)

    ax.set_xlabel("Plasticity — HX F1", fontsize=12)
    ax.set_ylabel("Stability — DV F1", fontsize=12)
    ax.set_title("Stability / Plasticity Tradeoff", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_forgetting_curve(cl_results):
    import pandas as pd
    colors = plt.cm.Set1(np.linspace(0, 1, len(cl_results)))

    fig, ax = plt.subplots(figsize=(10, 4))

    for (name, res), color in zip(cl_results.items(), colors):
        log        = pd.DataFrame(res["eval_log"])
        drift_at   = res["drift_at"]
        if "davidson" not in log.columns:
            continue
        post = log[log["batch"] >= (drift_at or 0)]
        ax.plot(post["batch"], post["davidson"],
                color=color, linewidth=1.8, label=name)

    if any(res["drift_at"] for res in cl_results.values()):
        drift = next(r["drift_at"] for r in cl_results.values() if r["drift_at"])
        ax.axvline(drift, color="black", linestyle="--", linewidth=1.2, label="drift")

    ax.set_xlabel("Batch")
    ax.set_ylabel("Davidson Accuracy")
    ax.set_title("Forgetting Curve — Davidson accuracy post-drift", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_posttrain_davidson(cl_results, tokenizer, label_map, val_stream_dv, device, model_dir="models"):
    import os
    from model_utils.model_builder import HateSpeechClassifier

    names, accs, f1s = [], [], []

    for name in cl_results:
        safe = name.replace("+", "_").replace(" ", "_")
        path = os.path.join(model_dir, f"{safe}.pt")
        if not os.path.exists(path):
            print(f"[SKIP] {name} — {path} not found")
            continue

        # carica il modello
        model = HateSpeechClassifier(num_labels=3)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model = model.to(device).eval()

        # valuta su Davidson val
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_stream_dv:
                enc = tokenizer(
                    batch["text"].tolist(),
                    return_tensors="pt", truncation=True,
                    max_length=128, padding=True,
                ).to(device)
                preds  = model(**enc).logits.argmax(dim=1).cpu().tolist()
                labels = [label_map[l] for l in batch["label"].tolist()]
                all_preds.extend(preds)
                all_labels.extend(labels)

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        names.append(name)
        accs.append(acc)
        f1s.append(f1)
        print(f"{name:<15} Acc={acc:.4f} | F1={f1:.4f}")

    if not names:
        print("No models found.")
        return

    x      = np.arange(len(names))
    width  = 0.35
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color=colors, alpha=0.7)
    bars2 = ax.bar(x + width/2, f1s,  width, label="F1 macro", color=colors, alpha=1.0)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Davidson — post-training evaluation on val set", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(cl_results, label_map):
    from sklearn.metrics import f1_score
    import torch

    classes = list(label_map.keys())
    names   = list(cl_results.keys())
    colors  = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, task, pred_key, label_key in zip(
        axes,
        ["Davidson POST", "HateXplain POST"],
        ["preds_dv", "preds_hx"],
        ["labels_dv", "labels_hx"]
    ):
        x     = np.arange(len(classes))
        width = 0.8 / len(names)

        for j, (name, res) in enumerate(cl_results.items()):
            preds  = res[pred_key].numpy()
            labels = res[label_key].numpy()
            f1s    = f1_score(labels, preds, average=None, zero_division=0)
            ax.bar(x + j * width, f1s, width, label=name, color=colors[j])

        ax.set_title(task, fontweight="bold")
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels(classes)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Per-class F1 — All Strategies", fontweight="bold")
    plt.tight_layout()
    plt.show()