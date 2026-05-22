# plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.metrics import compute_accuracy, compute_f1

def plot_loss(losses, smooth_window=3):
    smoothed_losses = np.convolve(losses, np.ones(smooth_window)/smooth_window, mode='valid')
    plt.figure(figsize=(8,4))
    plt.plot(smoothed_losses, marker='o')
    plt.title("Smoothed Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def plot_batch_metrics(preds, labels):
    batch_accuracies = [compute_accuracy(p, l) for p, l in zip(preds, labels)]
    batch_f1 = [compute_f1(p, l) for p, l in zip(preds, labels)]
    
    plt.figure(figsize=(8,4))
    plt.plot(batch_accuracies, marker='o', label="Batch Accuracy")
    plt.plot(batch_f1, marker='x', label="Batch F1")
    plt.xlabel("Batch")
    plt.ylabel("Score")
    plt.title("Metriche per batch")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_conf_matrix(preds, labels, label_map):
    flat_preds = torch.cat(preds).cpu()
    flat_labels = torch.cat(labels).cpu()
    conf_matrix = confusion_matrix(flat_labels, flat_preds)

    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix globale")
    plt.show()

def plot_classwise_accuracy(preds, labels, label_map):
    num_batches = len(preds)
    num_classes = len(label_map)
    accuracy_per_class = torch.zeros(num_batches, num_classes)

    for i, (p, l) in enumerate(zip(preds, labels)):
        for cls in range(num_classes):
            mask = l == cls
            if mask.sum() > 0:
                accuracy_per_class[i, cls] = (p[mask] == cls).float().mean()

    for cls_idx, cls_name in enumerate(label_map.keys()):
        plt.plot(range(num_batches), accuracy_per_class[:, cls_idx], marker='o', label=cls_name)

    plt.xlabel("Batch")
    plt.ylabel("Accuracy per class")
    plt.title("Accuratezza per classe durante l'apprendimento continuo")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_prediction_distribution(preds, label_map):
    for i, p in enumerate(preds):
        counts = pd.Series(p.cpu().numpy()).value_counts().sort_index()
        counts = counts.reindex(range(len(label_map)), fill_value=0)
        counts.plot(kind='bar')
        plt.title(f"Distribuzione predizioni batch {i+1}")
        plt.xlabel("Classe")
        plt.ylabel("Numero predizioni")
        plt.show()
