# utils/metrics.py
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=-1) if preds.ndim > 1 else preds
    return accuracy_score(labels, preds)

def compute_f1(preds, labels, average="macro"):
    preds = torch.argmax(preds, dim=-1) if preds.ndim > 1 else preds
    return f1_score(labels, preds, average=average)

def compute_confusion_matrix(preds, labels):
    preds = torch.argmax(preds, dim=-1) if preds.ndim > 1 else preds
    return confusion_matrix(labels, preds)

def compute_forgetting(old_acc, current_acc):
    """
    old_acc: list of accuracies on previous tasks when first learned
    current_acc: list of accuracies on previous tasks after training new task
    returns: list of forgetting per previous task
    """
    return [o - c for o, c in zip(old_acc, current_acc)]
