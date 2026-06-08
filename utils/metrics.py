# utils/metrics.py
import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_accuracy(preds, labels):
    preds  = preds.cpu()  if isinstance(preds,  torch.Tensor) else torch.tensor(preds)
    labels = labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    return (preds == labels).float().mean().item()


def compute_f1(preds, labels, average="macro"):
    preds  = preds.cpu().numpy()  if isinstance(preds,  torch.Tensor) else np.array(preds)
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    return f1_score(labels, preds, average=average, zero_division=0)


def compute_bwt(eval_log, task="davidson"):
    # negative means forgetting, positive means the old task actually improved
    values = [e[task] for e in eval_log if task in e]
    if not values:
        return float("nan")
    return values[-1] - max(values)


def compute_fwt(eval_log, new_task="hatexplain", boundary_batch=None):
    # zero-shot transfer: how well does the model do on the new task
    # before seeing any of its training data?
    if boundary_batch is None:
        return float("nan")
    candidates = [e for e in eval_log if e["batch"] <= boundary_batch and new_task in e]
    return candidates[-1][new_task] if candidates else float("nan")


def compute_aaa(eval_log, tasks):
    # average anytime accuracy captures the full learning curve, not just endpoints
    scores = [np.mean([e[t] for t in tasks if t in e])
              for e in eval_log if any(t in e for t in tasks)]
    return float(np.mean(scores)) if scores else float("nan")


def compute_all(eval_log, old_task="davidson", new_task="hatexplain", boundary_batch=None):
    bwt = compute_bwt(eval_log, old_task)
    fwt = compute_fwt(eval_log, new_task, boundary_batch)
    aaa = compute_aaa(eval_log, [old_task, new_task])

    print(f"BWT  ({old_task}): {bwt:+.4f}  {'(forgetting)' if bwt < 0 else '(positive transfer)'}")
    print(f"FWT  ({new_task}):  {fwt:.4f}  (zero-shot before training)")
    print(f"AAA  (all tasks):  {aaa:.4f}  (avg anytime accuracy)")
    return {"bwt": bwt, "fwt": fwt, "aaa": aaa}
