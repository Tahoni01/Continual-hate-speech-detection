# dataset/df_loader.py
import json
import re

import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# single source of truth for label encoding — imported by all other modules
LABEL_MAP = {"hatespeech": 0, "offensive": 1, "normal": 2}


def _clean(text):
    # keep the text as close to natural language as possible —
    # RoBERTa handles punctuation and casing well, so we only
    # strip noise that's meaningless to it (URLs, mentions, hashtag symbols)
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _process(df, val_size=0.2):
    # stratified split so class ratios are preserved in both sets
    df = df.copy()
    df["text"] = df["text"].apply(_clean)
    df = df[df["text"].apply(lambda t: isinstance(t, str) and len(t.split()) >= 3)]
    df = df.reset_index(drop=True)
    train, val = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=42, shuffle=True
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def load_davidson(val_size=0.2):
    ds = load_dataset("tdavidson/hate_speech_offensive", keep_in_memory=True)
    df = ds["train"].to_pandas()

    df = df.rename(columns={"tweet": "text"})
    df["label"]  = df["class"].map({0: "hatespeech", 1: "offensive", 2: "normal"})
    df["source"] = "davidson"
    df = df[["text", "label", "source"]]

    train, val = _process(df, val_size)
    print(f"Davidson   — train: {len(train):>5} | val: {len(val):>4}")
    return train, val


def load_hatexplain(val_size=0.2):
    url  = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
    data = json.loads(requests.get(url, timeout=30).text)

    rows = []
    for _, content in data.items():
        text  = " ".join(content["post_tokens"])
        votes = [a["label"] for a in content["annotators"]]
        # majority vote among 3 annotators
        label = max(set(votes), key=votes.count)
        rows.append({"text": text, "label": label, "source": "hatexplain"})

    train, val = _process(pd.DataFrame(rows), val_size)
    print(f"HateXplain — train: {len(train):>5} | val: {len(val):>4}")
    return train, val


def get_class_weights(*train_dfs):
    # useful when training without CL strategies to counteract Davidson's imbalance
    labels  = pd.concat(train_dfs)["label"].map(LABEL_MAP).values
    classes = np.array(sorted(LABEL_MAP.values()))
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    for name, idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        print(f"  {name:<12}: {weights[idx]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)
