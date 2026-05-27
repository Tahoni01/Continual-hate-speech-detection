import json
import requests
import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid(text):
    return isinstance(text, str) and len(text.split()) >= 3

def _split(df, val_size=0.2, random_state=42):
    """Split stratificato per label — restituisce (train_df, val_df)."""
    train, val = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)

def getdf_hatexplain(val_size=0.2):
    url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
    data = json.loads(requests.get(url, timeout=30).text)

    rows = []
    for post_id, content in data.items():
        text  = " ".join(content["post_tokens"])
        votes = [ann["label"] for ann in content["annotators"]]
        label = max(set(votes), key=votes.count)
        rows.append({"text": text, "label": label, "source": "hatexplain"})

    df = pd.DataFrame(rows)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].apply(is_valid)].reset_index(drop=True)

    train, val = _split(df, val_size)
    print(f"HateXplain — train: {len(train)} | val: {len(val)}")
    return train, val

def getdf_davidson(val_size=0.2):
    davidson   = load_dataset("tdavidson/hate_speech_offensive", keep_in_memory=True)
    df         = davidson["train"].to_pandas()
    label_map  = {0: "hatespeech", 1: "offensive", 2: "normal"}

    df = df.rename(columns={"tweet": "text"})
    df["label"]  = df["class"].map(label_map)
    df["source"] = "davidson"
    df = df[["text", "label", "source"]]
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].apply(is_valid)].reset_index(drop=True)

    train, val = _split(df, val_size)
    print(f"Davidson   — train: {len(train)} | val: {len(val)}")
    return train, val