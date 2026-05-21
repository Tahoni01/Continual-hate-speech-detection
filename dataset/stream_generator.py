import random
import pandas as pd


# =========================================
# 1. BASE STREAM CREATION
# =========================================

def create_stream(df, batch_size=32, shuffle=True):

    df = df.copy()

    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    stream = []

    for i in range(0, len(df), batch_size):

        batch = df.iloc[i:i + batch_size]

        stream.append(batch)

    return stream


# =========================================
# 2. CONTINUAL LEARNING STREAM (IMPORTANT FIX)
# =========================================

def create_continual_stream(
    df_list,
    batch_size=32,
    shuffle_within_dataset=True
):
    """
    Simula vero continual learning:
    dataset arrivano in ordine temporale
    (NO mixing tra domini)
    """

    full_stream = []

    for df in df_list:

        df = df.copy()

        if shuffle_within_dataset:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        for i in range(0, len(df), batch_size):

            batch = df.iloc[i:i + batch_size]

            full_stream.append(batch)

    return full_stream


# =========================================
# 3. DOMAIN LABEL STREAM (DEBUG + ANALYSIS)
# =========================================

def create_labeled_stream(
    df_list,
    batch_size=32
):
    """
    Ogni batch mantiene info dominio
    utile per analisi forgetting
    """

    stream = []

    for domain_id, df in enumerate(df_list):

        df = df.copy()
        df["domain_id"] = domain_id

        for i in range(0, len(df), batch_size):

            batch = df.iloc[i:i + batch_size]

            stream.append(batch)

    return stream


# =========================================
# 4. STREAM UTILITIES
# =========================================

def online_stream(stream):

    for batch in stream:
        yield batch


def stream_summary(stream, name="stream"):

    total_batches = len(stream)
    total_samples = sum(len(b) for b in stream)

    print(f"\n{name}")
    print("-" * 40)
    print("batches:", total_batches)
    print("samples:", total_samples)
    print("avg batch size:", round(total_samples / total_batches, 2))


def dataset_distribution(stream):

    df = pd.concat(stream)

    print("\nLABEL DISTRIBUTION")
    print(df["label"].value_counts())
