import pandas as pd


def create_stream(df, batch_size=32, shuffle=True):
    df = df.copy()
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]


def create_continual_stream(df_list, batch_size=32, shuffle_within_dataset=True):
    """
    Stream sequenziale per continual learning.
    Passa solo i train split — i val split restano separati per evaluation.
    """
    full_stream = []
    for df in df_list:
        df = df.copy()
        if shuffle_within_dataset:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        for i in range(0, len(df), batch_size):
            full_stream.append(df.iloc[i:i + batch_size])
    return full_stream


def online_stream(stream):
    for batch in stream:
        yield batch


def stream_summary(stream, name="stream"):
    total_samples = sum(len(b) for b in stream)
    print(f"\n{name} — {len(stream)} batch | {total_samples} samples | avg {round(total_samples/len(stream), 1)}/batch")


def dataset_distribution(stream):
    df = pd.concat(stream)
    print("\nLABEL DISTRIBUTION")
    print(df["label"].value_counts())