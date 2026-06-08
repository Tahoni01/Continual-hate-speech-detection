# dataset/stream_generator.py
import pandas as pd


def create_stream(df, batch_size=32):
    return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]


def create_continual_stream(*dfs, batch_size=32):
    # datasets are concatenated in order with no shuffling between them —
    # this is intentional: we want an abrupt distribution shift, not a gradual one
    batches = []
    for df in dfs:
        batches += [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    return batches


def stream_info(stream, name="stream"):
    total   = sum(len(b) for b in stream)
    sources = pd.concat(stream)["source"].value_counts().to_dict()
    print(f"{name}: {len(stream)} batches | {total} samples | {sources}")
