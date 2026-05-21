import random
import pandas as pd


def create_stream(df, batch_size=32, shuffle=True, drop_last=False):
    df = df.copy()
    if shuffle:
      
        df = df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)
    stream = []
  
    # creazione batch
    for start_idx in range(0, len(df), batch_size):

        batch = df.iloc[
            start_idx:start_idx + batch_size
        ]

        # skip ultimo batch piccolo
        if drop_last and len(batch) < batch_size:
            continue
        stream.append(batch)
    return stream


def merge_streams(streams, shuffle_streams=False):
    merged_stream = []

    for stream in streams:
        merged_stream.extend(stream)

    if shuffle_streams:
        random.shuffle(merged_stream)

    return merged_stream


def online_stream(stream):
    for batch in stream:

        yield batch


def stream_summary(stream, name="stream"):
    total_batches = len(stream)

    total_samples = sum(
        len(batch)
        for batch in stream
    )

    print(f"\nSTREAM SUMMARY: {name}")
    print("-" * 30)

    print(f"total batches: {total_batches}")
    print(f"total samples: {total_samples}")

    if total_batches > 0:

        print(
            f"avg batch size: "
            f"{round(total_samples / total_batches, 2)}"
        )


def dataset_distribution(stream):
    full_df = pd.concat(stream)

    print("\nLABEL DISTRIBUTION")
    print("-" * 30)

    print(
        full_df["label"]
        .value_counts(normalize=True)
    )
