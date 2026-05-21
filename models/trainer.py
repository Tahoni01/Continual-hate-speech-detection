import os
import yaml
import pandas as pd

from datetime import datetime

from transformers import (
    Trainer,
    EarlyStoppingCallback
)


def build_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics
):

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        tokenizer=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.01
            )
        ]
    )

    return trainer


def train_model(trainer):

    result = trainer.train()

    train_time = round(
        result.metrics["train_runtime"] / 60,
        2
    )

    print(f"training time: {train_time} min")

    return result


def create_run_dir(base_dir="saved_runs"):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = os.path.join(
        base_dir,
        f"run_{timestamp}"
    )

    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def save_run(
    trainer,
    tokenizer,
    config_dict,
    run_dir
):

    # model
    trainer.model.save_pretrained(run_dir)

    # tokenizer
    tokenizer.save_pretrained(run_dir)

    # logs
    logs = pd.DataFrame(
        trainer.state.log_history
    )

    logs.to_csv(
        os.path.join(run_dir, "logs.csv"),
        index=False
    )

    # config
    with open(
        os.path.join(run_dir, "config.yaml"),
        "w"
    ) as f:

        yaml.dump(config_dict, f)

    print(f"run salvata in: {run_dir}")
