import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, PeftModel
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

from IPython.display import display
import ast

class CustomClassifier(nn.Module):
    def __init__(self, model_name, config, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        base = AutoModel.from_pretrained(model_name, config=config)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", target_modules=["query", "value"]
        )
        self.base_model = get_peft_model(base, lora_cfg)

        # custom head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.LayerNorm(config.hidden_size//2),
            nn.GELU(),
            nn.Linear(config.hidden_size//2, config.num_labels)
        )
        # init head
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0)

    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

def save_model_pickle(model, save_path, tokenizer):
    model.base_model.save_pretrained(save_path, safe_serialization=True)
    torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier.pt"))
    tokenizer.save_pretrained(save_path)
    model.base_model.config.save_pretrained(save_path)

def load_model_pickle(path, model_name, device="cpu"):
    config = AutoConfig.from_pretrained(path)
    model = CustomClassifier(model_name, config)
    model.base_model.load_adapter(path, adapter_name="default")
    classifier_path = os.path.join(path, "classifier.pt")
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(path)

    return model, tokenizer

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from IPython.display import display

def export_epoch_metrics(logs):
    eval_metrics = logs[logs['eval_loss'].notna()]
    if not eval_metrics.empty:
        columns_to_export = [
            'epoch', 'eval_loss', 'eval_accuracy',
            'eval_precision', 'eval_recall', 'eval_f1'
        ]
        existing_cols = [col for col in columns_to_export if col in eval_metrics.columns]
        eval_metrics_table = eval_metrics[existing_cols].copy()

        rename_map = {
            'epoch': 'Epoch',
            'eval_loss': 'Loss',
            'eval_accuracy': 'Accuracy',
            'eval_precision': 'Precision',
            'eval_recall': 'Recall',
            'eval_f1': 'F1',
        }

        eval_metrics_table.rename(columns=rename_map, inplace=True)
        eval_metrics_table['Epoch'] = eval_metrics_table['Epoch'].astype(int)

        # ✅ Mostra tabella TSV direttamente
        print("Evaluation Metrics Table:")
        display(eval_metrics_table)

        # ✅ Mostra tabella come immagine
        fig, ax = plt.subplots(figsize=(len(eval_metrics_table.columns) * 2, len(eval_metrics_table) * 0.6 + 1))
        ax.axis('off')

        table = ax.table(
            cellText=eval_metrics_table.values,
            colLabels=eval_metrics_table.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.show()

    else:
        print("No evaluation metrics available.")

def plot_confusion(logs, num_labels, model_name):
    eval_metrics = logs[logs['eval_loss'].notna()]
    best_epoch = np.argmax(eval_metrics['eval_f1'])
    best_cm = eval_metrics.iloc[best_epoch]['eval_confusion_matrix']

    if isinstance(best_cm, str):
        best_cm = ast.literal_eval(best_cm)

    # Mappatura dinamica delle label
    if num_labels == 2:
        label_mapping = {
            0: "no hate",
            1: "hate"
        }
    else:
        label_mapping = {
            0: "age",
            1: "ethnicity",
            2: "gender",
            3: "nothing",
            4: "religion"
        }

    class_labels = [label_mapping[i] for i in sorted(label_mapping)]

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(best_cm), display_labels=class_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Best Confusion Matrix - Epoch {int(eval_metrics.iloc[best_epoch]["epoch"])}\nModel: {model_name}')

    # ✅ Mostra direttamente
    plt.show()

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import Dataset

def evaluate_model(model, tokenizer, df_test, device="cuda", output_dir="results"):
    # 1. Preparazione dataset
    test_data = Dataset.from_pandas(df_test[['tweet_soft', 'label']])

    # Tokenizzation
    def tokenize_fn(examples):
        return tokenizer(
            examples['tweet_soft'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    test_data = test_data.map(tokenize_fn, batched=True)
    test_data = test_data.remove_columns('tweet_soft').rename_column('label', 'labels')

    # 2. Model configuration
    model.to(device)
    model.eval()

    # 3. Prediction
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_data,
        batch_size=64,
        collate_fn=data_collator,
        shuffle=False
    )

    all_preds = []
    all_labels = []

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    # 4. Calcolo metriche
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

    # 5. Salvataggio risultati
    os.makedirs(output_dir, exist_ok=True)

    # Salva metriche
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "metrics.tsv"),
        sep='\t',
        index=False,
        columns=['accuracy', 'precision_macro', 'recall_macro','f1_macro']
    )

    return metrics


def test_plot_confusion_matrix(metrics, class_names=None, output_dir="results"):
    plt.figure(figsize=(10, 8))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['confusion_matrix']))]

    ConfusionMatrixDisplay(
        confusion_matrix=metrics['confusion_matrix'],
        display_labels=class_names
    ).plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
