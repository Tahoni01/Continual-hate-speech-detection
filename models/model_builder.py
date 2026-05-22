# model_builder.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_backbone=True, unfreeze_last_n_layers=2):
        """
        Modello di classificazione testuale con backbone transformers (DistilRoBERTa)
        e semplice linear head per continual learning.
        
        Args:
            model_name (str): nome del modello pretrained
            num_labels (int): numero di classi
            freeze_backbone (bool): se True blocca i pesi del backbone
        """
        super().__init__()

        # ---------------------------
        # Backbone
        # ---------------------------
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Freeze backbone se richiesto
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False     

        # Sblocca ultimi n layer se richiesto
        if unfreeze_last_n_layers > 0:
            layers = list(self.backbone.transformer.layer)  # lista degli strati DistilRoBERTa
            for layer in layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # ---------------------------
        # Head MLP migliorata
        # ---------------------------
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # Inizializzazione pesi
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]

        # Head
        logits = self.classifier(cls_token)

        # Loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
