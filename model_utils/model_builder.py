import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class HateSpeechClassifier(nn.Module):

    def __init__(
        self,
        model_name="distilbert/distilroberta-base",
        num_labels=3,
        unfreeze_last_n=3,
        dropout=0.1,
        class_weights=None,
    ):
        super().__init__()

        self.config   = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # freeze the whole backbone first, then selectively unfreeze the top layers —
        # lower layers capture general language patterns we want to keep intact
        for p in self.backbone.parameters():
            p.requires_grad = False

        layers = list(self.backbone.encoder.layer)
        for layer in layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        hidden = self.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels),
        )

        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def parameter_groups(self, lr_backbone=2e-5, lr_head=1e-4):
        # the head trains from scratch so it needs a higher LR than the backbone
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": lr_backbone},
            {"params": self.head.parameters(), "lr": lr_head},
        ]

    def forward(self, input_ids, attention_mask=None, labels=None):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(out.last_hidden_state[:, 0])  # CLS token
        loss   = self.criterion(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
