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

        # freeze everything first, then open up the top layers — lower layers
        # encode syntax and basic semantics that transfer well across domains,
        # so there's no point in updating them with hate speech data
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

        # xavier init keeps activations in a reasonable range from the first batch 
        # random weights would otherwise produce garbage logits until the head warms up
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # label_smoothing=0.1 prevents the model from becoming overconfident on
        # the offensive class, which dominates Davidson at 78%
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def parameter_groups(self, lr_backbone=2e-5, lr_head=1e-4):
        # head is initialized from scratch so it needs a higher LR to catch up
        # with the pretrained backbone in the early batches
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
