import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()

        # backbone
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # semplice classification head
        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        # init pesi head
        nn.init.kaiming_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(cls_token)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
