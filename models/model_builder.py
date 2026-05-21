import torch
import torch.nn as nn

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from peft import get_peft_model, LoraConfig


class CustomClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        config,
        class_weights=None,
        use_lora=True
    ):
        super().__init__()

        self.class_weights = class_weights

        # backbone
        backbone = AutoModel.from_pretrained(
            model_name,
            config=config
        )

        # LoRA
        if use_lora:

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["query", "value"]
            )

            self.backbone = get_peft_model(
                backbone,
                lora_config
            )

        else:
            self.backbone = backbone

        # classification head
        hidden_size = config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, config.num_labels)
        )

        self._init_head()

    def _init_head(self):

        for module in self.classifier:

            if isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding)

        loss = None

        if labels is not None:

            loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights
            )

            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
