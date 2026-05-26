import torch
import torch.nn.functional as F

class DistillationStrategy:
    def __init__(self, old_model=None, temperature=2.0, alpha=0.5):
        self.old_model = old_model
        self.temperature = temperature
        self.alpha = alpha

    def loss(self, model, inputs, labels, ce_loss_fn, device):

        # CE loss (new task)
        outputs = model(**inputs)
        ce_loss = ce_loss_fn(outputs.logits, labels)

        if self.old_model is None:
            return ce_loss

        # old model inference
        with torch.no_grad():
            old_outputs = self.old_model(**inputs)

        # KL distillation
        T = self.temperature

        soft_loss = F.kl_div(
            F.log_softmax(outputs.logits / T, dim=-1),
            F.softmax(old_outputs.logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)

        return self.alpha * ce_loss + (1 - self.alpha) * soft_loss