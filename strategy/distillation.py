# strategies/distillation.py
import torch
import torch.nn.functional as F

class DistillationStrategy:
    def __init__(self, old_model, temperature=2.0, alpha=0.5):
        self.old_model = old_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, new_outputs, labels, criterion):
        # Cross-entropy on new task
        ce_loss = criterion(new_outputs, labels)
        if self.old_model is None:
            return ce_loss
        # Soft targets from old model
        with torch.no_grad():
            old_outputs = self.old_model(labels)
        soft_loss = F.kl_div(
            F.log_softmax(new_outputs / self.temperature, dim=-1),
            F.softmax(old_outputs / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        return self.alpha * ce_loss + (1 - self.alpha) * soft_loss
