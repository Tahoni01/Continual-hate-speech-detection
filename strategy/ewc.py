# strategies/ewc.py
import torch

class EWCStrategy:
    def __init__(self, model, lambda_=0.4):
        self.model = model
        self.lambda_ = lambda_
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    def compute_fisher(self, dataloader, criterion):
        # Estimate diagonal of Fisher information matrix
        for inputs, labels in dataloader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.fisher[n] += p.grad.data ** 2
        # normalize
        for n in self.fisher:
            self.fisher[n] /= len(dataloader)

    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ * loss
