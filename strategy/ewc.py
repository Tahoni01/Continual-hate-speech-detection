import torch

class EWCStrategy:
    def __init__(self, model, lambda_=0.4):
        self.model = model
        self.lambda_ = lambda_
        self.params = {}
        self.fisher = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.detach().clone()
                self.fisher[n] = torch.zeros_like(p)

    def compute_fisher(self, dataloader, loss_fn, device):
        self.model.eval()

        for inputs, labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            self.model.zero_grad()

            outputs = self.model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2

        for n in self.fisher:
            self.fisher[n] /= len(dataloader)

    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()

        return self.lambda_ * loss