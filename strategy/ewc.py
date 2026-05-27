# strategy/ewc.py
import torch
from strategy.base import BaseStrategy

class EWCStrategy(BaseStrategy):
    def __init__(self, lambda_=0.4):
        self.lambda_ = lambda_
        self.params  = {}   # pesi di riferimento dopo ogni task
        self.fisher  = {}   # importanza stimata dei pesi

    def on_task_end(self, trainer, stream, label_map):
        """
        Chiamato dopo ogni task:
        1. Ricalcola la Fisher information sul task appena visto
        2. Aggiorna i params di riferimento con i pesi correnti
        """
        model  = trainer.model
        device = trainer.device
        model.eval()

        # reset fisher
        new_fisher = {n: torch.zeros_like(p)
                      for n, p in model.named_parameters() if p.requires_grad}

        n_batches = 0
        for batch in stream:
            inputs, labels = trainer.prepare_batch(batch, label_map)
            model.zero_grad()
            outputs = model(**inputs)
            loss = trainer.ce_loss(outputs.logits, labels)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_fisher[n] += p.grad.detach() ** 2
            n_batches += 1

        # media + accumulo (supporta task multipli)
        for n in new_fisher:
            new_fisher[n] /= max(n_batches, 1)
            if n in self.fisher:
                self.fisher[n] = (self.fisher[n] + new_fisher[n]) / 2
            else:
                self.fisher[n] = new_fisher[n]

        # snapshot dei pesi correnti come nuovo riferimento
        self.params = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

        model.train()
        print(f"[EWC] Fisher aggiornata su {n_batches} batch.")

    def compute_loss(self, trainer, inputs, labels, logits):
        if not self.params:
            return 0.0  # nessun task precedente ancora

        loss = 0.0
        for n, p in trainer.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()

        return self.lambda_ * loss