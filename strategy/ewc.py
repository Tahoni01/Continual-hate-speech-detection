import torch
from strategy.base import BaseStrategy


class EWCStrategy(BaseStrategy):
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_
        self.fisher  = {}
        self.params  = {}
        self.active  = False

    def on_task_end(self, trainer, old_batches, new_batches):
        model  = trainer.model
        device = trainer.device
        model.eval()

        # initialize Fisher accumulators for all trainable params
        fisher = {n: torch.zeros_like(p)
                  for n, p in model.named_parameters() if p.requires_grad}

        # use only old task batches for Fisher — avoids contamination from new task
        n_batches = 0
        for batch in old_batches:
            inputs, _ = trainer._encode(batch, trainer._last_label_map)
            model.zero_grad()

            # sample labels from the model's own distribution (Practical Fisher)
            with torch.no_grad():
                probs     = torch.softmax(model(**inputs).logits, dim=-1)
                sampled_y = torch.multinomial(probs, num_samples=1).squeeze(1)

            model(**inputs, labels=sampled_y).loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

            n_batches += 1

        if n_batches == 0:
            print("[EWC] Warning: empty stream, skipping Fisher update.")
            return

        for n in fisher:
            fisher[n] /= n_batches

        self.fisher = fisher
        self.params = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.active = True
        model.train()
        print(f"[EWC] Fisher computed on {n_batches} batches (lambda={self.lambda_})")

    def compute_loss(self, trainer, inputs, labels, logits):
        if not self.active:
            return 0.0

        penalty = sum(
            (self.fisher[n] * (p - self.params[n]) ** 2).sum()
            for n, p in trainer.model.named_parameters()
            if p.requires_grad and n in self.fisher
        )
        return self.lambda_ * penalty