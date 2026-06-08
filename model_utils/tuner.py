import copy
import torch
from torch.optim import AdamW

from strategy.ewc   import EWCStrategy
from strategy.derpp import DERppStrategy

SEARCH_SPACES = {
    EWCStrategy: [
        {"lambda_": 0.01},
        {"lambda_": 0.1},
        {"lambda_": 1.0},
    ],
    DERppStrategy: [
        {"alpha": 0.1, "beta": 1.0},
        {"alpha": 0.3, "beta": 1.0},
        {"alpha": 0.5, "beta": 1.0},
    ],
}


class ContinualTuner:
    def __init__(self, strategies, search_spaces=None, p=0.05, mini_epochs=2):
        self.strategies    = strategies
        self.search_spaces = search_spaces or SEARCH_SPACES
        self.p             = p
        self.mini_epochs   = mini_epochs

    def _accuracy(self, model, trainer, batches, encoded=False):
        # Evaluate accuracy on DataFrame batches or pre-encoded (enc, labels) tuples.
        if not batches:
            return 0.0
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in batches:
                if encoded:
                    enc, labels = batch
                    enc    = {k: v.to(trainer.device) for k, v in enc.items()}
                    labels = labels.to(trainer.device)
                else:
                    enc, labels = trainer._encode(batch, trainer._last_label_map)
                preds = model(**enc).logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        model.train()
        return correct / total if total > 0 else 0.0

    def _mini_train(self, model, trainer, batches, strategies=None):
        # fresh optimizer, avoids momentum bleed from the main optimizer
        optimizer = AdamW(model.parameters(), lr=trainer.lr)
        model.train()
        for _ in range(self.mini_epochs):
            for batch in batches:
                enc, labels = trainer._encode(batch, trainer._last_label_map)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out  = model(**enc, labels=labels)
                    loss = out.loss
                    for s in (strategies or []):
                        loss = loss + s.compute_loss(trainer, enc, labels, out.logits)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

    def tune(self, trainer, old_batches, new_batches):
        if not self.strategies:
            return
        if not new_batches:
            print("[Tuner] Not enough new task data — skipping HP search")
            return

        recent_batches = old_batches + new_batches
        print(f"\n[Tuner] HP search on {len(old_batches)} old + {len(new_batches)} new batches")

        # step 1: plasticity ceiling on new task with no regularization
        base_model = copy.deepcopy(trainer.model)
        self._mini_train(base_model, trainer, new_batches)
        A = self._accuracy(base_model, trainer, new_batches)
        print(f"[Tuner] Plasticity ceiling A = {A:.4f}")
        del base_model

        for strategy in self.strategies:
            space = self.search_spaces.get(type(strategy))
            if not space:
                continue

            # prefer strategy buffer for forgetting eval — spans the full old task
            old_encoded = strategy.old_task_batches(trainer)
            if old_encoded is not None:
                n = sum(len(b[1]) for b in old_encoded)
                print(f"[Tuner] Forgetting eval on strategy buffer ({n} samples)")

            best_config, best_forgetting = None, float("inf")
            print(f"[Tuner] Searching {type(strategy).__name__} over {len(space)} configs...")

            for config in space:
                trial_model    = copy.deepcopy(trainer.model)
                trial_strategy = copy.deepcopy(strategy)
                trial_strategy.__dict__.update(config)
                trial_strategy.active = True

                self._mini_train(trial_model, trainer, recent_batches,
                                 strategies=[trial_strategy])

                A_star = self._accuracy(trial_model, trainer, new_batches)

                if old_encoded is not None:
                    forgetting = 1.0 - self._accuracy(trial_model, trainer,
                                                       old_encoded, encoded=True)
                else:
                    forgetting = 1.0 - self._accuracy(trial_model, trainer, old_batches)

                print(f"  {config} → A*={A_star:.4f} forgetting={forgetting:.4f}", end="")

                if A_star >= A * (1 - self.p):
                    if forgetting < best_forgetting:
                        best_forgetting = forgetting
                        best_config     = config
                    print(" ✓")
                else:
                    print(" ✗ (plasticity too low)")

                del trial_model

            if best_config:
                strategy.__dict__.update(best_config)
                print(f"[Tuner] Best: {best_config} (forgetting={best_forgetting:.4f})")
            else:
                print(f"[Tuner] No valid config found — keeping defaults")