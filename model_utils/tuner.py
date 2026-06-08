import copy
import torch
from torch.optim import AdamW

from strategy.ewc   import EWCStrategy
from strategy.derpp import DERppStrategy

# grid of hyperparameters to search at drift detection.
# Replay is excluded because buffer_size can't be changed retroactively
# once the buffer is already filled with samples from the first task.
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
    """
    Finds optimal strategy hyperparameters at drift detection using only
    data seen so far — no look-ahead, fully compatible with online CL.

    Follows the algorithm from De Lange et al.: first find the plasticity
    ceiling (max accuracy with no regularization), then pick the config
    that minimizes forgetting while staying within p% of that ceiling.
    """

    def __init__(self, strategies, search_spaces=None, p=0.05, mini_epochs=2):
        self.strategies    = strategies
        self.search_spaces = search_spaces or SEARCH_SPACES
        self.p             = p
        self.mini_epochs   = mini_epochs

    def _split_batches(self, recent_batches):
        # the recent buffer contains a mix of old and new task data
        # we split by source to evaluate forgetting and plasticity separately
        if not recent_batches:
            return [], []
        sources = [b["source"].iloc[0] for b in recent_batches]
        if len(set(sources)) == 1:
            return recent_batches, []
        split = next(i for i in range(1, len(sources)) if sources[i] != sources[i-1])
        return recent_batches[:split], recent_batches[split:]

    def _accuracy(self, model, trainer, batches):
        if not batches:
            return 0.0
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in batches:
                enc, labels = trainer._encode(batch, trainer._last_label_map)
                preds = model(**enc).logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        model.train()
        return correct / total if total > 0 else 0.0

    def _mini_train(self, model, trainer, batches, strategies=None):
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

    def tune(self, trainer, recent_batches):
        if not recent_batches or not self.strategies:
            return

        old_batches, new_batches = self._split_batches(recent_batches)
        if not new_batches:
            print("[Tuner] Not enough new task data — skipping HP search")
            return

        print(f"\n[Tuner] HP search on {len(old_batches)} old + {len(new_batches)} new batches")

        # step 1: plasticity ceiling -> how well can the model learn the new task
        # with no regularization at all?
        base_model = copy.deepcopy(trainer.model)
        self._mini_train(base_model, trainer, new_batches)
        A = self._accuracy(base_model, trainer, new_batches)
        print(f"[Tuner] Plasticity ceiling A = {A:.4f}")
        del base_model

        # step 2: for each strategy, find the config that minimizes forgetting
        # while keeping plasticity within p% of the ceiling
        for strategy in self.strategies:
            space = self.search_spaces.get(type(strategy))
            if not space:
                continue

            best_config, best_forgetting = None, float("inf")
            print(f"[Tuner] Searching {type(strategy).__name__} over {len(space)} configs...")

            for config in space:
                trial_model    = copy.deepcopy(trainer.model)
                trial_strategy = copy.deepcopy(strategy)
                trial_strategy.__dict__.update(config)
                trial_strategy.active = True

                self._mini_train(trial_model, trainer, recent_batches,
                                 strategies=[trial_strategy])

                A_star     = self._accuracy(trial_model, trainer, new_batches)
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
