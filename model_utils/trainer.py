import torch
from collections import deque
from torch.optim import AdamW
from tqdm import tqdm
from river.drift import ADWIN


class ContinualTrainer:
    """
    Online continual learning trainer with ADWIN drift detection.

    Each batch error rate is fed to ADWIN. When drift is detected, the tuner
    searches for optimal strategy hyperparameters on the recent buffer, then
    on_task_end activates all strategy hooks.
    """

    def __init__(self, model, tokenizer, device, lr=2e-5, lr_head=1e-4,
                 adwin_delta=0.002, buffer_size=200, lr_decay=0.1):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.lr        = lr
        self.lr_head   = lr_head
        self.lr_decay  = lr_decay
        self.optimizer = self._make_optimizer()
        self.scaler    = torch.amp.GradScaler()

        self.strategies = []
        self.eval_log   = []
        self.tuner      = None

        self.detector       = ADWIN(delta=adwin_delta)
        self.adwin_delta    = adwin_delta
        self.drift_detected = False
        self.drift_batch    = None

        # deque gives O(1) append and automatic eviction a plain list would
        # be O(n) every time the buffer overflows, which adds up over 1000+ batches
        self._recent_batches = deque(maxlen=buffer_size)

    def _make_optimizer(self):
        # if the model exposes parameter_groups, use differential LR
        # otherwise fall back to a single group for simpler architectures
        if hasattr(self.model, "parameter_groups"):
            return AdamW(self.model.parameter_groups(self.lr, self.lr_head))
        return AdamW(self.model.parameters(), lr=self.lr)

    # ── strategy management ──────────────────────────────────────────────────

    def add_strategy(self, strategy):
        self.strategies.append(strategy)
        return self

    def set_tuner(self, tuner):
        self.tuner = tuner
        return self

    def on_task_end(self, stream=None):
        src = list(stream) if stream is not None else list(self._recent_batches)
        for s in self.strategies:
            s.on_task_end(self, src)

    #   checkpoint 

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # reset everything, a loaded model should behave like a freshly trained one

        self.model.load_state_dict(torch.load(path, map_location=self.device,
                                              weights_only=True))
        self.optimizer      = self._make_optimizer()
        self.scaler         = torch.amp.GradScaler()
        self.strategies     = []
        self.eval_log       = []
        self.tuner          = None
        self.detector       = ADWIN(delta=self.adwin_delta)
        self.drift_detected = False
        self.drift_batch    = None
        self._recent_batches.clear()

    #   encoding 

    def _encode(self, batch, label_map):
        enc = self.tokenizer(
            batch["text"].tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        labels = torch.tensor(
            [label_map[l] for l in batch["label"].tolist()],
            dtype=torch.long,
            device=self.device,
        )
        return enc, labels

    #    evaluation 

    def _run_eval(self, val_stream, label_map, return_preds=False):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_stream:
                enc, labels = self._encode(batch, label_map)
                preds = self.model(**enc).logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        self.model.train()

        p, l = torch.cat(all_preds), torch.cat(all_labels)
        return (p, l) if return_preds else (p == l).float().mean().item()

    def evaluate(self, val_stream, label_map):
        return self._run_eval(val_stream, label_map, return_preds=True)

    #    drift detection 

    def _update_detector(self, preds, labels, batch_idx):
        # per-batch mean is more stable than per-sample binary values
        # binary feed caused ADWIN to trigger during the learning phase
        error_rate = (preds != labels).float().mean().item()
        self.detector.update(error_rate)

        if self.detector.drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.drift_batch    = batch_idx
            print(f"\n[ADWIN] Drift detected at batch {batch_idx}")

            # decay LR so the model doesn't overwrite old knowledge too aggressively
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * self.lr_decay
            print(f"[ADWIN] LR decayed by {self.lr_decay}x")

            if self.tuner is not None:
                self.tuner.tune(self, list(self._recent_batches))
            self.on_task_end()

    #    training loop 

    def train(self, stream, label_map, val_streams: dict, eval_every=50):
        """
        Returns a dict with 'task' (CE on current batch only) and 'total'
        (CE + strategy contributions) loss curves for plotting.
        """
        self._last_label_map = label_map
        self.model.train()
        task_losses, total_losses = [], []

        for i, batch in enumerate(tqdm(stream, desc="Training")):
            enc, labels = self._encode(batch, label_map)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out        = self.model(**enc, labels=labels)
                total_loss = out.loss + sum(
                    s.compute_loss(self, enc, labels, out.logits)
                    for s in self.strategies
                )

            # set_to_none is faster than zero_(), skips the memset entirely
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            # clip before the optimizer step so spikes don't corrupt the weights
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            #track both losses — task loss is the clean signal for plotting,
            # total loss includes replay/regularization overhead
            task_losses.append(out.loss.item())
            total_losses.append(total_loss.item())

            with torch.no_grad():
                preds = out.logits.detach().argmax(dim=1)
            self._update_detector(preds, labels, i)

            self._recent_batches.append(batch)

            if i % eval_every == 0:
                entry = {"batch": i}
                for name, vs in val_streams.items():
                    entry[name] = self._run_eval(vs, label_map)
                self.eval_log.append(entry)

        # one final eval at the end of the stream regardless of eval_every
        entry = {"batch": len(stream)}
        for name, vs in val_streams.items():
            entry[name] = self._run_eval(vs, label_map)
        self.eval_log.append(entry)

        return {"task": task_losses, "total": total_losses}